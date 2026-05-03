# Retrieval Evaluation Guide

This document describes how to evaluate the quality and accuracy of both the PDF extraction pipeline and the retrieval system described in `docs/production_retrieval_guide.md`, covering metrics, test strategies, and when to run each check.

---

## Overview

There are three distinct layers to evaluate:

1. **PDF extraction pipeline** — does the text coming out of the PDF accurately represent the source document?
2. **Retrieval system** — does the database return the right documents for a given query?
3. **End-to-end agent responses** — is the final answer faithful and correct relative to the retrieved documents?

Each layer needs different metrics and test strategies.

---

## Part 1: Evaluating the PDF Extraction Pipeline

### 1.1 Extraction Accuracy

The goal is confirming that `extract_raw_text` captures all meaningful text without loss or corruption. Build a **golden dataset**: a set of PDFs where the ground truth text is manually recorded, then compare extraction output against it.

The metric is **Character Error Rate (CER)**:

```python
import editdistance

def character_error_rate(ground_truth: str, extracted: str) -> float:
    return editdistance.eval(ground_truth, extracted) / len(ground_truth)
```

**Acceptable thresholds:**
- Digital PDFs via `pdfplumber`: CER < 2%
- Scanned PDFs via AWS Textract: CER < 8% (varies with scan quality)

Flag any document exceeding its threshold for manual review before it enters the database.

---

### 1.2 Cleaning Effectiveness

Test that `clean_extracted_text` resolves the specific artefacts it targets without destroying legitimate content. These are deterministic unit tests — run them on every change to the cleaning function:

```python
cases = [
    ("Cor-\nporation",  "Corporation"),    # hyphenated line break
    ("$5,000.00",       "5000.00"),         # currency normalisation
    ("Acme  Corp",      "Acme Corp"),       # double spaces
    ("line1\n\n\n\nline2", "line1\n\nline2") # excessive newlines
]

for raw, expected in cases:
    result = clean_extracted_text(raw)
    assert result == expected, f"Expected {expected!r}, got {result!r}"
```

---

### 1.3 Structured Field Extraction Accuracy

For the LLM extraction step, build a labelled set of 50–100 documents where the correct field values are known, then measure **field-level accuracy** per document type:

```python
def field_accuracy(predictions: list[dict], ground_truth: list[dict], field: str) -> float:
    correct = sum(
        p.get(field) == g.get(field)
        for p, g in zip(predictions, ground_truth)
    )
    return correct / len(predictions)
```

**Evaluate separately for:**
- Each field: `doc_type`, `title`, `client`, `date`, `total_amount`
- Each document type: invoice, contract, claim

**Known failure modes to watch for:**
- `total_amount`: LLM picks up a subtotal or pre-tax amount instead of the final total
- `date`: ambiguous formats (`01/02/2024` — day-first or month-first?)
- `doc_type`: near-certain correct; any failure is a prompt issue

Re-run this evaluation whenever the extraction prompt or LLM model changes.

---

## Part 2: Evaluating the Retrieval System

### 2.1 Building a Retrieval Test Set

You need **query–relevance pairs**: queries where the correct documents are known in advance. For this project, that means annotating which documents should be returned for queries like:

- "invoices over $50,000" → `{INV-002, INV-003}`
- "Healthcare Partners contract" → `{CON-001}`
- "medical expense claim" → `{CLM-001}`

In production with a large document set, source queries from user logs and use human annotators to label relevance.

---

### 2.2 Keyword Retrieval Metrics

**Precision@K** — of the top K results returned, what fraction are actually relevant:

```python
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k
```

**Recall@K** — of all relevant documents, what fraction appear in the top K:

```python
def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant)
```

**Mean Reciprocal Rank (MRR)** — measures how high the first correct result ranks, averaged across queries. The most important metric for a document assistant where users expect the right document near the top:

```python
def mrr(results: list[list], relevant: list[set]) -> float:
    scores = []
    for retrieved, rel in zip(results, relevant):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in rel:
                scores.append(1 / rank)
                break
        else:
            scores.append(0)
    return sum(scores) / len(scores)
```

**Target baselines:**
- Precision@3 > 0.80
- MRR > 0.85

---

### 2.3 Filter Correctness

Amount and type filters (`retrieve_by_amount_range`, `retrieve_by_type`) have deterministic correct answers. Test them with exact set assertions — any failure is a bug, not a quality issue:

```python
# Type filter
results = retriever.retrieve_by_type("invoice")
assert {r.doc_id for r in results} == {"INV-001", "INV-002", "INV-003"}

# Amount range filter
results = retriever.retrieve_by_amount_range(min_amount=50000)
assert {r.doc_id for r in results} == {"INV-002", "INV-003", "CON-001"}

# Approximate amount filter
results = retriever.retrieve_by_approximate_amount(amount=70000, percentage=10)
assert "INV-002" in {r.doc_id for r in results}  # INV-002 total is $69,300

# Exact amount filter
results = retriever.retrieve_by_exact_amount(amount=2450)
assert {r.doc_id for r in results} == {"CLM-001"}
```

---

## Part 3: Evaluating End-to-End Response Quality

Even if extraction and retrieval score well individually, the final agent response can still be wrong. Use **LLM-as-judge** evaluation to score responses against their source documents:

```python
import anthropic
import json

client = anthropic.Anthropic()

EVAL_PROMPT = """
You are evaluating a document assistant response.

Question: {question}
Retrieved documents: {documents}
Assistant response: {response}

Score the response on each dimension from 1 (poor) to 5 (excellent):
1. Faithfulness: Is every claim in the response supported by the retrieved documents?
2. Completeness: Does the response fully answer the question asked?
3. Correctness: Are all numbers, names, and dates accurate?

Return JSON only:
{{"faithfulness": N, "completeness": N, "correctness": N, "reasoning": "brief explanation"}}
"""

def evaluate_response(question: str, documents: list[str], response: str) -> dict:
    result = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": EVAL_PROMPT.format(
                question=question,
                documents="\n---\n".join(documents),
                response=response
            )
        }]
    )
    return json.loads(result.content[0].text)
```

Run this across a fixed set of test questions after any change to prompts, retrieval logic, or the LLM model. A drop in `faithfulness` after a prompt change is a regression signal.

**Minimum acceptable scores per dimension: 4/5**

---

## Part 4: Regression Testing Strategy

Combine all three evaluation layers into a single regression suite that runs before any production deployment:

```python
def run_regression_suite():
    results = {
        "extraction_cer":        evaluate_extraction_cer(golden_dataset),
        "field_accuracy":        evaluate_field_extraction(labelled_docs),
        "retrieval_precision":   evaluate_retrieval_precision(query_relevance_pairs),
        "retrieval_mrr":         evaluate_retrieval_mrr(query_relevance_pairs),
        "filter_assertions":     run_filter_unit_tests(),
        "llm_judge_scores":      evaluate_end_to_end(test_questions)
    }
    return results
```

Store results alongside each deployment. If any metric drops below its baseline, block the deployment and investigate the specific failure layer before proceeding.

---

## Summary: What to Measure and When

| Stage | Metric | Trigger |
|---|---|---|
| PDF extraction | Character Error Rate | When adding new document types |
| Text cleaning | Unit test assertions | Every code change |
| LLM field extraction | Field-level accuracy per doc type | When changing the extraction prompt or model |
| Keyword retrieval | Precision@3, MRR | Every retrieval logic change |
| Filter retrieval | Exact set assertions | Every code change |
| End-to-end responses | LLM-as-judge faithfulness / completeness / correctness | Before every production deployment |
