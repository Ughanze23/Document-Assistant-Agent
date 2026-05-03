# Production Retrieval Guide

This document describes the production implementation that would replace `SimulatedRetriever` in `src/retrieval.py`. It covers two concerns: the PDF extraction pipeline that produces the `content` field, and the relational database schema that stores and indexes it.

---

## Part 1: PDF Extraction Pipeline

### Overview

Every document in the system originates as a PDF. Before anything reaches the database, the PDF passes through three stages: extraction, cleaning, and structured field parsing.

```
PDF file
   └─→ Raw text extraction
          └─→ Text cleaning
                 └─→ LLM structured field parsing
                            └─→ INSERT into documents table
```

---

### Stage 1: Raw Text Extraction

For PDFs with embedded text (most digital invoices, contracts, and claims), use `pdfplumber`:

```python
import pdfplumber

def extract_raw_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)
```

For scanned PDFs (image-only, no embedded text layer), extraction requires OCR. AWS Textract is the recommended option for financial documents because it understands table layouts — critical for invoices with line-item grids:

```python
import boto3

def extract_via_textract(pdf_path: str) -> str:
    client = boto3.client("textract")
    with open(pdf_path, "rb") as f:
        response = client.detect_document_text(Document={"Bytes": f.read()})
    blocks = [b["Text"] for b in response["Blocks"] if b["BlockType"] == "LINE"]
    return "\n".join(blocks)
```

Use `pdfplumber` first. Fall back to Textract only if `pdfplumber` returns an empty or near-empty string, which signals a scanned document.

---

### Stage 2: Text Cleaning

Raw extraction output is noisy. Common problems include collapsed spacing, broken words across line breaks, and inconsistent currency formatting. Clean before storing:

```python
import re

def clean_extracted_text(raw: str) -> str:
    # Collapse multiple spaces and tabs into one
    text = re.sub(r'[ \t]+', ' ', raw)

    # Fix hyphenated line breaks (e.g. "Cor-\nporation" → "Corporation")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse more than two consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalise currency: "$5,000.00" and "5000" both become "5000.00"
    text = re.sub(r'\$\s*(\d+),(\d{3})', r'\1\2', text)

    return text.strip()
```

The cleaned output is what gets written to the `content` column. It must be human-readable and search-friendly — no artefacts that would cause full-text search to miss legitimate matches.

---

### Stage 3: Structured Field Extraction

Rather than relying on regex to pull out fields like `total`, `client`, or `date`, pass the cleaned text to an LLM. This produces the structured data that populates the `metadata` JSONB column and the typed columns like `doc_type`.

```python
import anthropic
import json

client = anthropic.Anthropic()

EXTRACTION_PROMPT = """
Extract the following fields from the document text and return valid JSON only.
If a field is not present, use null.

Fields:
- doc_type: one of "invoice", "contract", "claim"
- title: document title or identifier
- client: client or counterparty name
- date: document date in YYYY-MM-DD format
- total: total monetary amount as a number (no currency symbols)
- status: document status if present

Document:
{content}
"""

def extract_structured_fields(cleaned_text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(content=cleaned_text)
        }]
    )
    return json.loads(response.content[0].text)
```

The returned dict maps directly to the `metadata` column and the typed columns in the schema described in Part 2.

---

### Full Pipeline Function

```python
def ingest_pdf(pdf_path: str, doc_id: str) -> dict:
    raw = extract_raw_text(pdf_path)
    if len(raw.strip()) < 50:               # likely a scanned PDF
        raw = extract_via_textract(pdf_path)

    cleaned = clean_extracted_text(raw)
    fields = extract_structured_fields(cleaned)

    return {
        "doc_id": doc_id,
        "content": cleaned,
        "doc_type": fields.get("doc_type"),
        "title": fields.get("title"),
        "metadata": fields
    }
```

The original PDF binary is stored in S3 (or equivalent object storage) keyed by `doc_id`. It is never written to the database — only the extracted and cleaned text lives in the `content` column.

---

## Part 2: Relational Database Schema

### Table Definition

```sql
CREATE TABLE documents (
    doc_id       TEXT        PRIMARY KEY,
    title        TEXT        NOT NULL,
    content      TEXT        NOT NULL,
    doc_type     TEXT        NOT NULL CHECK (doc_type IN ('invoice', 'contract', 'claim')),
    metadata     JSONB       NOT NULL DEFAULT '{}',

    -- Typed columns extracted from metadata for indexed filtering
    client       TEXT,
    doc_date     DATE,
    total_amount NUMERIC(15, 2),
    status       TEXT,

    -- Full-text search vector, auto-maintained by PostgreSQL
    search_vec   tsvector    GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                     setweight(to_tsvector('english', coalesce(content, '')), 'B')
                 ) STORED,

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Why typed columns alongside `metadata` JSONB?**

`metadata` stores the full extraction output and is flexible enough to hold any field. But range queries on JSONB numerics (`metadata->>'total'`) cannot use a standard B-tree index efficiently. Promoting `total_amount`, `doc_date`, and `client` into typed columns allows the database to index them properly.

---

### Indexes

```sql
-- Full-text search
CREATE INDEX idx_documents_search ON documents USING GIN (search_vec);

-- Metadata filtering
CREATE INDEX idx_documents_doc_type    ON documents (doc_type);
CREATE INDEX idx_documents_total       ON documents (total_amount);
CREATE INDEX idx_documents_client      ON documents (client);
CREATE INDEX idx_documents_date        ON documents (doc_date);

-- Arbitrary JSONB field queries
CREATE INDEX idx_documents_metadata   ON documents USING GIN (metadata);
```

---

### Mapping `SimulatedRetriever` Methods to SQL

#### `retrieve_by_keyword`

```sql
SELECT
    doc_id,
    title,
    content,
    doc_type,
    metadata,
    ts_rank(search_vec, query) AS relevance_score
FROM
    documents,
    websearch_to_tsquery('english', $1) AS query
WHERE
    search_vec @@ query
ORDER BY
    relevance_score DESC
LIMIT $2;
```

`websearch_to_tsquery` accepts natural language input (`"acme invoice"`) and handles operator parsing automatically — no manual keyword splitting needed.

#### `retrieve_by_type`

```sql
SELECT doc_id, title, content, doc_type, metadata, 1.0 AS relevance_score
FROM documents
WHERE doc_type = $1;
```

#### `retrieve_by_amount_range`

```sql
SELECT doc_id, title, content, doc_type, metadata, 1.0 AS relevance_score
FROM documents
WHERE total_amount >= $1   -- omit clause if min_amount is None
  AND total_amount <= $2   -- omit clause if max_amount is None
ORDER BY total_amount DESC;
```

#### `retrieve_by_approximate_amount`

```sql
SELECT
    doc_id,
    title,
    content,
    doc_type,
    metadata,
    1.0 - (ABS(total_amount - $1) / ($1 * $2 / 100.0)) AS relevance_score
FROM documents
WHERE total_amount BETWEEN ($1 * (1 - $2/100.0)) AND ($1 * (1 + $2/100.0))
ORDER BY relevance_score DESC;
-- $1 = target amount, $2 = tolerance percentage
```

#### `get_document_by_id`

```sql
SELECT doc_id, title, content, doc_type, metadata, 1.0 AS relevance_score
FROM documents
WHERE doc_id = $1;
```

#### `get_statistics`

```sql
SELECT
    COUNT(*)                        AS total_documents,
    COUNT(total_amount)             AS documents_with_amounts,
    SUM(total_amount)               AS total_amount,
    AVG(total_amount)               AS average_amount,
    MIN(total_amount)               AS min_amount,
    MAX(total_amount)               AS max_amount,
    jsonb_object_agg(doc_type, cnt) AS document_types
FROM documents
LEFT JOIN (
    SELECT doc_type, COUNT(*) AS cnt
    FROM documents
    GROUP BY doc_type
) type_counts USING (doc_type);
```

---

### Insert

The output of `ingest_pdf` maps directly to an insert:

```python
import psycopg2

def store_document(conn, doc: dict):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (doc_id, title, content, doc_type, metadata, client, total_amount)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE
                SET content      = EXCLUDED.content,
                    metadata     = EXCLUDED.metadata,
                    total_amount = EXCLUDED.total_amount;
        """, (
            doc["doc_id"],
            doc["title"],
            doc["content"],
            doc["doc_type"],
            json.dumps(doc["metadata"]),
            doc["metadata"].get("client"),
            doc["metadata"].get("total")
        ))
    conn.commit()
```

The `ON CONFLICT ... DO UPDATE` clause makes re-ingesting a corrected PDF idempotent.

---

## Dependency Summary

| Purpose | Library / Service |
|---|---|
| PDF text extraction | `pdfplumber` |
| OCR for scanned PDFs | AWS Textract |
| Text cleaning | Python `re` (stdlib) |
| Structured field extraction | Anthropic SDK (`claude-sonnet-4-6`) |
| Database | PostgreSQL 15+ |
| Python DB driver | `psycopg2` or `asyncpg` |
| PDF binary storage | AWS S3 or equivalent |
