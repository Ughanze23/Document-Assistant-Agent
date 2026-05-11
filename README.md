# Document Assistant

A multi-agent document processing assistant built with LangChain and LangGraph. It answers questions, summarizes documents, and performs calculations on financial and healthcare documents using a conversational interface.

## What it does

The assistant classifies each user request and routes it to one of three specialised agents:

- **Q&A Agent** — answers specific questions about document content with source citations
- **Summarization Agent** — extracts key points and generates summaries
- **Calculation Agent** — performs mathematical operations on numerical document data

## Agent Architecture

![LangGraph Agent Architecture](./docs/langgraph_agent_architecture.png)

## Project Structure

```
├── main.py               # Entry point and interactive CLI
├── src/
│   ├── agent.py          # LangGraph workflow, agent nodes, and graph definition
│   ├── assistant.py      # DocumentAssistant class — session management and workflow orchestration
│   ├── retrieval.py      # Document store and retrieval strategies
│   ├── schemas.py        # Pydantic models for structured agent outputs
│   ├── prompts.py        # Prompt templates for each agent type
│   └── tools.py          # LangChain tools (document search, reader, calculator)
├── docs/                 # Architecture diagrams
├── requirements.txt
└── .env.example
```

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`:

```
OPENAI_API_KEY=your-key-here
```

## Running

```bash
python main.py
```

The CLI will prompt for a user ID, then accept natural language queries. Available commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands and example queries |
| `/docs` | List all documents in the system |
| `/quit` | Exit the assistant |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `MODEL_NAME` | `gpt-4o` | LLM model to use |
| `TEMPERATURE` | `0.1` | Sampling temperature |
| `SESSION_STORAGE_PATH` | `./sessions` | Where session files are saved |

---

## Design Decisions

### Intent routing over a single agent

Each request is first classified into `qa`, `summarization`, `calculation`, or `unknown` before any tool is called. This keeps each agent's system prompt narrow and focused — the Q&A prompt emphasises source citation, the summarization prompt emphasises key-point extraction, and the calculation prompt emphasises numeric precision. A single general-purpose agent would require a larger, vaguer prompt that degrades reliability on all three task types.

`unknown` intent falls back to the Q&A agent rather than returning an error, because an attempt at an answer is more useful than a refusal when intent is ambiguous.

### ReAct agent per node

Each of the three task agents uses LangGraph's `create_react_agent`. This gives each agent its own tool-call / observation loop, so it can issue multiple document searches or calculations before producing a final answer. A single LLM call with tools would not allow the model to react to intermediate results.

All four tools (`document_search`, `document_reader`, `document_statistics`, `calculator`) are available to every agent. Restricting tools by agent type was considered but rejected — a summarization request may still need the calculator for percentage breakdowns, and cross-tool availability keeps the routing logic simple.

### Error handling

- **Calculator**: the expression is validated against an allowlist of characters (`[0-9+\-*/%().\s]`) and a blocklist of dangerous keywords before `eval` is called with an empty builtins namespace. This prevents code injection while keeping the tool dependency-free.
- **Tool errors**: each tool catches all exceptions and returns an error string. This lets the ReAct loop see the error as an observation and either retry with a corrected call or surface it in the final answer, rather than raising an unhandled exception that would abort the entire workflow.
- **Workflow errors**: `process_message` wraps the workflow invocation in a `try/except` and always runs `_save_session()` in the `finally` block, so a mid-run failure never leaves the session in an unsaved state.

---

## State and Memory

### In-turn state (`AgentState`)

`AgentState` is a `TypedDict` that LangGraph passes between nodes within a single workflow invocation. It holds:

| Field | Type | Purpose |
|-------|------|---------|
| `user_input` | `str` | Raw text from the current turn |
| `messages` | `List[BaseMessage]` | Full message thread for the current agent (accumulated with `add_messages`) |
| `intent` | `UserIntent` | Classification result from `classify_intent` |
| `next_step` | `str` | Routing signal read by `should_continue` |
| `conversation_summary` | `str` | Rolling summary injected into agent prompts |
| `active_documents` | `List[str]` | Document IDs referenced in this turn |
| `current_response` | `dict` | Raw result dict returned by `invoke_react_agent` |
| `tools_used` | `List[str]` | Names of tools called during this turn |
| `actions_taken` | `List[str]` | Node names executed (accumulated with `add_messages`) |

`actions_taken` uses an `add_messages`-style reducer so values from each node are appended rather than overwritten. All other fields are replaced on each node update.

### Cross-turn memory (checkpointer)

The workflow is compiled with `MemorySaver` as its checkpointer. LangGraph persists the full `AgentState` snapshot keyed by `thread_id` (set to the session ID). On the next turn, `conversation_summary` and `active_documents` from the previous snapshot are loaded and injected into the new initial state, giving agents awareness of prior context without replaying the full message history.

The `update_memory` node runs after every agent turn. It sends the current message thread to the LLM and produces a rolling `conversation_summary` and a list of `active_documents`, which are written back into the state for the next turn.

### Session persistence (disk)

`DocumentAssistant` separately serialises `SessionState` to a JSON file under `./sessions/<session_id>.json` after every turn. This survives process restarts, unlike `MemorySaver` which is in-process only. On `start_session`, if a file exists for the given session ID it is loaded, restoring `conversation_history` and `document_context`.

**Scope and clearing:** each `thread_id` / session ID is fully isolated. There is no shared state across sessions. A session is effectively cleared by starting a new one with a different ID.

---

## Structured Outputs

### How schemas are applied

Every LLM call that must return structured data uses `llm.with_structured_output(SchemaClass)`, where `SchemaClass` is a Pydantic `BaseModel` defined in `src/schemas.py`. LangChain translates the Pydantic model into a JSON schema and passes it to the model as a tool or response format constraint, then deserialises the raw response into a validated Python object.

The schemas in use are:

| Schema | Used by | Key constraints |
|--------|---------|----------------|
| `UserIntent` | `classify_intent` | `intent_type` is a `Literal` of four values; `confidence` is `float` with `ge=0.0, le=1.0` |
| `AnswerResponse` | `qa_agent` | `confidence` bounded `0.0–1.0`; `sources` defaults to empty list |
| `SummarizationResponse` | `summarization_agent` | `key_points` is a non-optional list |
| `CalculationResponse` | `calculation_agent` | `result` is `float`; `expression` is required |
| `UpdateMemoryResponse` | `update_memory` | `document_ids` defaults to empty list |

### Validation

Pydantic validates every field on deserialisation. `Literal` fields reject any value outside the allowed set. Numeric bounds (`ge`, `le`) raise a `ValidationError` if the model returns a value out of range. `default_factory=list` ensures list fields are never `None` even if the model omits them.

### Failure handling

If the model returns a response that cannot be deserialised into the target schema (malformed JSON, missing required field, out-of-range value), LangChain raises a `ValidationError` or `OutputParserException`. This propagates to the `try/except` in `process_message`, which returns `{"success": False, "error": <message>}` to the caller. The session is still saved via the `finally` block so no state is lost.
