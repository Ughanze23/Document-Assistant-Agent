# Document Assistant

A multi-agent document processing assistant built with LangChain and LangGraph. It answers questions, summarizes documents, and performs calculations on financial and healthcare documents using a conversational interface.

## What it does

The assistant classifies each user request and routes it to one of three specialised agents:

- **Q&A Agent** — answers specific questions about document content with source citations
- **Summarization Agent** — extracts key points and generates summaries
- **Calculation Agent** — performs mathematical operations on numerical document data

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
