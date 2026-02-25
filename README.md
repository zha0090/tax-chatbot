# TaxGPT Financial Chatbot

A hybrid RAG chatbot that answers financial and tax questions using structured data, regulatory documents, and AI-powered retrieval. Built with Django, FAISS, NetworkX, and OpenAI.

---

## Quick links

| | |
|---|---|
| **Author** | **Alex Gong** |
| **Demo video** | [Watch on Loom](https://www.loom.com/share/46ae0a007bda4ccbb47f05c7c349268b) |
| **GitHub** | [DevLaiGer](https://github.com/DevLaiGer) |

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Quick start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Project structure](#4-project-structure)
5. [Design decisions](#5-design-decisions)
6. [Data sources](#6-data-sources)
7. [API](#7-api)
8. [Testing](#8-testing)
9. [Ingestion](#9-ingestion)
10. [Tech stack](#10-tech-stack)
11. [Cost](#11-cost)
12. [License](#12-license)

---

## 1. Prerequisites

- **Python** 3.12 or higher
- **OpenAI API key** - [Get one here](https://platform.openai.com/api-keys)
- **Disk space** - 500MB for vector store and data files

---

## 2. Quick start

```bash
# 1. Clone and set up
git clone https://github.com/DevLaiGer/taxGPT-chatbot.git && cd taxGPT-chatbot
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Place data files in refers/
# Download from the provided Google Drive link:
#   - tax_data.csv
#   - i1040gi.pdf
#   - usc26@118-78.pdf
#   - MIC_3e_Ch11.ppt

# 4. Initialize
python manage.py migrate
python manage.py ingest --clear

# 5. Run
python manage.py runserver
# Open http://localhost:8000
```

---

## 3. Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Query Router (GPT-4o-mini) │  Classifies intent, extracts entities
└──────────┬──────────────────┘
           │
    ┌──────┼──────────┐
    ▼      ▼          ▼
┌──────┐ ┌──────┐ ┌───────────┐
│Vector│ │Graph │ │Structured │
│Search│ │Search│ │  Lookup   │
│(FAISS)│ │(NX) │ │ (pandas)  │
└──┬───┘ └──┬───┘ └─────┬─────┘
   └─────────┼───────────┘
             ▼
    ┌────────────────┐
    │ Context Fusion  │
    └───────┬────────┘
            ▼
    ┌────────────────┐
    │  GPT-4o-mini   │  Generates answer from merged context
    └───────┬────────┘
            ▼
        Response
```

### Three retrieval lanes

| Lane | Backend | Best for | Example |
|------|---------|----------|---------|
| **Vector** | FAISS + OpenAI embeddings | Tax rules, IRS instructions, legal code, conceptual questions | "How do I report rental income on Form 1040?" |
| **Graph** | NetworkX knowledge graph | Comparisons, rankings, relationships between entities | "Which state has the highest tax rate for partnerships?" |
| **Structured** | pandas on the CSV | Exact numerical aggregations, filtered statistics | "What is the total tax owed by corporations in 2022?" |

The router uses GPT-4o-mini with few-shot examples to classify queries and extract entities (taxpayer type, state, year, income source, deduction type), then activates the appropriate lanes.

---

## 4. Project structure

```
├── config/                  # Django settings, URLs, WSGI
├── chat/                    # Chat API endpoint (/api/chat/)
│   └── views.py             # ChatView - wired to the pipeline
├── ingestion/               # Data ingestion pipeline
│   ├── parsers/
│   │   ├── csv_parser.py    # Row-level + summary chunking
│   │   ├── pdf_parser.py    # Page-aware overlapping chunks
│   │   └── ppt_parser.py    # .ppt (OLE binary) + .pptx support
│   ├── embedder.py          # OpenAI embedding wrapper with batching
│   └── management/commands/
│       └── ingest.py        # python manage.py ingest
├── retrieval/               # Search and retrieval
│   ├── vector_search.py     # FAISS vector store with persistence
│   ├── graph_builder.py     # Builds NetworkX graph from CSV
│   ├── graph_search.py      # Graph queries and context generation
│   ├── router.py            # GPT-based query classification
│   └── pipeline.py          # Orchestrates all retrieval lanes
├── frontend/
│   └── templates/chat.html  # Chat UI
├── tests/                   # 60 tests across all modules
│   ├── test_parsers.py
│   ├── test_vector_store.py
│   └── test_knowledge_graph.py
├── eval/
│   ├── eval_dataset.json    # Test Q&A pairs
│   └── run_eval.py          # Automated accuracy evaluation
└── refers/                  # Data files (not committed - see setup)
```

---

## 5. Design decisions

### Why hybrid retrieval instead of pure RAG?

My first instinct was a standard vector-only RAG pipeline, embed everything, search by similarity, feed context to the LLM. But once I looked at the actual data, I realized that wouldn't cut it. When someone asks "what's the average tax rate for corporations in CA?", they want a precise number computed from the CSV, not a fuzzy match against a text chunk that happens to mention California.

So I split retrieval into three lanes: vector search for unstructured documents (PDFs, PPT), a knowledge graph for entity relationships and comparisons, and pandas for exact numerical queries. A lightweight LLM router classifies each query and picks the right lane(s). It adds a small amount of latency, but the accuracy improvement is significant, the system gives deterministic answers for structured questions and semantic answers for everything else.

### Why FAISS over ChromaDB?

I actually started with ChromaDB since it's the go-to for quick RAG prototypes. But it broke immediately on Python 3.14 due to a pydantic v1 dependency conflict. Rather than pinning to an older Python version, I switched to FAISS. It's dependency-light, Meta battle-tested it at billion-scale, and it gives me the same cosine similarity search. I wrapped it with a simple JSON sidecar for metadata storage, which keeps things self-contained, no external service, no version conflicts.

### Why NetworkX over Neo4j?

With 5,000 CSV records mapping to roughly 30 entity nodes and 290 edges, spinning up a Neo4j instance felt like overkill. NetworkX loads the whole graph into memory in milliseconds, handles all the traversals I need (comparisons, rankings, neighborhood lookups), and serializes to a pickle file. If this were millions of records with complex relationship queries, I'd absolutely reach for Neo4j. But for this scope, an in-process graph keeps the project self-contained, `pip install` and you're done, no database servers to manage.

### Why GPT-4o-mini for routing?

I considered regex-based routing, but tax queries come in too many forms, "what's the avg corp tax in CA" vs "average tax rate for corporations in California" vs "how much do corps pay in CA". Writing rules for all those variants is brittle. GPT-4o-mini handles this naturally, extracts structured entities from free-text, and costs fractions of a cent per query. The few-shot prompt keeps the output format consistent (always valid JSON), and the ~200ms it adds is barely noticeable in practice.

### Chunking strategy

I tailored chunking per data source instead of using a one-size-fits-all approach:

- **CSV**: Each row becomes a natural-language sentence (e.g., "Corporation in California, 2021: income $85,000, tax rate 24.5%..."), plus I pre-compute group summaries by state, taxpayer type, etc. so aggregate questions hit meaningful chunks instead of random rows.
- **PDF**: Overlapping character-based chunks (1000 chars, 200 overlap). The overlap is important, without it, questions that span a page break or section boundary get lost.
- **PPT**: One chunk per slide, with template boilerplate filtered out. Slides are naturally self-contained units.

### Legacy .ppt handling

The `MIC_3e_Ch11.ppt` file is a pre-2007 binary PowerPoint, `python-pptx` only handles `.pptx`. I didn't want to require LibreOffice or any OS-level tooling just for one file, so I dug into the OLE compound document format and wrote a parser that reads `TextBytesAtom` and `TextCharsAtom` records directly from the PowerPoint Document stream. It's low-level, but it works on any OS with zero extra dependencies.

---

## 6. Data sources

| File | Type | Content | Records/Pages |
|------|------|---------|---------------|
| `tax_data.csv` | CSV | Tax transactions: taxpayer types, states, income, deductions, tax rates | 5,000 rows |
| `i1040gi.pdf` | PDF | IRS Form 1040 Instructions (2023) | 114 pages |
| `usc26@118-78.pdf` | PDF | US Internal Revenue Code (Title 26) | 7,058 pages |
| `MIC_3e_Ch11.ppt` | PPT | Microeconomics Ch. 11: Taxes and Tax Policy | 20 slides |

---

## 7. API

### POST /api/chat/

```json
// Request
{ "query": "What is the average tax rate for corporations in California?" }

// Response
{
  "query": "What is the average tax rate for corporations in California?",
  "answer": "The average tax rate for corporations in California is 23.95%.",
  "sources": ["tax_data.csv", "knowledge_graph"],
  "routing": {
    "lanes": ["graph", "structured"],
    "entities": { "taxpayer_type": "Corporation", "state": "CA" }
  }
}
```

### GET /api/health/

Returns `{"status": "ok"}`.

---

## 8. Testing

```bash
# Run all tests (60 tests)
python -m pytest tests/ -v

# By module
python -m pytest tests/test_parsers.py -v           # Parsers (23 tests)
python -m pytest tests/test_vector_store.py -v     # Embeddings + FAISS (13 tests)
python -m pytest tests/test_knowledge_graph.py -v  # Knowledge graph (24 tests)

# Run the evaluation suite
python eval/run_eval.py
```

**Note:** Vector store tests require a valid `OPENAI_API_KEY` in `.env`. Parser and graph tests run without an API key.

---

## 9. Ingestion

```bash
python manage.py ingest --clear              # All sources (fresh)
python manage.py ingest --source csv         # CSV only (~2 min)
python manage.py ingest --source pdf_1040    # 1040 PDF (~5 min)
python manage.py ingest --source ppt         # PPT (~30 sec)
python manage.py ingest --source pdf_irc --irc-max-pages 100  # IRC subset
```

The ingest command parses each file, generates OpenAI embeddings, stores vectors in FAISS, and builds the NetworkX knowledge graph from the CSV data.

---

## 10. Tech stack

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | Django + DRF | 6.0 |
| LLM | OpenAI GPT-4o-mini | - |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim |
| Vector store | FAISS (faiss-cpu) | 1.13 |
| Knowledge graph | NetworkX | 3.6 |
| Data processing | pandas, pdfplumber, python-pptx, olefile | - |
| Testing | pytest + pytest-django | - |

---

## 11. Cost

The entire project runs on OpenAI API credits. Approximate costs:

| Operation | Tokens | Cost |
|-----------|--------|------|
| Embed all datasets | ~5M tokens | ~$0.10 |
| 100 chat queries | ~500K tokens | ~$0.15 |
| **Total for full setup + testing** | | **< $0.50** |

---

## 12. License

This project was built as a technical assessment and is not licensed for commercial use.
