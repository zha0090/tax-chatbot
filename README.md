# TaxGPT Financial Chatbot

A hybrid RAG chatbot that answers financial and tax questions using structured data, regulatory documents, and AI-powered retrieval. Built with Django, FAISS, NetworkX, and OpenAI.

## Prerequisites

- Python 3.12 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- ~500MB disk space for vector store and data files

## Quick Start

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

## Architecture

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

### Three Retrieval Lanes

The system routes each query to one or more retrieval strategies based on the question type:

| Lane | Backend | Best For | Example |
|------|---------|----------|---------|
| **Vector** | FAISS + OpenAI embeddings | Tax rules, IRS instructions, legal code, conceptual questions | "How do I report rental income on Form 1040?" |
| **Graph** | NetworkX knowledge graph | Comparisons, rankings, relationships between entities | "Which state has the highest tax rate for partnerships?" |
| **Structured** | pandas on the CSV | Exact numerical aggregations, filtered statistics | "What is the total tax owed by corporations in 2022?" |

The router uses GPT-4o-mini with few-shot examples to classify queries and extract entities (taxpayer type, state, year, income source, deduction type), then activates the appropriate lanes.

## Project Structure

```
├── config/                  # Django settings, URLs, WSGI
├── chat/                    # Chat API endpoint (/api/chat/)
│   └── views.py             # ChatView — wired to the pipeline
├── ingestion/               # Data ingestion pipeline
│   ├── parsers/
│   │   ├── csv_parser.py    # Row-level + summary chunking
│   │   ├── pdf_parser.py    # Page-aware overlapping chunks
│   │   └── ppt_parser.py    # .ppt (OLE binary) + .pptx support
│   ├── embedder.py          # OpenAI embedding wrapper with batching
│   └── management/commands/
│       └── ingest.py        # `python manage.py ingest`
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
└── refers/                  # Data files (not committed — see setup)
```

## Design Decisions

### Why hybrid retrieval instead of pure RAG?

Financial data is inherently multi-modal. A question like "average tax rate for corporations in CA" requires exact computation over structured data — vector similarity search would return approximate matches from nearby text chunks, not the precise answer. By routing structured/numerical queries to pandas and relationship queries to the knowledge graph, the system achieves deterministic accuracy where it matters.

### Why FAISS over ChromaDB?

ChromaDB's dependency on pydantic v1 is incompatible with Python 3.14. FAISS is dependency-light, battle-tested at scale (developed by Meta), and provides the same cosine similarity search with zero configuration overhead. The `VectorStore` class wraps FAISS with a JSON sidecar for metadata, giving us filtering and persistence without an external service.

### Why NetworkX over Neo4j?

For 5,000 records with ~30 entity nodes and ~290 edges, an in-process graph is faster and simpler than spinning up a database server. NetworkX loads the entire graph into memory in milliseconds, supports all the traversal patterns we need, and serializes to a pickle file. If the dataset scaled to millions of records, Neo4j would be the right choice — but for this scope, NetworkX keeps the system self-contained with zero infrastructure.

### Why GPT-4o-mini for routing?

The query router needs to understand natural language intent and extract structured entities — a task that's awkward with regex but trivial for an LLM. GPT-4o-mini adds ~200ms latency and costs fractions of a cent per query, while providing robust classification across diverse phrasings. The few-shot prompt ensures consistent JSON output.

### Chunking strategy

Each data source gets a tailored chunking approach:
- **CSV**: Each row becomes a natural-language sentence, plus pre-computed group summaries (by state, taxpayer type, etc.) for aggregate queries
- **PDF**: Overlapping character-based chunks (1000 chars, 200 overlap) that preserve section context across chunk boundaries
- **PPT**: One chunk per slide, with template/placeholder text filtered out

### Legacy .ppt handling

The `MIC_3e_Ch11.ppt` file uses the pre-2007 binary PowerPoint format. Rather than requiring LibreOffice or PowerPoint for conversion, the parser reads the OLE compound document directly and extracts text from `TextBytesAtom` and `TextCharsAtom` records in the PowerPoint Document stream — a zero-dependency approach that works on any OS.

## Data Sources

| File | Type | Content | Records/Pages |
|------|------|---------|---------------|
| `tax_data.csv` | CSV | Tax transactions: taxpayer types, states, income, deductions, tax rates | 5,000 rows |
| `i1040gi.pdf` | PDF | IRS Form 1040 Instructions (2023) | 114 pages |
| `usc26@118-78.pdf` | PDF | US Internal Revenue Code (Title 26) | 7,058 pages |
| `MIC_3e_Ch11.ppt` | PPT | Microeconomics Ch. 11: Taxes and Tax Policy | ~20 slides |

## API

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

## Testing

```bash
# Run all tests (60 tests)
python -m pytest tests/ -v

# By module
python -m pytest tests/test_parsers.py -v          # Parsers (23 tests)
python -m pytest tests/test_vector_store.py -v      # Embeddings + FAISS (13 tests)
python -m pytest tests/test_knowledge_graph.py -v   # Knowledge graph (24 tests)

# Run the evaluation suite
python eval/run_eval.py
```

Note: Vector store tests require a valid `OPENAI_API_KEY` in `.env`. Parser and graph tests run without an API key.

## Ingestion

```bash
python manage.py ingest --clear              # All sources (fresh)
python manage.py ingest --source csv         # CSV only (~2 min)
python manage.py ingest --source pdf_1040    # 1040 PDF (~5 min)
python manage.py ingest --source ppt         # PPT (~30 sec)
python manage.py ingest --source pdf_irc --irc-max-pages 100  # IRC subset
```

The ingest command parses each file, generates OpenAI embeddings, stores vectors in FAISS, and builds the NetworkX knowledge graph from the CSV data.

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | Django + DRF | 6.0 |
| LLM | OpenAI GPT-4o-mini | — |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim |
| Vector Store | FAISS (faiss-cpu) | 1.13 |
| Knowledge Graph | NetworkX | 3.6 |
| Data Processing | pandas, pdfplumber, python-pptx, olefile | — |
| Testing | pytest + pytest-django | — |

## Cost

The entire project runs on OpenAI API credits. Approximate costs:

| Operation | Tokens | Cost |
|-----------|--------|------|
| Embed all datasets | ~5M tokens | ~$0.10 |
| 100 chat queries | ~500K tokens | ~$0.15 |
| **Total for full setup + testing** | | **< $0.50** |

## Author

Alex Gong
GitHub: [DevLaiGer](https://github.com/DevLaiGer)

## License

This project was built as a technical assessment and is not licensed for commercial use.
