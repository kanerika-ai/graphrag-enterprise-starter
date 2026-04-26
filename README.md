# GraphRAG vs Flat RAG — Enterprise Knowledge Base Comparison

A working implementation comparing **standard vector RAG** against **graph-enhanced RAG** on corporate knowledge base. Demonstrates when graph RAG provides meaningfully better answers and when flat RAG is sufficient.

**100% open-source, runs entirely locally** — uses HuggingFace `transformers` to run LLMs on-device (no API key, no credits needed) and sentence-transformers for embeddings. Models are downloaded once and cached locally.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              16 PDF Documents (data/pdfs/)                       │
│   (Engineering, Sales, AI Research, HR, Finance, Security)      │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
        ┌──────▼──────┐                ┌──────▼───────┐
        │ PDF Ingestion│                │ PDF Ingestion │
        │ (pdfplumber) │                │ (pdfplumber)  │
        └──────┬──────┘                └──────┬───────┘
               │                              │
       ┌───────▼───────┐          ┌───────────▼──────────┐
       │  Flat Vector   │          │   Graph-Enhanced      │
       │     RAG        │          │       RAG             │
       │                │          │                       │
       │ 1. Chunk docs  │          │ 1. Local LLM extracts │
       │ 2. Embed with  │          │    entities &         │
       │    MiniLM      │          │    relationships      │
       │ 3. Store in    │          │ 2. Build NetworkX     │
       │    ChromaDB    │          │    knowledge graph    │
       │ 4. Top-k       │          │ 3. Detect communities │
       │    similarity  │          │ 4. Graph traversal    │
       │ 5. Local LLM   │          │    + vector search    │
       │    generates   │          │ 5. Local LLM generates│
       └───────┬───────┘          └───────────┬──────────┘
               │                              │
       ┌───────▼──────────────────────────────▼──────┐
       │         LLM-as-Judge Evaluation              │
       │   (completeness + accuracy, scored 1-5)      │
       └──────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Optional) Choose a model in .env
cp .env.example .env
# Default is Qwen/Qwen2.5-1.5B-Instruct — no token needed

# 3. Generate the PDF dataset
python data/generate_pdfs.py

# 4. Run the full comparison (10 questions)
#    First run downloads the model (~3 GB) then runs entirely offline
python run_comparison.py

# 5. Or try a single question
python demo_single_question.py "Who does Dr. Anika Patel collaborate with?"
```

### Supported Local Models

Any HuggingFace instruction-tuned model works. Set `HF_MODEL` in `.env`:

| Model | Disk | CPU speed | Quality | HF model ID |
|---|---|---|---|---|
| `Qwen2.5-1.5B-Instruct` | ~3 GB | Fast | Good | `Qwen/Qwen2.5-1.5B-Instruct` (**default**) |
| `TinyLlama-1.1B-Chat` | ~2 GB | Fastest | Fair | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `Qwen2.5-3B-Instruct` | ~6 GB | Medium | Better | `Qwen/Qwen2.5-3B-Instruct` |
| `Phi-3-mini-4k-instruct` | ~7 GB | Slow on CPU | Strong | `microsoft/Phi-3-mini-4k-instruct` |

Models are downloaded once to `~/.cache/huggingface/` and reused on subsequent runs. No API key required.

## Dataset: Arcturus Systems Inc.

A 180-person enterprise software company with **16 interconnected PDF documents** spanning:

| Department | Documents | Key People |
|---|---|---|
| Engineering | Team overview, Project Atlas spec, March outage report, Tech radar | Sarah Chen (VP), Marcus Rivera, Priya Sharma, Tom Nakamura, Lisa Park |
| Sales | Org structure, Project Beacon, Meridian Corp deal | Derek Hoffman (VP), Rachel Kim, Carlos Mendez |
| AI Research | Team overview, Project Cortex | Dr. Anika Patel |
| HR | Dept overview, Q2 hiring plan | Diana Reeves (VP) |
| Finance | Dept overview, FY2025 budget | Michael Torres (CFO) |
| Security | Team overview | Frank Morrison (CISO) |
| Executive | Board minutes, Strategic partnerships | James Whitfield (CEO), Olivia Foster (CTO) |

The documents are **intentionally interconnected** — people, projects, and decisions span multiple documents, creating a web of relationships that a knowledge graph can capture but flat chunk retrieval often misses.

## Data Engineering Pipeline

```
PDF files (data/pdfs/)
    │
    ▼
pdfplumber — extract raw text from each page
    │
    ▼
clean_text() — parse headers, extract title/department/doc_id, clean body
    │
    ▼
chunk_documents() — split into overlapping 300-word chunks
    │
    ▼
embed_texts() — sentence-transformers (all-MiniLM-L6-v2, local)
    │
    ▼
ChromaDB — in-memory vector store with cosine similarity
```

The ingestion pipeline ([src/ingest.py](src/ingest.py)) handles the full PDF-to-chunks transformation. The embedding module ([src/embeddings.py](src/embeddings.py)) runs entirely locally using HuggingFace models — no API calls needed.

## Side-by-Side Comparison: Where Graph RAG Wins (and Where It Doesn't)

### Category 1: Single-Document Lookups — TIE

These questions have answers contained in a single document. Both approaches handle them well.

| Question | Flat RAG | Graph RAG | Winner |
|---|---|---|---|
| "What programming languages does engineering use?" | **Python, Go, TypeScript** — pulled directly from eng-001 | **Python, Go, TypeScript** — same answer via entity lookup | **Tie** |
| "What caused the March 2025 outage?" | **Misconfigured rate limiter, skipped review** — found in eng-003 | **Same answer** — entity extraction captured the incident | **Tie** |
| "How much was the Series C?" | **$40M from Apex Ventures** — found in fin-001 | **Same** | **Tie** |

**Takeaway**: For simple factual lookups where the answer lives in one place, flat RAG is faster to index and equally accurate. Graph RAG adds overhead without benefit here.

---

### Category 2: Multi-Hop Reasoning — GRAPH RAG WINS

These questions require connecting information across 3-5 documents.

| Question | Flat RAG | Graph RAG | Winner |
|---|---|---|---|
| "What's the dependency chain: Atlas → Beacon → Meridian?" | Retrieves Atlas OR Beacon doc, **misses the full chain**. Often omits the SOC 2 dependency for Meridian. | Traverses `Atlas → feeds → Beacon → enables → Sales → Meridian` and pulls in `SOC 2 → blocks → Meridian`. **Full chain.** | **Graph RAG** |
| "Who does Dr. Anika Patel collaborate with?" | Finds 2-3 of her mentions (typically ai-001, ai-002). **Misses collaborations mentioned in other depts' docs** (sales-002, eng-004, sec-001). | Entity node "Dr. Anika Patel" has edges to every collaborator. **Finds all 5+ connections.** | **Graph RAG** |
| "If Atlas is delayed, what's affected downstream?" | Mentions Beacon dependency but **can't trace the full cascade** to Meridian and revenue impact. | Walks the dependency graph: `Atlas → Beacon → Sales targets → Meridian ($3.2M)`. Also connects to Cortex via shared data infrastructure. | **Graph RAG** |

**Takeaway**: When the answer requires traversing relationships that span multiple documents, graph RAG's entity-relationship structure provides a decisive advantage. Flat RAG retrieves the most similar chunks but can't "follow the thread."

---

### Category 3: Entity Aggregation — GRAPH RAG WINS

These questions ask about everything related to a specific entity that appears across many documents.

| Question | Flat RAG | Graph RAG | Winner |
|---|---|---|---|
| "What are CTO Olivia Foster's responsibilities?" | Finds 1-2 documents mentioning her. **Misses her role in partnerships, board presence, and oversight of security.** | Aggregates all edges connected to "Olivia Foster" node: reports (Patel, Morrison), board role, partnerships oversight, Atlas approval. **Comprehensive.** | **Graph RAG** |
| "List every project, owner, and status" | Retrieves 3-4 project-related chunks. **Misses at least one project** (typically Cortex or the DataMesh acquisition). | All entities typed as "Project" are indexed. **Returns all 5 projects** with owners and timelines. | **Graph RAG** |

**Takeaway**: Entities that are "distributed" across the corpus — mentioned in many documents but never fully described in one — are precisely where graph RAG excels. The knowledge graph acts as a unified entity profile.

---

### Category 4: Theme Summarization — GRAPH RAG WINS

These open-ended questions require synthesizing patterns across the entire corpus.

| Question | Flat RAG | Graph RAG | Winner |
|---|---|---|---|
| "What are the biggest risks facing Arcturus Systems?" | Identifies 2-3 risks from the most relevant chunks (typically the outage and burn rate). | Community detection groups related entities, surfacing **5-6 interconnected risks**: Atlas delays → Beacon → Meridian, hiring gaps, burn rate, SOC 2 blockers. | **Graph RAG** |
| "How does AI/ML flow through the organization?" | Finds the AI Research overview. **Misses downstream dependencies** in Sales, Security, and partnerships. | Traces the full AI influence graph: `Patel → Cortex → Beacon → Sales`, `Patel → Threat Detection → Security`, `Patel → Microsoft partnership`. | **Graph RAG** |

**Takeaway**: For questions about cross-cutting themes, graph RAG's community summaries provide pre-computed thematic views that flat retrieval can't match.

---

### Summary Scorecard

| Category | # Questions | Flat RAG | Graph RAG | Winner |
|---|---|---|---|---|
| Single-Doc Lookup | 3 | ~4.5/5 | ~4.5/5 | Tie |
| Multi-Hop Reasoning | 3 | ~2.5/5 | ~4.5/5 | **Graph RAG (+2.0)** |
| Entity Aggregation | 2 | ~3.0/5 | ~4.5/5 | **Graph RAG (+1.5)** |
| Theme Summarization | 2 | ~2.5/5 | ~4.0/5 | **Graph RAG (+1.5)** |
| **Overall** | **10** | **~3.2/5** | **~4.4/5** | **Graph RAG** |

> **Scores above are representative estimates.** Run `python run_comparison.py` to get actual LLM-evaluated scores on your own run. Results vary by model — larger models produce better scores.

## When to Use Which

| Use Flat RAG when... | Use Graph RAG when... |
|---|---|
| Questions target a single document/topic | Questions require connecting multiple documents |
| Your corpus has minimal cross-references | Entities and relationships span many documents |
| Low latency is critical | Answer quality on complex queries matters most |
| Compute budget is tight | You can afford a one-time indexing cost |
| Documents are self-contained (e.g., FAQs) | Documents describe interconnected systems (e.g., org docs, incident reports, project specs) |

## Cost & Performance

Everything runs locally — zero API costs, works offline after first model download.

| Metric | Flat RAG | Graph RAG |
|---|---|---|
| Indexing time | ~5s (local embeddings only) | ~5-15min (local LLM extraction + embed + summaries) |
| Query latency | ~5s (1 embed + 1 local LLM call) | ~15s (entity ID + embed + local LLM call) |
| API cost | $0 — fully local | $0 — fully local |
| GPU required? | No (CPU fine) | No (CPU fine, but GPU 10x faster) |
| Disk (first run) | ~50MB (embedding model) | ~50MB + ~3GB (LLM model download) |

## Project Structure

```
graphrag-enterprise-starter/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── run_comparison.py            # Full 10-question evaluation
├── demo_single_question.py      # Quick single-question demo
├── data/
│   ├── knowledge_base.py        # Document content + 10 evaluation questions
│   ├── generate_pdfs.py         # Script to create PDF files
│   └── pdfs/                    # Generated PDF documents (gitignored)
├── src/
│   ├── ingest.py                # PDF ingestion pipeline (read, clean, chunk)
│   ├── embeddings.py            # sentence-transformers embeddings (local)
│   ├── llm.py                   # Local HuggingFace transformers LLM interface
│   ├── flat_rag.py              # ChromaDB-based flat vector RAG
│   ├── graph_rag.py             # NetworkX + entity extraction graph RAG
│   └── compare.py               # Evaluation engine with LLM-as-judge scoring
└── output/
    └── comparison_results.json  # Generated after running comparison
```

## How It Works

### Data Pipeline
1. **Generate** PDF documents from the knowledge base (`python data/generate_pdfs.py`)
2. **Ingest** PDFs using pdfplumber — extract text, parse headers, clean content
3. **Chunk** into overlapping 300-word segments with metadata tracking

### Flat Vector RAG
1. **Embed** chunks using `all-MiniLM-L6-v2` (sentence-transformers, runs locally)
2. **Store** in ChromaDB with cosine similarity
3. **Retrieve** top-k chunks by query similarity
4. **Generate** answer using local HuggingFace model via `transformers`

### Graph-Enhanced RAG
1. **Extract** entities (people, projects, technologies) and relationships from each document using local HuggingFace model
2. **Build** a directed knowledge graph in NetworkX
3. **Detect** communities using greedy modularity optimization
4. **Summarize** each community into a thematic description
5. **At query time**: identify relevant entities → traverse graph (2-hop BFS) → retrieve community summaries → augment with vector search → generate answer

### Evaluation
- Each answer is scored by the LLM judge on **completeness** (1-5) and **accuracy** (1-5)
- Scores are compared against expected answers from the evaluation set
- Results are aggregated by question category

## Tech Stack

All open-source:

| Component | Tool | Role |
|---|---|---|
| LLM | **HuggingFace transformers** (`Qwen2.5-1.5B-Instruct` default, local) | Generation, entity extraction, evaluation |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Vector embeddings, runs locally on CPU |
| Vector Store | **ChromaDB** | In-memory similarity search |
| Knowledge Graph | **NetworkX** | Graph storage and traversal |
| PDF Processing | **pdfplumber** | Text extraction from PDFs |
| PDF Generation | **fpdf2** | Create dataset PDFs |

No OpenAI, no Neo4j, no external databases, no API keys. Runs 100% offline after first model download.

## License

MIT
