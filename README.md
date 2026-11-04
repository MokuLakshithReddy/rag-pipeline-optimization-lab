# RAG Pipeline Optimization Lab

A production-oriented Retrieval-Augmented Generation (RAG) engineering, benchmarking, and optimization platform developed for the Kaggle Competition: **Gen AI Project 4 – Building & Optimizing RAG Pipelines**.

- **Official Competition Link**: [https://www.kaggle.com/competitions/gen-ai-project-4-building-optimizing-rag-pipelines](https://www.kaggle.com/competitions/gen-ai-project-4-building-optimizing-rag-pipelines)
- **Competition Audit & Specification**: [docs/COMPETITION_SPEC.md](docs/COMPETITION_SPEC.md)

---

## 1. Problem Statement & Motivation

Basic "PDF chatbot" tutorials typically perform naive fixed-window character splitting, embed chunks through a single dense vector model, and fetch nearest neighbors using plain vector similarity. In real-world enterprise environments and rigorous competition settings, this naive approach suffers from severe pathologies:
1. **Semantic Boundary Mutilation**: Slicing text mid-sentence or mid-proposition destroys the syntactic coherence of key factual statements.
2. **Dense vs. Keyword Mismatch**: Dense embeddings frequently fail on exact part numbers, technical acronyms, and specialized nomenclature that lexical search easily captures. Conversely, keyword search (BM25) fails when user queries are paraphrased.
3. **Score Distribution Incompatibility**: Naively summing raw cosine similarities and BM25 scores leads to severe score calibration artifacts.
4. **The "Lost in the Middle" Effect**: LLM attention degrades for relevant passages placed in the middle of large context windows.
5. **Indirect Prompt Injection**: Malicious or adversarial documents ingested into the index can hijack generation prompts.

This platform implements an end-to-end, reproducible RAG engineering suite with multi-format parsing, multi-strategy chunking, hybrid Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, Lost-in-the-Middle context ordering, and strict evidence-grounded generation with abstention guards.

---

## 2. Key Architecture & Features

```
[ Documents (PDF, MD, TXT, JSON, CSV) ]
                  |
         (Safe Parsing & Hash Dedup)
                  v
[ Hierarchical Recursive Chunking (512 / 64) ]
                  |
     +------------+------------+
     |                         |
     v                         v
[ Dense Index ]          [ Sparse Index ]
SentenceTransformers        Okapi BM25
 (all-MiniLM-L6-v2)     (Exact Keyword Match)
     |                         |
     +------------+------------+
                  |
                  v
[ Hybrid Reciprocal Rank Fusion (RRF k=60) ]
                  |
                  v
[ Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2) ]
                  |
                  v
[ Context Optimizer (Lost-in-the-Middle + Deduplication) ]
                  |
                  v
[ Evidence-Grounded Generator & Abstention Guard ]
```

- **Universal Document Loader**: Safely parses `.pdf` (preserving page numbers), `.txt`, `.md`, `.json`, and `.csv`.
- **Three Chunking Strategies**: Fixed-size chunking, Recursive character chunking, and Semantic shift chunking.
- **Dual Representation Indexing**: SentenceTransformers dense vector search (`all-MiniLM-L6-v2`) with L2 normalization + Okapi BM25 sparse index.
- **Hybrid Fusion (RRF)**: Reciprocal Rank Fusion ($k=60$) dynamically merges dense and sparse rankings without arbitrary score scaling.
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top candidate chunks with full query-chunk cross-attention.
- **Context Optimizer**: Formats source citations, removes redundant chunks, and places top evidence at the beginning and end of context.
- **Security by Design**: Path traversal protection, 25MB file upload ceiling, content type whitelisting, prompt injection containment, and safe error logging.

---

## 3. Benchmark Results

Controlled single-variable experiments were conducted against a verified 30-document multi-domain benchmark corpus with 20 challenging ground-truth queries (conceptual paraphrases, exact keywords, multi-hop, distractors):

| Experiment ID | Retrieval Strategy | Chunking Strategy | Reranker | Hit@5 | MRR | NDCG@5 | Latency (Mean) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense-Fixed256` | Dense | Fixed (256/32) | False | **1.000** | **0.919** | **0.938** | 30.2ms |
| `EXP-02-Chunking-Recursive256` | Dense | Recursive (256/32) | False | **0.944** | **0.907** | **0.917** | 0.4ms |
| `EXP-03-Chunking-Recursive512` | Dense | Recursive (512/64) | False | **1.000** | **0.926** | **0.944** | 0.5ms |
| `EXP-04-Sparse-BM25` | Sparse | Recursive (512/64) | False | **1.000** | **0.956** | **0.966** | 0.8ms |
| `EXP-05-Hybrid-RRF` | Hybrid | Recursive (512/64) | False | **1.000** | **1.000** | **1.000** | 1.2ms |
| `EXP-06-Hybrid-Reranked` | Hybrid | Recursive (512/64) | Neural Cross-Encoder | **1.000** | **0.972** | **0.979** | 236.9ms |

*Detailed analysis is documented in [reports/retrieval_analysis.md](reports/retrieval_analysis.md) and [reports/experiment_summary.md](reports/experiment_summary.md).*

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.10)
- `uv` (recommended) or standard `pip`

### Step 1: Clone and Set Up Virtual Environment
```bash
# Create virtual environment
uv venv .venv --python 3.12

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install project and dependencies
uv pip install -e .
```

### Step 2: Configure Environment
```bash
cp .env.example .env
```

---

## 5. Running the Application

### Option A: Interactive Streamlit UI
```bash
uv run streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Features:
- Upload and inspect custom documents.
- Load the standardized benchmark corpus with 1 click.
- Switch between Hybrid, Dense, and Sparse retrieval in real time.
- Toggle Cross-Encoder reranking and query enhancement.
- View grounded answers, source citations, latency breakdowns, and retrieved chunks.

### Option B: FastAPI Backend Server
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Option C: Docker Compose
```bash
docker-compose up --build
```

---

## 6. Running Tests & Benchmarks

Execute the complete 46-test Pytest test suite covering unit, integration, mathematical metric bounds, and security checks:
```bash
uv run pytest tests -v
```

Execute the controlled experiment benchmark suite:
```bash
uv run python src/experiments/runner.py
```

---

## 7. Kaggle Competition Solution

The dedicated competition submission notebook is located at:
- [`notebooks/competition_solution.ipynb`](notebooks/competition_solution.ipynb)

It is completely self-contained and handles:
- Running offline inside Kaggle GPU/CPU kernels without external API calls.
- Adapting to both Q&A submission schemas (`id,prediction`) and ranked retrieval schemas (`query_id,retrieved_doc_ids`).
- Enforcing row-count integrity and null-value validation.

---

## 8. Repository Structure

```
rag-pipeline-optimization-lab/
├── api/                       # FastAPI REST API & Pipeline State
├── app/                       # Streamlit Interactive Demonstration UI
├── src/
│   ├── ingestion/             # Universal loader, models, and validation
│   ├── chunking/              # Fixed, Recursive, and Semantic chunkers
│   ├── embeddings/            # SentenceTransformers and Fast embedders
│   ├── retrieval/             # Dense, Sparse BM25, and Hybrid RRF retrievers
│   ├── reranking/             # Cross-Encoder reranker and fallbacks
│   ├── generation/            # Evidence-grounded generation and abstention
│   ├── evaluation/            # Precision, Recall, MRR, NDCG, and Faithfulness
│   └── experiments/           # Controlled benchmark runner and tracker
├── notebooks/
│   ├── competition_solution.ipynb
│   └── engineering_benchmarks.ipynb
├── tests/
│   ├── unit/                  # Ingestion, chunking, retrieval, generation tests
│   ├── integration/           # Pipeline and API integration tests
│   └── security/              # Path traversal, upload validation, injection tests
├── data/
│   └── benchmarks/            # Standardized benchmark corpus and queries
├── reports/                   # Generated benchmark CSV and analysis reports
├── configs/                   # Pipeline YAML configuration
├── docs/                      # Architectural, evaluation, security, and Kaggle specs
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 9. License

This repository is governed by the project's [Restricted Proprietary License](LICENSE).

