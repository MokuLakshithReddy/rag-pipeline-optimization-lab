# RAG Pipeline Optimization Lab

A production-grade Retrieval-Augmented Generation (RAG) optimization, benchmarking, and multi-tenant deployment platform developed for the Kaggle Competition: **Gen AI Project 4 – Building & Optimizing RAG Pipelines**.

- **Official Competition Link**: [https://www.kaggle.com/competitions/gen-ai-project-4-building-optimizing-rag-pipelines](https://www.kaggle.com/competitions/gen-ai-project-4-building-optimizing-rag-pipelines)
- **Competition Audit & Specification**: [docs/COMPETITION_SPEC.md](docs/COMPETITION_SPEC.md)

---

## 1. Problem Statement & Motivation

Standard "PDF chatbot" tutorials typically perform naive fixed-window character splitting, embed chunks through a single dense vector model, and fetch nearest neighbors using plain vector similarity. In real-world enterprise environments and competition evaluations, this naive approach suffers from fundamental engineering bottlenecks:
1. **Semantic Boundary Mutilation**: Slicing text mid-sentence or mid-proposition destroys the syntactic coherence of key factual statements.
2. **Dense vs. Keyword Mismatch**: Dense embeddings frequently fail on exact part numbers, technical acronyms, and specialized nomenclature that lexical search easily captures. Conversely, keyword search (BM25) fails when user queries are paraphrased.
3. **Score Incompatibility**: Naively summing raw cosine similarities and BM25 scores leads to severe calibration artifacts.
4. **The "Lost in the Middle" Effect**: LLM attention degrades for relevant passages placed in the middle of large context windows.
5. **Indirect Prompt Injection & Multi-Tenant Bleed**: Shared global state leads to cross-tenant data leakage, while adversarial documents can hijack LLM generation prompts.
6. **Conflating Relevance with Factual Entailment**: Using relevance rerankers (e.g. MS-MARCO) to score faithfulness yields misleading results because relevance does not verify premise-to-hypothesis entailment.

This platform implements an end-to-end, reproducible RAG engineering suite with multi-format parsing, multi-strategy chunking, hybrid Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, Lost-in-the-Middle context ordering, dedicated proposition-level NLI entailment verification, authenticated multi-tenant state isolation, and strict evidence-grounded generation with abstention guards.

---

## 2. Key Architecture & Features

```
[ Documents (PDF, MD, TXT, JSON, CSV) ]
                  |
     (Safe Parsing & Tenant Hash Dedup)
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
                  |
                  v
[ Dedicated NLI Factual Entailment Evaluator ]
```

- **Universal Document Loader**: Safely parses `.pdf` (preserving page numbers), `.txt`, `.md`, `.json`, and `.csv`.
- **Three Chunking Strategies**: Fixed-size chunking, Recursive character chunking, and Semantic shift chunking.
- **Dual Representation Indexing**: SentenceTransformers dense vector search (`all-MiniLM-L6-v2`) with L2 normalization + Okapi BM25 sparse index.
- **Hybrid Fusion (RRF)**: Reciprocal Rank Fusion ($k=60$) dynamically merges dense and sparse rankings without arbitrary score scaling.
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top candidate chunks with full query-chunk cross-attention (with enforced strict mode to prevent silent lexical fallback).
- **Context Optimizer**: Formats source citations, removes redundant chunks, and places top evidence at the beginning and end of context.
- **Dedicated NLI Factual Entailment**: `FactualEntailmentEvaluator` decomposes answers into proposition claims and verifies 3-class entailment (`entailed`, `contradicted`, `neutral`) with transparent reporting of evaluation mode (`heuristic_claim_verifier`, `transformer_nli`, or `llm_judge`).
- **Multi-Tenant State Isolation**: `MultiTenantStateManager` isolates uploads, vector stores, and BM25 indices per tenant while reusing singleton neural weights across requests.
- **Fail-Closed Security**: Rejects all unconfigured requests with HTTP 401; maps tenant credentials strictly via `API_KEYS`; blocks cross-tenant header escalation with HTTP 403; enforces 25MB upload ceilings, path traversal sanitization, and prompt injection containment.

---

## 3. Controlled Benchmark Evidence & Scalability Analysis

> [!NOTE]
> **Controlled Laboratory Benchmark Evidence**: All metrics and latency numbers documented below represent controlled laboratory benchmark evidence executed on standardized hardware under controlled test conditions, rather than unsubstantiated claims of unconstrained enterprise production scale.

### A. Controlled Retrieval Ablation Suite
Controlled single-variable experiments were conducted against a verified 30-document multi-domain benchmark corpus with 20 challenging ground-truth queries:

| Experiment ID | Retrieval Strategy | Chunking Strategy | Reranker (Status) | Generator | Hit@5 | MRR | NDCG@5 | Latency (p95) | Delta RSS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense-Fixed256` | Dense | Fixed (256/32) | False (none) | mock | **1.000** | **0.919** | **0.938** | 39.2ms | +12.3MB |
| `EXP-02-Chunking-Recursive256` | Dense | Recursive (256/32) | False (none) | mock | **0.944** | **0.907** | **0.917** | 1.4ms | +25.1MB |
| `EXP-03-Chunking-Recursive512` | Dense | Recursive (512/64) | False (none) | mock | **1.000** | **0.926** | **0.944** | 1.2ms | +1.5MB |
| `EXP-04-Sparse-BM25` | Sparse | Recursive (512/64) | False (none) | mock | **1.000** | **0.956** | **0.966** | 1.6ms | +0.1MB |
| `EXP-05-Hybrid-RRF` | Hybrid | Recursive (512/64) | False (none) | mock | **1.000** | **1.000** | **1.000** | 4.0ms | +0.0MB |
| `EXP-06-Hybrid-Reranked` | Hybrid | Recursive (512/64) | Neural Cross-Encoder | mock | **1.000** | **0.972** | **0.979** | 307.1ms | +101.5MB |

### B. End-to-End Real Neural LLM Generation & Factual Verification
In addition to mock extractive heuristics used for retrieval ablations, an end-to-end benchmark was executed using an active local neural generative model (`google/flan-t5-small`) evaluated with transparently reported proposition-level NLI:

| Experiment ID | Retrieval Pipeline | Generator Model | Mode | MRR | NDCG@5 | Answer F1 | Faithfulness | Generation Latency | Total Latency (p95) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `NEURAL-01-Dense` | Dense (all-MiniLM-L6-v2) | `google/flan-t5-small` | `enhanced` | **0.926** | 0.944 | **0.562** | **0.938** | 3102.8ms | 4575.0ms |
| `NEURAL-02-Sparse` | Sparse (BM25) | `google/flan-t5-small` | `enhanced` | **0.956** | 0.966 | **0.543** | **0.912** | 4120.4ms | 8368.0ms |
| `NEURAL-03-Hybrid-RRF` | Hybrid RRF | `google/flan-t5-small` | `enhanced` | **1.000** | 1.000 | **0.446** | **0.950** | 4145.1ms | 11671.0ms |
| `NEURAL-04-Hybrid-Reranked` | Hybrid + Cross-Encoder | `google/flan-t5-small` | `enhanced` | **0.972** | 0.980 | **0.449** | **0.938** | 2829.5ms | 4648.8ms |

#### Apples-to-Apples Generation Ablation (Same 20 Queries, Same Citation-Stripped Metric)
To scientifically isolate how much of the F1 improvement originates from prompt design & grounded synthesis versus the metric change (stripping citation tags), both generation engines were evaluated on the **exact same 20 queries using the identical citation-stripped F1 metric**:

| Generation Engine | Prompt Strategy | Fallback Synthesis | Answer Token F1 | Factual Faithfulness | Generation Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Legacy Generator** | Terse (`Context: ...\nQuestion: ...\nAnswer:`) | None (Raw fragments) | **0.191** | 0.950 | 5251.9ms |
| **Enhanced Generator** | Guided Sentence Instruction | Grounded Sentence Synthesis | **0.446** | 0.950 | 4145.1ms |

- **Isolated Prompt & Synthesis Delta**: **+0.255 F1 gain (+133.8%)** purely attributable to guided sentence instructions and grounded synthesis, holding the evaluation metric strictly identical.

### C. Multi-Scale Corpus, Concurrency & Throughput Stress Testing
To stress-test performance beyond the baseline 30-document development set, the pipeline was benchmarked across multi-scale corpus tiers (up to 500 documents / **1,503 chunks**) and simulated concurrent client workloads (1 to 20 concurrent threads):

#### Corpus Scaling & Isolated Memory Footprint
*Note: Embedding model weights are pre-warmed prior to tier profiling to cleanly isolate index overhead from model initialization.*

| Corpus Tier | Chunks Indexed | Indexing Time | Baseline RSS | Post-Index RSS | Index Memory Delta | Query Latency (Mean) | Latency (p95) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small (30 docs)** | **93 chunks** | 2.98s | 548.5 MB | 625.5 MB | **+77.0 MB** | 31.49 ms | 45.0 ms |
| **Medium (100 docs)** | **303 chunks** | 5.61s | 604.1 MB | 584.6 MB | **+0.0 MB** | 1.66 ms | 2.05 ms |
| **Large (500 docs)** | **1,503 chunks** | 12.25s | 584.6 MB | 598.4 MB | **+13.7 MB** | **7.82 ms** | **10.73 ms** |

#### In-Process Concurrency & Sustained Throughput (Evaluated on 500-Doc / 1,503-Chunk Tier)
| Concurrency Level | Warm-Cache QPS | Cold-Cache QPS | Warm Latency (Mean) | Warm Latency (p95) | Cold Latency (p95) | Peak RSS | Memory Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 worker** | **117.5 req/s** | **19.2 req/s** | 8.42 ms | **11.23 ms** | 71.56 ms | 601.3 MB | +2.8 MB |
| **5 workers** | **97.9 req/s** | **81.5 req/s** | 50.21 ms | **67.86 ms** | 76.94 ms | 601.4 MB | +0.1 MB |
| **10 workers** | **96.9 req/s** | **80.7 req/s** | 98.51 ms | **134.30 ms** | 154.20 ms | 601.6 MB | +0.2 MB |
| **20 workers** | **92.2 req/s** | **80.0 req/s** | 197.87 ms | **302.44 ms** | 310.86 ms | 602.8 MB | +1.2 MB |

- **Throughput Qualification**: The reported QPS measures **in-process retriever throughput** (warm-cache retrieval vs. cold-cache neural query embedding on CPU); full HTTP requests through the FastAPI REST gateway include ASGI framework parsing and JSON serialization.

### D. The Latency-Accuracy Pareto Frontier & Production Scope
- **Low-Latency Production (<20ms SLA)**: Pure **Hybrid RRF** (`EXP-05`) achieves optimal MRR (1.000) and NDCG (1.000) on the development benchmark with a 4.0ms p95 latency and negligible incremental memory footprint.
- **Deep-Reasoning Production (>500ms SLA)**: Cross-Encoder neural reranking (`EXP-06`) adds full transformer cross-attention, resolving subtle candidate ambiguities at the cost of ~300ms CPU latency and +101.5MB Delta RSS.
- **Controlled Laboratory Scope**: These results establish rigorous empirical bounds for medium-scale retrieval. Enterprise deployments scaling to millions of documents require distributed sharding (e.g. Qdrant/Milvus cluster) and GPU-accelerated batch reranking.

*Detailed analysis is documented in [reports/retrieval_analysis.md](reports/retrieval_analysis.md), [reports/experiment_summary.md](reports/experiment_summary.md), [reports/neural_generation_summary.md](reports/neural_generation_summary.md), and [reports/scalability_benchmark_report.md](reports/scalability_benchmark_report.md).*

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

### Option B: Authenticated FastAPI Backend Server
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).
Authenticate using API Key headers:
```bash
# Querying via tenant-authenticated endpoint
curl -X POST "http://localhost:8000/v1/query" \
     -H "X-API-Key: test-api-key-default" \
     -H "Content-Type: application/json" \
     -d '{"query": "How does Raft handle leader election?"}'
```

### Option C: Docker Compose
```bash
docker-compose up --build
```

---

## 6. Running Tests & Benchmarks

Execute the complete test suite covering unit, integration, multi-tenancy, authentication, mathematical metric bounds, cross-encoder strict mode, Kaggle submission validation, and security checks:
```bash
uv run pytest tests -v
```

Execute the controlled retrieval ablation suite:
```bash
uv run python src/experiments/runner.py
```

Execute the live neural LLM generation benchmark:
```bash
uv run python src/experiments/run_neural_benchmark.py --model-name google/flan-t5-small
```

Execute the multi-scale corpus, concurrency, and throughput stress benchmark:
```bash
uv run python src/experiments/benchmark_scalability.py --corpus-tiers 30 100 500 --concurrency-levels 1 5 10 20
```

Validate a Kaggle competition submission file against the test set schema:
```bash
uv run python src/evaluation/kaggle_validator.py --submission reports/sample_submission.csv --test-set data/benchmarks/rag_benchmark.json
```

---

## 7. Kaggle Competition Solution & Submission Verification

The dedicated competition submission notebook and offline validation harness are located at:
- Notebook: [`notebooks/competition_solution.ipynb`](notebooks/competition_solution.ipynb)
- Submission Schema Validator: [`src/evaluation/kaggle_validator.py`](src/evaluation/kaggle_validator.py)
- Integration Verification Suite: [`tests/integration/test_kaggle_validation.py`](tests/integration/test_kaggle_validation.py)

Key Verification Guarantees:
- **Autonomous Schema Conformance**: Dynamically inspects and adapts to arbitrary `sample_submission.csv` schemas (whether Q&A text columns, ranked document IDs, or multi-target formats) without hardcoded assumptions.
- **Strict Verification Assertions**: Verifies row counts, column names, non-null constraints, and UTF-8 encoding before outputting `/kaggle/working/submission.csv`.
- **Pre-Submission Validator**: The standalone `KaggleSubmissionValidator` guarantees zero NaN/null values, exact query ID alignment with the test set, and absence of duplicate predictions prior to submitting to Kaggle.
- **Controlled Benchmark Standby**: The pipeline operates in controlled standby mode; when the official competition dataset files (`train.csv`, `test.csv`, `documents.parquet`) are downloaded, the runner seamlessly transitions from the synthetic development set to official evaluation without pipeline redesign.

---

## 8. Repository Structure

```
rag-pipeline-optimization-lab/
├── api/                       # Authenticated FastAPI REST API & Multi-Tenant State
├── app/                       # Streamlit Interactive Demonstration UI
├── src/
│   ├── ingestion/             # Universal loader, models, and validation
│   ├── chunking/              # Fixed, Recursive, and Semantic chunkers
│   ├── embeddings/            # SentenceTransformers and Fast embedders
│   ├── retrieval/             # Dense, Sparse BM25, and Hybrid RRF retrievers
│   ├── reranking/             # Cross-Encoder reranker and fallbacks
│   ├── generation/            # Evidence-grounded generation and abstention
│   ├── evaluation/            # Metrics, Evaluator, and FactualEntailmentEvaluator
│   └── experiments/           # Controlled benchmark runner and neural benchmark
├── notebooks/
│   ├── competition_solution.ipynb
│   └── engineering_benchmarks.ipynb
├── tests/
│   ├── unit/                  # Ingestion, chunking, retrieval, metric bounds tests
│   ├── integration/           # Pipeline, API auth, and multi-tenant isolation tests
│   └── security/              # Path traversal, upload validation, injection tests
├── data/
│   └── benchmarks/            # Standardized benchmark corpus and queries
├── reports/                   # Generated benchmark CSVs and analysis reports
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
