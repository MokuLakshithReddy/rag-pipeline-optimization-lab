# System Architecture & Technical Design

## 1. High-Level Architectural Topology

The RAG Pipeline Optimization Lab is structured around modular, decoupled components conforming to modern high-performance retrieval architectures:

```
+-----------------------------------------------------------------------------------+
|                                 Client Interfaces                                 |
|         Streamlit Dashboard (Port 8501)   |   FastAPI REST API (Port 8000)        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                 API & State Layer                                 |
|    Security Middleware (CORS, Request ID, Nosniff, Rate Limits, Timing Headers)   |
|    Pydantic Schema Validation & Global PipelineState                              |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             Ingestion & Validation                                |
|    Multi-Format Document Parsing (PDF, Markdown, TXT, JSON, CSV)                  |
|    Path Traversal Guard, Size Caps (25MB), SHA-256 Deduplication, Encoding Detect|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                Chunking Engine                                    |
|    Fixed-Size Chunker | Recursive Boundary Chunker | Semantic Shift Chunker       |
|    Metadata Preservation (Doc ID, Page #, Start/End Char Spans)                   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             Dual Representation Index                             |
|    Dense Vectors: SentenceTransformers (all-MiniLM-L6-v2) + L2 Cosine Norm        |
|    Sparse Inverted Index: Okapi BM25 with Token Normalization                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                               Retrieval & Ranking                                 |
|    Hybrid Reciprocal Rank Fusion (RRF k=60) & Weighted Fusion                     |
|    Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) with Deterministic Fallback    |
|    Context Optimizer (Lost-in-the-Middle Reordering, Dedup, Token Budget Guard)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        Generation & Evidence Grounding                            |
|    Extractive / LLM Grounded Answer Synthesis                                     |
|    Source Citation Tagging [Source X: filename, Page Y]                           |
|    Prompt Injection Defense Boundaries (<untrusted_context>) & Abstention Guard   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Ingestion & Parsing (`src/ingestion/`)
- **UniversalDocumentLoader**: Supports `.pdf` (via `pypdf` with page tracking), `.txt`, `.md`, `.json`, and `.csv`.
- **Validation**: Enforces 25MB limits, extension whitelisting, cryptographic deduplication using SHA-256, and sanitizes filenames against directory traversal (`../`).

### 2.2 Chunking Strategies (`src/chunking/`)
- **FixedSizeChunker**: Window slicing with configurable step size.
- **RecursiveCharacterChunker**: Splits on hierarchical separators (`["\n\n", "\n", ". ", " ", ""]`) to maintain complete thoughts and semantic propositions.
- **SemanticChunker**: Computes semantic shifts between adjacent sentences to split at natural thematic transitions.

### 2.3 Retrieval Multi-Index (`src/retrieval/`)
- **DenseRetriever**: Encodes queries and chunks into unit-length vectors. Computes cosine similarity via dot product.
- **SparseRetriever**: Tokenizes text into lowercase lexical tokens, evaluates BM25 scores with term frequency and document frequency saturation.
- **HybridRetriever**: Blends rank signals via Reciprocal Rank Fusion:
  $$ \text{Score}_{\text{RRF}}(d) = \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{k + \text{rank}_{\text{sparse}}(d)} $$
  eliminating scoring distribution mismatch between unbounded BM25 scores and cosine metrics.

### 2.4 Reranking & Context Assembly (`src/reranking/`, `src/retrieval/context_optimizer.py`)
- **CrossEncoderReranker**: Evaluates full cross-attention over query-document token pairs.
- **Lost-in-the-Middle Reordering**: Alternates highest scoring passages between the start and end of prompt context.
- **Token Budget Manager**: Truncates context safely within maximum character/token budget.

### 2.5 Generation & Prompt Safety (`src/generation/`)
- **LLMGenerator**: Supports deterministic extractive citation synthesis and Google Gemini API integration.
- **Abstention Protocol**: When similarity or term overlap fails minimum threshold, abstains with standardized message.
- **Prompt Framing**: Context is wrapped in explicit `<untrusted_context>` blocks to neutralize prompt injection instructions.

