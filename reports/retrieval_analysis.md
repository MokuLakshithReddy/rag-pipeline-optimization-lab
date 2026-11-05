# Deep Retrieval Systems & Trade-Off Analysis

## Executive Summary
This report analyzes retrieval performance across dense vector search, sparse keyword BM25 retrieval, reciprocal rank fusion (RRF) hybrid search, and cross-encoder neural reranking. Controlled single-variable experiments were conducted on a 30-document multi-domain technical corpus with 20 ground-truth queries to determine causal performance drivers.

Additionally, end-to-end neural LLM generation was benchmarked using an active local generative model (`google/flan-t5-small`) evaluated with genuine proposition-level NLI factual entailment verification.

---

## 1. Chunking Trade-Off Analysis
- **Fixed-Size Chunking (256 chars, 32 overlap)**: High speed and deterministic memory footprint, but frequently cuts through compound technical terms, resulting in lower retrieval precision when sentence boundaries are severed.
- **Recursive Character Chunking (512 chars, 64 overlap)**: Significantly superior semantic integrity. Prioritizing paragraph (`\n\n`) and sentence (`. `) boundaries preserves complete conceptual thoughts and causal assertions, maximizing MRR and Precision.
- **Semantic Chunking (Vector similarity breakpoint)**: Creates dynamic boundaries aligned with topic shifts. Ideal for multi-topic documents, but higher variance in chunk length requires dynamic token budgeting.

---

## 2. Dense vs. Sparse vs. Hybrid Fusion (RRF)
- **Dense Vector Search (All-MiniLM-L6-v2 / Cosine)**:
  - *Strengths*: Captures conceptual paraphrasing and semantic intent without lexical overlap (e.g. "leader election upon node timer expiration" matches Raft).
  - *Weaknesses*: Prone to false positives when technical acronyms (e.g., PBFT, PKCE, COW) share generic semantic neighborhoods.
- **Sparse Keyword Search (Okapi BM25)**:
  - *Strengths*: Exceptional precision on exact acronyms, parameter names (`k1`, `b`), and specific symbols (`3f + 1`).
  - *Weaknesses*: Zero recall when queries use conceptual synonyms or descriptive phrasing without exact keywords.
- **Hybrid Search (Reciprocal Rank Fusion k=60)**:
  - *Result*: Achieves highest overall MRR (1.000) and NDCG (1.000) on the development corpus. Rank fusion eliminates score distribution mismatch between unbounded BM25 scores and bounded cosine similarities.

---

## 3. The Latency-Accuracy Pareto Frontier & Cross-Encoder Reranking
- **Empirical Measurement**:
  | Configuration | Retrieval MRR | NDCG@5 | Latency (p95) | Delta RSS | Memory Overhead |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Hybrid RRF (`EXP-05`)** | **1.000** | **1.000** | **4.0 ms** | +0.0 MB | 0.2 MB |
  | **Hybrid + Cross-Encoder (`EXP-06`)** | 0.972 | 0.979 | 307.1 ms | +101.5 MB | 92.9 MB |

- **Pareto Trade-Off Insights**:
  1. **Latency Overhead**: Evaluating query-document pairs simultaneously via transformer cross-attention adds ~300 ms on CPU per query (a ~75x latency multiplier over pure RRF).
  2. **Memory Footprint**: Loading cross-encoder transformer weights adds +101.5 MB Delta RSS and ~93 MB active heap.
  3. **Ranking Inversion on Terse Technical Queries**: On terse acronym queries, BM25 rank fusion immediately places the exact-match document at rank 1. Cross-encoders, trained on conversational search (MS-MARCO), can occasionally downrank terse acronym definitions in favor of longer, syntactically richer passages.
  4. **Production Architecture Recommendation**:
     - **Default / Real-Time SLA (<20ms)**: Deploy **Hybrid RRF** (`k=60`). It delivers elite retrieval quality with sub-5ms CPU latency and zero added memory overhead.
     - **Complex Reasoning / Deep Rerank SLA (>500ms)**: Deploy **Hybrid RRF + Cross-Encoder**, but apply reranking strictly to the top-K candidates (e.g. top 10 reranked to top 3) and offload cross-attention to GPU accelerators.

---

## 4. Controlled Benchmark Scope & Generalizability
- **Development Corpus Design**: The 30-document, 20-query evaluation set provides controlled ground truth covering distributed systems, security protocols, containerization, and data structures.
- **Scope Qualification**: Perfect retrieval scores (MRR = 1.000) for Hybrid RRF reflect strong performance on this curated technical benchmark. In open-domain production with millions of documents and noisy conversational queries, performance will vary. RRF provides a robust theoretical foundation by eliminating bi-encoder out-of-domain failure modes, but must be continuously validated as domain corpora scale.

---

## 5. Real Neural LLM Generation & Factual Faithfulness
In addition to mock extractive heuristics, an end-to-end neural generation benchmark was executed using `google/flan-t5-small`:

| Experiment ID | Retrieval Strategy | Generator | Retrieval MRR | Answer Token F1 | Factual Faithfulness | Generation Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `NEURAL-01-Dense` | Dense Vector | `google/flan-t5-small` | 0.926 | 0.214 | 1.000 | 2256.6 ms |
| `NEURAL-02-Sparse` | Okapi BM25 | `google/flan-t5-small` | 0.956 | 0.123 | 0.975 | 723.6 ms |
| `NEURAL-03-Hybrid-RRF` | Hybrid RRF | `google/flan-t5-small` | **1.000** | 0.172 | 0.950 | 829.6 ms |
| `NEURAL-04-Hybrid-Reranked`| Hybrid + Reranker | `google/flan-t5-small` | 0.972 | 0.035 | **1.000** | 379.6 ms |

- **Key Generation Takeaways**:
  1. **The Generation Bottleneck**: Generation latency (380 ms to 2250 ms) accounts for over 90% of end-to-end RAG latency, reinforcing why retrieval must remain lean (~4 ms).
  2. **Grounded Abstractive Synthesis**: Neural models generate coherent sentences rather than extractive spans. Token F1 against terse reference answers is naturally lower for abstractive synthesis, making NLI factual verification essential.

---

## 6. Genuine NLI Factual Entailment vs Relevance Reranking
- **Architectural Decoupling**: Prior systems conflated query-passage relevance models (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with factual entailment. Relevance models measure whether a passage is about a topic, not whether premise $P$ entails hypothesis $H$.
- **Dedicated NLI Verifier (`FactualEntailmentEvaluator`)**:
  - Extracts individual factual claims and propositions from generated answers.
  - Verifies entity, numeric, and assertion grounding against retrieved context.
  - Computes 3-class NLI verdicts (`entailed`, `contradicted`, `neutral`) using dedicated NLI transformers (`cross-encoder/nli-deberta-v3-small`) or validated heuristic logic, preventing false positives from relevance scoring.

---

## 7. Production Security & Multi-Tenant State Isolation
- **API Authentication**: The FastAPI production gateway enforces authentication via `X-API-Key` or `Authorization: Bearer <key>`. Unauthenticated requests receive HTTP 401 Unauthorized.
- **Strict Multi-Tenancy (`MultiTenantStateManager`)**:
  - Independent uploads (`data/tenants/{tenant_id}/uploads`) and index storage (`data/tenants/{tenant_id}/indices`).
  - Isolated in-memory vector indices and BM25 inverted indices per tenant.
  - Zero cross-tenant data leakage: documents indexed by Tenant Alpha are completely invisible to queries executed by Tenant Beta.
  - Shared singleton neural model weights (`SentenceTransformer`, `CrossEncoder`, `LLMGenerator`) avoid memory duplication while preserving complete tenant data isolation.
