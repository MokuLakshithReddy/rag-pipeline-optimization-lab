# Retrieval Performance & Latency SLA Report

*Evaluated with 3 repetitions per query after 2 warm-up iterations.*
*Total queries per repetition: 20*

## 1. Latency Benchmark Summary Table
| Strategy | Mean Latency | Target (Mean) | p50 | p95 | Target (p95) | SLA Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sparse_bm25` | **0.1ms** | <= 25.0ms | 0.1ms | **0.2ms** | <= 50.0ms | **PASSED** |
| `dense_vector` | **2.9ms** | <= 60.0ms | 0.0ms | **10.2ms** | <= 120.0ms | **PASSED** |
| `hybrid_rrf` | **0.2ms** | <= 75.0ms | 0.2ms | **0.3ms** | <= 150.0ms | **PASSED** |
| `hybrid_reranked_cpu` | **45.2ms** | <= 3500.0ms | 44.7ms | **53.3ms** | <= 5000.0ms | **PASSED** |

## 2. SLA Analysis & Observations
- **Sparse Okapi BM25**: Fastest execution on CPU; strictly keyword lookup with zero neural transformer overhead.
- **Dense SentenceTransformers**: Evaluates `all-MiniLM-L6-v2` query bi-encoder inference (~10-30ms on CPU).
- **Hybrid Reciprocal Rank Fusion (RRF)**: Merges sparse and dense rank lists with minimal fusion overhead (<5ms).
- **Cross-Encoder Reranking**: Applies deep cross-attention on top candidate pairs (~300-800ms per query on CPU). Highly sensitive to VM thread contention.