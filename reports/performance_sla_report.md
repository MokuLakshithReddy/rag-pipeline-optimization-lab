# Retrieval Performance & Latency SLA Report

*Evaluated with 3 repetitions per query after 2 warm-up iterations.*
*Total queries per repetition: 20*

## 1. Latency Benchmark Summary Table
| Strategy | Mean Latency | Target (Mean) | p50 | p95 | Target (p95) | SLA Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sparse_bm25` | **0.6ms** | <= 5.0ms | 0.1ms | **0.3ms** | <= 15.0ms | **PASSED** |
| `dense_vector` | **3.1ms** | <= 25.0ms | 0.0ms | **11.0ms** | <= 50.0ms | **PASSED** |
| `hybrid_rrf` | **0.2ms** | <= 10.0ms | 0.2ms | **0.4ms** | <= 25.0ms | **PASSED** |
| `hybrid_reranked_cpu` | **49.9ms** | <= 350.0ms | 48.6ms | **54.7ms** | <= 600.0ms | **PASSED** |

## 2. SLA Analysis & Observations
- **Sparse Okapi BM25**: Fastest execution on CPU; strictly keyword lookup with zero neural transformer overhead.
- **Dense SentenceTransformers**: Evaluates `all-MiniLM-L6-v2` query bi-encoder inference (~2-10ms on CPU).
- **Hybrid Reciprocal Rank Fusion (RRF)**: Merges sparse and dense rank lists with minimal fusion overhead (<1ms).
- **Cross-Encoder Reranking**: Applies deep cross-attention on top candidate pairs (~40-80ms per query on CPU). Target ceiling of 350ms provides appropriate headroom for shared CI environments while strictly catching any real regressions.