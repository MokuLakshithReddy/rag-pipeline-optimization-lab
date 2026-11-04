# Retrieval-Augmented Generation (RAG) Optimization Benchmark Summary

*Generated on: 2026-11-04T18:50:25.982527+00:00*
*Total Experiments Executed: 6*

## 1. Controlled Experiment Results Table

| Experiment ID | Strategy | Chunker (Size/Overlap) | Embedding Model | Reranker | Hit@K | MRR | NDCG@K | Latency (p95 ms) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense` | **dense** | fixed (256/32) | `all-MiniLM-L6-v2` | False | **1.000** | **1.000** | **1.000** | 23.8ms |
| `EXP-02-Chunking-Recursive` | **dense** | recursive (256/32) | `all-MiniLM-L6-v2` | False | **1.000** | **1.000** | **1.000** | 0.1ms |
| `EXP-03-Chunking-Size-512` | **dense** | recursive (512/64) | `all-MiniLM-L6-v2` | False | **1.000** | **1.000** | **1.000** | 0.1ms |
| `EXP-04-Sparse-BM25` | **sparse** | recursive (512/64) | `all-MiniLM-L6-v2` | False | **1.000** | **1.000** | **1.000** | 0.2ms |
| `EXP-05-Hybrid-RRF` | **hybrid** | recursive (512/64) | `all-MiniLM-L6-v2` | False | **1.000** | **1.000** | **1.000** | 0.3ms |
| `EXP-06-Hybrid-Reranked` | **hybrid** | recursive (512/64) | `all-MiniLM-L6-v2` | True | **1.000** | **1.000** | **1.000** | 1358.8ms |

## 2. Baseline vs. Optimized Architecture Comparison

- **Baseline (`EXP-01-Baseline-Dense`)**: Strategy: `dense`, MRR: `1.0000`, NDCG: `1.0000`, Latency: `20.7ms`
- **Optimal MRR Configuration (`EXP-01-Baseline-Dense`)**: Strategy: `dense`, MRR: `1.0000` (**+0.0% relative gain**)
- **Optimal NDCG Configuration (`EXP-01-Baseline-Dense`)**: Strategy: `dense`, NDCG: `1.0000` (**+0.0% relative gain**)