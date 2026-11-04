# Retrieval-Augmented Generation (RAG) Optimization Benchmark Summary

*Generated on: 2026-11-04T19:17:22.156683+00:00*
*Total Experiments Executed: 6*

## 1. Controlled Experiment Results Table

| Experiment ID | Strategy | Chunker (Size/Overlap) | Reranker (Status) | Hit@K | Precision@K | Recall@K | MRR | NDCG@K | Total Latency (p95) | Peak Heap |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense-Fixed256` | **dense** | fixed (256/32) | False (none) | **1.000** | 0.200 | 1.000 | **0.919** | **0.938** | 36.0ms | 0.3MB |
| `EXP-02-Chunking-Recursive256` | **dense** | recursive (256/32) | False (none) | **0.944** | 0.189 | 0.944 | **0.907** | **0.917** | 0.5ms | 0.2MB |
| `EXP-03-Chunking-Recursive512` | **dense** | recursive (512/64) | False (none) | **1.000** | 0.200 | 1.000 | **0.926** | **0.944** | 0.6ms | 0.1MB |
| `EXP-04-Sparse-BM25` | **sparse** | recursive (512/64) | False (none) | **1.000** | 0.200 | 1.000 | **0.956** | **0.966** | 1.0ms | 0.2MB |
| `EXP-05-Hybrid-RRF` | **hybrid** | recursive (512/64) | False (none) | **1.000** | 0.200 | 1.000 | **1.000** | **1.000** | 1.4ms | 0.2MB |
| `EXP-06-Hybrid-Reranked` | **hybrid** | recursive (512/64) | True (neural_cross_encoder) | **1.000** | 0.200 | 1.000 | **0.972** | **0.979** | 278.6ms | 92.8MB |

## 2. Baseline vs. Optimized Architecture Comparison

- **Baseline (`EXP-01-Baseline-Dense-Fixed256`)**: Strategy: `dense`, Precision@K: `0.200`, Recall@K: `1.000`, MRR: `0.9185`, NDCG: `0.9382`, Latency (p95): `36.0ms`
- **Optimal MRR Configuration (`EXP-05-Hybrid-RRF`)**: Strategy: `hybrid`, MRR: `1.0000` (**+8.9% relative gain**)
- **Optimal NDCG Configuration (`EXP-05-Hybrid-RRF`)**: Strategy: `hybrid`, NDCG: `1.0000` (**+6.6% relative gain**)