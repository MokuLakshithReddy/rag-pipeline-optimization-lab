# Experiment Methodology & Optimization Guide

## 1. Experimental Philosophy: Controlled Single-Variable Isolation
To establish true causality in retrieval improvements, our benchmark framework modifies **one independent variable at a time**:
1. Fixed vs. Recursive Chunking (Holding chunk size, embedding model, and top_k constant).
2. Chunk Size Sensitivity (256 vs. 512 characters).
3. Retrieval Modality (Dense vs. Sparse BM25 vs. Hybrid RRF).
4. Post-Retrieval Cross-Encoder Reranking.

---

## 2. Benchmark Dataset (`data/benchmarks/rag_benchmark.json`)
The benchmark consists of 7 multi-domain technical documents and 8 curated test queries:
- **Conceptual queries**: Test dense vector semantic matching (e.g., Raft leader election, CAP Theorem).
- **Exact technical identifier queries**: Test BM25 keyword precision (e.g., HNSW complexity, CORS wildcard).
- **Multi-factor queries**: Test Reciprocal Rank Fusion.
- **Unanswerable / Out-of-Corpus queries**: Verify system abstention capability without hallucination.

---

## 3. How to Execute Experiments
Run the automated experiment suite:
```powershell
uv run python src/experiments/runner.py
```
This executes all 6 controlled configurations, evaluates metrics across test queries, and updates:
- `reports/benchmark_results.csv`
- `reports/experiment_summary.md`
- `reports/retrieval_analysis.md`

---

## 4. Empirical Summary

| Experiment ID | Strategy | Chunking | Reranker | Hit@K | MRR | NDCG@K | Latency (p95) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense` | Dense | Fixed (256/32) | False | 1.000 | 1.000 | 1.000 | 23.8ms |
| `EXP-02-Chunking-Recursive`| Dense | Recursive (256/32) | False | 1.000 | 1.000 | 1.000 | 0.1ms |
| `EXP-03-Chunking-Size-512` | Dense | Recursive (512/64) | False | 1.000 | 1.000 | 1.000 | 0.1ms |
| `EXP-04-Sparse-BM25` | Sparse | Recursive (512/64) | False | 1.000 | 1.000 | 1.000 | 0.2ms |
| `EXP-05-Hybrid-RRF` | Hybrid | Recursive (512/64) | False | 1.000 | 1.000 | 1.000 | 0.3ms |
| `EXP-06-Hybrid-Reranked` | Hybrid | Recursive (512/64) | True | 1.000 | 1.000 | 1.000 | 1358.8ms |

