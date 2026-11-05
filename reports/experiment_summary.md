# Retrieval-Augmented Generation (RAG) Optimization Benchmark Summary

*Generated on: 2026-11-05T02:55:30Z*
*Total Controlled Experiments: 6 Retrieval Ablations + 4 Neural LLM Generation Benchmarks*

## 1. Controlled Retrieval Experiment Results Table

| Experiment ID | Strategy | Chunker (Size/Overlap) | Reranker (Status) | Generator | Hit@K | Precision@K | Recall@K | MRR | NDCG@K | Total Latency (p95) | Delta RSS | Peak Heap |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense-Fixed256` | **dense** | fixed (256/32) | False (none) | mock | **1.000** | 0.200 | 1.000 | **0.919** | **0.938** | 39.2ms | +12.3MB | 0.3MB |
| `EXP-02-Chunking-Recursive256` | **dense** | recursive (256/32) | False (none) | mock | **0.944** | 0.189 | 0.944 | **0.907** | **0.917** | 1.4ms | +25.1MB | 0.2MB |
| `EXP-03-Chunking-Recursive512` | **dense** | recursive (512/64) | False (none) | mock | **1.000** | 0.200 | 1.000 | **0.926** | **0.944** | 1.2ms | +1.5MB | 0.1MB |
| `EXP-04-Sparse-BM25` | **sparse** | recursive (512/64) | False (none) | mock | **1.000** | 0.200 | 1.000 | **0.956** | **0.966** | 1.6ms | +0.1MB | 0.2MB |
| `EXP-05-Hybrid-RRF` | **hybrid** | recursive (512/64) | False (none) | mock | **1.000** | 0.200 | 1.000 | **1.000** | **1.000** | 4.0ms | +0.0MB | 0.2MB |
| `EXP-06-Hybrid-Reranked` | **hybrid** | recursive (512/64) | True (neural_cross_encoder) | mock | **1.000** | 0.200 | 1.000 | **0.972** | **0.979** | 307.1ms | +101.5MB | 92.9MB |

---

## 2. Real Neural LLM Generation & Factual Grounding Results

*Evaluated with active local transformer generator `google/flan-t5-small` and genuine 3-class proposition NLI verification:*

| Experiment ID | Retrieval Pipeline | Generator Model | MRR | NDCG@5 | Answer F1 | Factual Faithfulness | Generation Latency | Total Latency (p95) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `NEURAL-01-Dense` | Dense (all-MiniLM-L6-v2) | `google/flan-t5-small` | **0.926** | 0.944 | **0.214** | **1.000** | 2256.6ms | 6216.1ms |
| `NEURAL-02-Sparse` | Sparse (BM25) | `google/flan-t5-small` | **0.956** | 0.966 | **0.123** | **0.975** | 723.6ms | 1623.7ms |
| `NEURAL-03-Hybrid-RRF` | Hybrid RRF | `google/flan-t5-small` | **1.000** | 1.000 | **0.172** | **0.950** | 829.6ms | 1597.0ms |
| `NEURAL-04-Hybrid-Reranked` | Hybrid + Cross-Encoder | `google/flan-t5-small` | **0.972** | 0.980 | **0.035** | **1.000** | 379.6ms | 1112.5ms |

---

## 3. Baseline vs. Optimized Architecture Comparison

- **Baseline (`EXP-01-Baseline-Dense-Fixed256`)**: Strategy: `dense`, Precision@K: `0.200`, Recall@K: `1.000`, MRR: `0.9185`, NDCG: `0.9382`, Latency (p95): `39.2ms`
- **Optimal MRR Configuration (`EXP-05-Hybrid-RRF`)**: Strategy: `hybrid`, MRR: `1.0000` (**+8.9% relative gain**)
- **Optimal NDCG Configuration (`EXP-05-Hybrid-RRF`)**: Strategy: `hybrid`, NDCG: `1.0000` (**+6.6% relative gain**)
- **Optimal Latency Profile**: `EXP-05-Hybrid-RRF` provides sub-5ms retrieval without the ~300ms cross-attention penalty of `EXP-06`.

---

## 4. Controlled Benchmark Scope & Pareto Analysis

- **Controlled Development Benchmark**: The 30-document / 20-query evaluation set serves as a targeted testbed to isolate retrieval variables (chunking granularity, sparse vs dense representations, rank fusion formulas). Perfect ranking on this set does not imply universal superiority across unconstrained web domains.
- **Pareto Trade-Off**:
  - `Hybrid RRF`: 4.0ms p95, 0MB Delta RSS, 1.000 MRR on development corpus. Ideal for low-latency SLA (<20ms) and low-resource environments.
  - `Hybrid + Cross-Encoder`: 307.1ms p95, +101.5MB Delta RSS, 0.972 MRR. Provides deep cross-attention for complex queries, but requires GPU acceleration to achieve real-time latency.

---

## 5. Methodological & Verification Rigor

- **Genuine NLI Verification**: Evaluated using `FactualEntailmentEvaluator` (claim decomposition, entity grounding, 3-class entailment/neutral/contradiction) rather than conflating MS-MARCO query-passage relevance scores with premise-hypothesis entailment.
- **Incremental Memory (Delta RSS)**: Measures net physical memory change (`RSS_after - RSS_before`) attributable to that experiment pass, capturing native C++/PyTorch allocations as well as Python heap.
- **Strict Reranker Validation**: Strict mode is enforced on Cross-Encoder reranking (`strict_mode=True`); silent fallback to lexical reranking is prohibited and raises a hard runtime exception.
