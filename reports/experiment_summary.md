# Retrieval-Augmented Generation (RAG) Optimization Benchmark Summary

*Generated on: 2026-11-04T20:02:43.060142+00:00*
*Total Experiments Executed: 6*

## 1. Controlled Experiment Results Table

| Experiment ID | Strategy | Chunker (Size/Overlap) | Reranker (Status) | Generator | Hit@K | Precision@K | Recall@K | MRR | NDCG@K | Total Latency (p95) | Delta RSS | Peak Heap |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-01-Baseline-Dense-Fixed256` | **dense** | fixed (256/32) | False (none) | mock | **1.000** | 0.200 | 1.000 | **0.919** | **0.938** | 39.2ms | +12.3MB | 0.3MB |
| `EXP-02-Chunking-Recursive256` | **dense** | recursive (256/32) | False (none) | mock | **0.944** | 0.189 | 0.944 | **0.907** | **0.917** | 1.4ms | +25.1MB | 0.2MB |
| `EXP-03-Chunking-Recursive512` | **dense** | recursive (512/64) | False (none) | mock | **1.000** | 0.200 | 1.000 | **0.926** | **0.944** | 1.2ms | +1.5MB | 0.1MB |
| `EXP-04-Sparse-BM25` | **sparse** | recursive (512/64) | False (none) | mock | **1.000** | 0.200 | 1.000 | **0.956** | **0.966** | 1.6ms | +0.1MB | 0.2MB |
| `EXP-05-Hybrid-RRF` | **hybrid** | recursive (512/64) | False (none) | mock | **1.000** | 0.200 | 1.000 | **1.000** | **1.000** | 4.0ms | +0.0MB | 0.2MB |
| `EXP-06-Hybrid-Reranked` | **hybrid** | recursive (512/64) | True (neural_cross_encoder) | mock | **1.000** | 0.200 | 1.000 | **0.972** | **0.979** | 307.1ms | +101.5MB | 92.9MB |

## 2. Baseline vs. Optimized Architecture Comparison

- **Baseline (`EXP-01-Baseline-Dense-Fixed256`)**: Strategy: `dense`, Precision@K: `0.200`, Recall@K: `1.000`, MRR: `0.9185`, NDCG: `0.9382`, Latency (p95): `39.2ms`
- **Optimal MRR Configuration (`EXP-05-Hybrid-RRF`)**: Strategy: `hybrid`, MRR: `1.0000` (**+8.9% relative gain**)
- **Optimal NDCG Configuration (`EXP-05-Hybrid-RRF`)**: Strategy: `hybrid`, NDCG: `1.0000` (**+6.6% relative gain**)

## 3. Resource & Verification Methodology Notes

- **Incremental Memory (Delta RSS)**: Measures the net physical memory change (`RSS_after - RSS_before`) attributable specifically to that experiment pass, capturing native C++/PyTorch allocations as well as Python heap.
- **Peak Heap**: Measures maximum Python-managed heap allocations traced via `tracemalloc`.
- **Strict Reranker Validation**: Strict mode is enforced on Cross-Encoder reranking (`strict_mode=True`); silent fallback to lexical reranking is prohibited and raises a hard runtime exception.
- **Answer Generation Evidence**: When executed with `provider='mock'`, answer F1 and faithfulness reflect deterministic extractive baseline heuristics. Run `python src/experiments/runner.py --llm-provider local` or `--llm-provider gemini` to benchmark neural generation.
- **Factual Faithfulness**: Evaluates sentence-level claim factual entailment and entity grounding against retrieved evidence passages, rather than bag-of-words token presence.