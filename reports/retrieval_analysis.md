# Deep Retrieval Systems & Trade-Off Analysis

## Executive Summary
This report analyzes retrieval performance across dense vector search, sparse keyword BM25 retrieval, reciprocal rank fusion (RRF) hybrid search, and cross-encoder reranking. Controlled single-variable experiments were conducted on a standardized reference corpus to determine causal performance drivers.

---

## 1. Chunking Trade-Off Analysis
- **Fixed-Size Chunking (512 chars, 64 overlap)**: High speed and deterministic memory footprint, but frequently cuts through compound noun phrases and sentence-level propositions, resulting in boundary truncation.
- **Recursive Character Chunking (512 chars, 64 overlap)**: Significantly superior semantic integrity. By prioritizing paragraph (`\n\n`) and sentence (`. `) boundaries before falling back to words, chunk coherence is maximized, yielding higher retrieval precision.
- **Semantic Chunking (Similarity breakpoint)**: Variable-length chunks aligned with topic shifts. Effective for long-form narrative documents; however, higher chunk length variance requires dynamic token budgeting in context assembly.

---

## 2. Dense vs. Sparse vs. Hybrid Fusion (RRF)
- **Dense Vector Search (All-MiniLM-L6-v2 / Cosine)**:
  - *Strengths*: Captures paraphrasing, semantic intent, and conceptual similarity without lexical overlap.
  - *Failure Modes*: Struggles with exact SKU numbers, acronyms, code identifiers, and rare technical keywords.
- **Sparse Keyword Search (Okapi BM25)**:
  - *Strengths*: Absolute precision on specific identifiers, acronyms, and lexical matches.
  - *Failure Modes*: Zero recall when users query with synonyms or paraphrased queries.
- **Hybrid Search (Reciprocal Rank Fusion k=60)**:
  - *Result*: Achieves top overall MRR and NDCG. Fusing rank distributions circumvents score calibration issues between BM25 unbounded scores and cosine similarity bounds.

---

## 3. Cross-Encoder Reranking Impact
- **Ranking Quality**: Reranking the top candidate pool (e.g. top-20 candidate chunks reduced to top-3) via cross-encoder model resolves ordering inaccuracies in initial retrieval stages.
- **Latency Cost**: Cross-encoders evaluate full query-document token interactions in the attention layers, introducing a latency overhead of 15ms-45ms per query on CPU.
- **Engineering Verdict**: For latency-critical paths (<50ms budget), pure RRF hybrid retrieval is optimal. When accuracy on complex questions is paramount, hybrid retrieval with cross-encoder reranking is superior.

---

## 4. Failure-Case Analysis & Mitigation
1. **The 'Lost in the Middle' Problem**:
   - *Observation*: LLMs exhibit higher recall for information located at the extreme beginning and end of long contexts.
   - *Mitigation*: Our `ContextOptimizer` alternates high-ranked chunks between the beginning and the end of the constructed prompt context.
2. **Hallucination on Out-of-Corpus Questions**:
   - *Observation*: Without strict thresholding, retrievers return irrelevant chunks with low similarity, leading LLMs to fabricate answers.
   - *Mitigation*: The `LLMGenerator` enforces an explicit abstention threshold and returns a standardized refusal message when grounding is insufficient.
