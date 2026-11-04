# Deep Retrieval Systems & Trade-Off Analysis

## Executive Summary
This report analyzes retrieval performance across dense vector search, sparse keyword BM25 retrieval, reciprocal rank fusion (RRF) hybrid search, and cross-encoder reranking. Controlled single-variable experiments were conducted on a diverse 30-document multi-domain technical corpus with 20 ground-truth queries to determine causal performance drivers.

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
  - *Result*: Achieves highest overall MRR and NDCG across diverse query types. Rank fusion eliminates score distribution mismatch between unbounded BM25 scores and bounded cosine similarities.

---

## 3. Cross-Encoder Reranking Impact
- **Ranking Quality**: Evaluating query-document pairs simultaneously via transformer cross-attention resolves ambiguous candidate orderings produced by initial bi-encoder retrieval.
- **Latency & Resource Trade-Off**: Cross-Encoder attention adds ~30ms-70ms inference latency per query on CPU.
- **Deployment Recommendation**: For ultra-low latency requirements (<10ms SLA), use pure RRF Hybrid retrieval. For maximum answer accuracy on complex reasoning tasks, use Hybrid Retrieval + Cross-Encoder Reranking.

---

## 4. Failure-Case Analysis & Mitigation
1. **The 'Lost in the Middle' Problem**:
   - *Observation*: LLMs exhibit higher recall for information located at the extreme beginning and end of long contexts.
   - *Mitigation*: Our `ContextOptimizer` alternates high-ranked chunks between the beginning and the end of the constructed prompt context.
2. **Hallucination on Out-of-Corpus Questions**:
   - *Observation*: Without strict thresholding, retrievers return irrelevant chunks with low similarity, leading LLMs to fabricate answers.
   - *Mitigation*: The `LLMGenerator` enforces an explicit abstention threshold and returns a standardized refusal message when grounding is insufficient.
