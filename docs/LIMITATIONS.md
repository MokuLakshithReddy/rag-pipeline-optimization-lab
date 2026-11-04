# Known Limitations & Engineering Assumptions

## 1. Kaggle Competition Inaccessibility
- **Access Status**: Direct public fetch to `https://www.kaggle.com/competitions/gen-ai-project-4-building-optimizing-rag-pipelines` returns HTTP 404.
- **Unresolved Factors**: Exact competition training/test CSV schema and specific ground-truth target files cannot be verified until user credentials or platform access are granted.
- **Resolution Path**: Follow instructions in `docs/COMPETITION_SPEC.md` to configure Kaggle CLI. The notebook `notebooks/competition_solution.ipynb` contains an adapter ready to ingest the actual files.

## 2. Hardware & Runtime Constraints
- **GPU Acceleration**: Dense embedding and Cross-Encoder reranking run on CPU by default. While fast for batches of up to several hundred chunks, large corpora (>100,000 chunks) should be run with CUDA (`device="cuda"`).
- **In-Memory Indices**: Currently uses numpy-based vector search and in-memory BM25. For datasets exceeding 1,000,000 documents, persistent distributed indices (e.g., Faiss on disk, Qdrant, or Milvus) are recommended.

## 3. External LLM Generation
- By default, `LLM_PROVIDER=mock` executes deterministic extractive answer synthesis to provide zero-cost, fully offline reproducibility.
- Setting `LLM_PROVIDER=gemini` requires exporting `GEMINI_API_KEY`.

