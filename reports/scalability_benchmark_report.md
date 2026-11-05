# Production Scalability, Concurrency & Throughput Benchmark Report

- **Evaluation Timestamp**: 2026-11-05 05:27:39 UTC
- **Profiling Tooling**: `psutil` Process Resident Set Size (RSS), `concurrent.futures.ThreadPoolExecutor`
- **Tested Corpus Tiers**: [30, 100, 500] documents (1503 total chunks on largest tier)
- **Tested Concurrency**: [1, 5, 10, 20] concurrent workers

## 1. Corpus Scaling & Subprocess-Isolated Memory Footprint Table

| Corpus Size | Total Chunks | Indexing Time (s) | Baseline RSS | Post-Index RSS | Index Memory Delta | Query Latency (Mean) | Latency (p95) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30 docs** | **93 chunks** | 1.43s | 544.9 MB | 624.1 MB | **+79.24 MB** | **13.95 ms** | **24.66 ms** |
| **100 docs** | **303 chunks** | 3.64s | 543.6 MB | 595.1 MB | **+51.48 MB** | **15.46 ms** | **18.07 ms** |
| **500 docs** | **1503 chunks** | 15.54s | 544.1 MB | 617.6 MB | **+73.48 MB** | **18.37 ms** | **22.85 ms** |

## 2. In-Process Concurrency, Sustained Throughput & Latency Scaling Table

*(Evaluated on large corpus tier: **500 documents**, **1503 chunks** across 100 requests)*

| Concurrency Level | Warm-Cache QPS | Cold-Cache QPS | Warm Latency (Mean) | Warm Latency (p95) | Cold Latency (p95) | Peak RSS | Memory Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 workers** | **205.8 req/s** | **44.3 req/s** | 4.83 ms | **15.25 ms** | 26.7 ms | 614.6 MB | -6.5 MB |
| **5 workers** | **163.5 req/s** | **205.8 req/s** | 29.82 ms | **50.27 ms** | 38.47 ms | 615.6 MB | +0.9 MB |
| **10 workers** | **127.3 req/s** | **129.8 req/s** | 75.29 ms | **189.02 ms** | 107.31 ms | 616.5 MB | +0.9 MB |
| **20 workers** | **133.0 req/s** | **106.9 req/s** | 134.77 ms | **203.49 ms** | 312.14 ms | 617.8 MB | +1.4 MB |

## 3. Engineering Analysis & Production Observations
1. **Corpus Volume & Chunk Alignment**: Each technical document is structured into multiple substantive sections (1,200–1,600 chars), producing **3 to 4 chunks per document**. 500 documents yield **1503 discrete chunks**, establishing genuine multi-chunk retrieval at scale.
2. **Subprocess-Isolated Memory Overhead**: Each corpus tier is evaluated in a dedicated fresh subprocess. Baseline RSS is recorded *after* document and chunk construction, and vector caching is disabled (`enable_cache=False`). The reported Index Memory Delta isolates pure index overhead (sparse BM25 inverted index + dense FAISS vector matrix) without cross-tier memory reuse or cache retention.
3. **Empirical Throughput Scaling (Warm-Cache vs Cold-Cache)**:
   - **In-Process Warm-Cache Retrieval**: When query embeddings are cached, in-process hybrid search on the 500-doc (1503-chunk) tier sustains **127.3 to 205.8 queries per second**, with p95 latencies ranging from 15.2 ms (1 worker) to 203.5 ms under high concurrency (20 workers).
   - **Cold-Cache Neural Inference**: For unseen queries requiring live transformer forward passes on CPU (`all-MiniLM-L6-v2`), sustained throughput reaches **44.3 to 205.8 queries per second** with p95 latencies between 26.7 ms and 312.1 ms.
   - **Scope Qualification**: This benchmark measures in-process retriever throughput; full end-to-end FastAPI HTTP requests will include ASGI framework overhead and network serialization.
4. **Controlled Laboratory Scope**: These results establish rigorous empirical bounds for medium-scale retrieval. Enterprise deployments scaling to millions of documents require distributed sharding (e.g. Qdrant/Milvus cluster) and GPU-accelerated batch reranking.