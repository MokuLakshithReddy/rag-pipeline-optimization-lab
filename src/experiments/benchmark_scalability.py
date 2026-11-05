"""Production Scalability, Concurrency, Throughput, and Memory Benchmark Suite.

Evaluates RAG pipeline scaling across:
1. Multi-scale corpus tiers:
   - Small: 30 documents (~93 chunks)
   - Medium: 100 documents (~303 chunks)
   - Large / Stress: 500 documents (~1,503 chunks)
2. Subprocess-isolated Resident Set Size (RSS) memory profiling via psutil:
   - Evaluates each tier in a dedicated fresh subprocess.
   - Measures memory baseline AFTER document/chunk construction.
   - Disables embedding vector cache (enable_cache=False) to avoid duplicate dict retention.
   - Reports genuine index memory delta (post_index_rss - baseline_rss).
3. Concurrency stress testing:
   - 1, 5, 10, 20 concurrent workers
   - In-Process Warm-Cache Retrieval Throughput (QPS)
   - Cold-Cache Neural Embedding & Retrieval Throughput (QPS)
4. Generates reports/scalability_benchmark_report.json and reports/scalability_benchmark_report.md.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import psutil

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.embeddings.registry import get_embedder
from src.ingestion.models import Document, DocumentMetadata
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scalability_benchmark")


def get_current_rss_mb() -> float:
    """Return current process Resident Set Size in Megabytes after garbage collection."""
    gc.collect()
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _build_rich_technical_doc(doc_id: str, title: str, domain: str, body: str, counter: int) -> Document:
    """Construct multi-paragraph technical document (1,200 - 1,600 chars) producing 3-4 chunks."""
    full_text = (
        f"# Technical Reference: {title} ({domain.upper()})\n\n"
        f"## 1. Architectural Foundations\n"
        f"{body}\n"
        f"In distributed and enterprise architectures, this pattern guarantees predictable latency "
        f"and bounded state complexity under high concurrency.\n\n"
        f"## 2. Protocol Internals & Invariants\n"
        f"The execution model decomposes state transitions into discrete monotonic phases. "
        f"Replication logs enforce strict linearizability through epoch numbers and term counters. "
        f"Replicas reject stale proposals using term comparisons, preventing split-brain states.\n\n"
        f"## 3. Operational Constraints & Production Tuning\n"
        f"Production deployments must monitor network round-trip times and CPU cache line contention. "
        f"Recommended timeout configurations should accommodate tail latencies up to the 99.9th percentile. "
        f"System integrity depends on robust checkpointing and deterministic replay mechanisms. "
        f"(Document Archive Index: {counter:05d}, Domain: {domain})."
    )
    return Document(
        id=doc_id,
        text=full_text,
        metadata=DocumentMetadata(
            source=f"{doc_id}.md",
            doc_id=doc_id,
            file_type=".md"
        )
    )


def generate_scaled_corpus(target_doc_count: int) -> List[Document]:
    """Generate high-fidelity technical corpus of arbitrary size for stress testing."""
    base_path = Path("data/benchmarks/rag_benchmark.json")
    with open(base_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_corpus = data["corpus"]
    documents: List[Document] = []

    # 1. Expand base corpus items into rich technical documentation
    for idx, item in enumerate(base_corpus):
        doc = _build_rich_technical_doc(
            doc_id=item["id"],
            title=item["source"].replace(".md", "").replace("_", " ").title(),
            domain="systems",
            body=item["text"],
            counter=idx
        )
        documents.append(doc)

    if len(documents) >= target_doc_count:
        return documents[:target_doc_count]

    # 2. Procedural expansion across systems, databases, ML, networking, security
    domains = [
        ("consensus", [
            ("Raft Leader Leases", "Raft leader leases guarantee read-only linearizability without quorum round-trips by bounding clock drift."),
            ("Zab Atomic Broadcast", "ZooKeeper Atomic Broadcast (Zab) enforces causal ordering through a designated primary proposing transactions with monotonic epoch and zxid counters."),
            ("Viewstamped Replication", "Viewstamped Replication separates state machine execution from replica group coordination across configuration view changes."),
            ("Spanner TrueTime API", "Google Spanner synchronizes distributed multi-datacenter transactions using TrueTime API bounded uncertainty intervals with atomic GPS and rubidium clocks.")
        ]),
        ("storage", [
            ("WiscKey Key-Value Separation", "WiscKey minimizes LSM write amplification by separating keys from values, storing keys in SSTables and append-only values in a separate vLog."),
            ("Fractured Mirrors Layout", "The Fractured Mirrors database storage model combines DSM columnar memory efficiency with NSM row-oriented storage for hybrid transactional/analytical processing."),
            ("B-epsilon Tree Buffers", "B-epsilon trees place in-node message buffers along search paths to batch writes, achieving an asymptotic trade-off between write-optimized LSMs and read-optimized B+ trees."),
            ("ZNS Flash SSD Storage", "Zoned Namespaces (ZNS) expose raw sequential write zones directly to database software, completely eliminating device-level Flash Translation Layer (FTL) garbage collection pauses.")
        ]),
        ("networking", [
            ("eBPF XDP Packet Filtering", "eBPF eXpress Data Path (XDP) executes sandboxed kernel bytecode directly within the network driver NIC ring buffer, dropping DDoS traffic at 100M packets per second."),
            ("BBR Congestion Control", "Bottleneck Bandwidth and RTT (BBR) congestion control measures maximum delivery rate and minimum round-trip time rather than reacting to packet loss."),
            ("HTTP/3 Multiplexing", "HTTP/3 uses QUIC over UDP to establish zero-RTT handshake multiplexing, preventing HOL blocking when packet loss occurs on individual streams."),
            ("RDMA RoCEv2 Kernel Bypass", "RDMA over Converged Ethernet (RoCEv2) enables direct remote memory access without CPU interrupts or OS kernel TCP stack context switching.")
        ]),
        ("ai_systems", [
            ("vLLM PagedAttention Memory", "PagedAttention eliminates KV-cache memory fragmentation in LLM inference by allocating key-value blocks non-contiguously in virtual memory pages similar to OS virtual memory."),
            ("Speculative Decoding Draft", "Speculative decoding uses a lightweight draft model to generate K candidate tokens in parallel, verified in a single forward pass of the target model."),
            ("Tensor Parallelism Communication", "Megatron-LM tensor parallelism splits weight matrices across GPUs along row and column dimensions, communicating activations via All-Reduce collectives."),
            ("FP8 Mixed Precision", "FP8 mixed-precision training alternates between E4M3 for forward activations and weights and E5M2 for backward gradients with wider dynamic range.")
        ]),
        ("kernel_os", [
            ("io_uring Ring Buffers", "Linux io_uring provides zero-copy asynchronous system call dispatch via shared submission and completion ring buffers in kernel memory."),
            ("Read-Copy Update Synchronization", "RCU lock-free synchronization allows read operations to proceed without locks while deferring reclaim of old data versions until all reader grace periods finish."),
            ("cgroups v2 Resource Hierarchy", "Control Groups (cgroups v2) unifies CPU, memory, IO, and pid controller hierarchies to enforce resource limits on container workloads without cross-controller priority inversions."),
            ("Transparent Huge Pages", "Transparent Huge Pages map 2MB physical pages to amortize TLB miss overhead in database buffer pools, requiring MADV_HUGEPAGE tuning to avoid memory fragmentation.")
        ]),
        ("security", [
            ("AES-GCM-SIV Nonce Resistance", "AES-GCM-SIV provides synthetic initialization vector authentication, preventing catastrophic plaintext disclosure even if an encryption nonce is accidentally reused."),
            ("Post-Quantum ML-KEM Kyber", "ML-KEM (CRYSTALS-Kyber) is a lattice-based module Learning-With-Errors key encapsulation mechanism standardized for post-quantum cryptographic security."),
            ("AMD SEV-SNP Memory Isolation", "AMD SEV-SNP provides hardware-enforced memory encryption and remote attestation, protecting VM memory from untrusted host hypervisors and physical memory probing."),
            ("ZK-SNARK Polynomial Proofs", "ZK-SNARK proofs use KZG or FRI polynomial commitments to prove arithmetic circuit satisfiability in constant time without revealing secret witness values.")
        ])
    ]

    counter = len(documents)
    while len(documents) < target_doc_count:
        for domain, entries in domains:
            for title, content in entries:
                if len(documents) >= target_doc_count:
                    break
                doc_id = f"scaled_{domain}_{counter:04d}"
                doc = _build_rich_technical_doc(doc_id, title, domain, content, counter)
                documents.append(doc)
                counter += 1

    return documents


def evaluate_single_tier_isolated(doc_count: int) -> Dict[str, Any]:
    """Execute evaluation of a single corpus tier inside an isolated subprocess."""
    base_path = Path("data/benchmarks/rag_benchmark.json")
    with open(base_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [q["query"] for q in data["benchmark_queries"]]

    # 1. Instantiate embedder with embedding cache disabled to prevent dictionary retention
    embedder = get_embedder(model_name="all-MiniLM-L6-v2", enable_cache=False)

    # 2. Generate and chunk corpus
    t_chunk_start = time.perf_counter()
    docs = generate_scaled_corpus(doc_count)
    chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(docs)
    chunk_time_s = time.perf_counter() - t_chunk_start

    # 3. Clean garbage collection & capture BASELINE RSS AFTER document/chunk construction
    gc.collect()
    baseline_rss_mb = get_current_rss_mb()

    # 4. Build Sparse BM25 Index
    t_sparse_start = time.perf_counter()
    sparse = SparseRetriever()
    sparse.index_chunks(chunks)
    sparse_index_time_s = time.perf_counter() - t_sparse_start

    # 5. Build Dense Vector Index
    t_dense_start = time.perf_counter()
    dense = DenseRetriever(embedder=embedder)
    dense.index_chunks(chunks)
    dense_index_time_s = time.perf_counter() - t_dense_start

    # 6. Capture POST-INDEX RSS and calculate TRUE index memory delta
    gc.collect()
    post_index_rss_mb = get_current_rss_mb()
    index_memory_delta_mb = round(post_index_rss_mb - baseline_rss_mb, 2)

    # 7. Single-query latency measurement for Hybrid RRF
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, fusion_method="rrf", rrf_k=60)
    for q in queries[:2]:
        _ = hybrid.retrieve(q, top_k=5)

    latencies_ms = []
    for q in queries:
        t0 = time.perf_counter()
        _ = hybrid.retrieve(q, top_k=5)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "doc_count": doc_count,
        "total_chunks": len(chunks),
        "chunking_time_s": round(chunk_time_s, 2),
        "sparse_index_time_s": round(sparse_index_time_s, 3),
        "dense_index_time_s": round(dense_index_time_s, 2),
        "total_index_time_s": round(sparse_index_time_s + dense_index_time_s, 2),
        "baseline_rss_mb": round(baseline_rss_mb, 1),
        "post_index_rss_mb": round(post_index_rss_mb, 1),
        "index_memory_delta_mb": index_memory_delta_mb,
        "query_latency_mean_ms": round(float(np.mean(latencies_ms)), 2),
        "query_latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "query_latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2)
    }
    return result


def evaluate_concurrency_and_throughput(
    hybrid_pipeline: HybridRetriever,
    queries: List[str],
    concurrency_levels: List[int] = [1, 5, 10, 20],
    requests_per_test: int = 100
) -> List[Dict[str, Any]]:
    """Benchmark sustained throughput (QPS) and latency under warm-cache and cold-cache query conditions."""
    results: List[Dict[str, Any]] = []

    # Warm up pipeline
    for q in queries[:3]:
        _ = hybrid_pipeline.retrieve(q, top_k=5)

    for concurrency in concurrency_levels:
        logger.info(f"Testing Concurrency Level: {concurrency} workers ({requests_per_test} requests)...")
        rss_before = get_current_rss_mb()

        # 1. Warm-Cache Workload (repeated queries from pool hitting query cache)
        warm_workload = [queries[i % len(queries)] for i in range(requests_per_test)]
        warm_latencies_ms: List[float] = []

        def _execute_query(q_str: str) -> float:
            t0 = time.perf_counter()
            _ = hybrid_pipeline.retrieve(q_str, top_k=5)
            return (time.perf_counter() - t0) * 1000.0

        t_warm_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_execute_query, q) for q in warm_workload]
            for fut in concurrent.futures.as_completed(futures):
                warm_latencies_ms.append(fut.result())
        total_warm_time_s = time.perf_counter() - t_warm_start

        warm_qps = requests_per_test / total_warm_time_s if total_warm_time_s > 0 else 0.0

        # 2. Cold-Cache Workload (distinct synthetic queries forcing neural transformer embedding)
        cold_workload = [f"{queries[i % len(queries)]} (context variation {i})" for i in range(requests_per_test)]
        cold_latencies_ms: List[float] = []

        t_cold_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_execute_query, q) for q in cold_workload]
            for fut in concurrent.futures.as_completed(futures):
                cold_latencies_ms.append(fut.result())
        total_cold_time_s = time.perf_counter() - t_cold_start

        cold_qps = requests_per_test / total_cold_time_s if total_cold_time_s > 0 else 0.0

        rss_after = get_current_rss_mb()

        results.append({
            "concurrency_level": concurrency,
            "total_requests": requests_per_test,
            "warm_cache_time_s": round(total_warm_time_s, 2),
            "warm_cache_throughput_qps": round(warm_qps, 1),
            "warm_latency_mean_ms": round(float(np.mean(warm_latencies_ms)), 2),
            "warm_latency_p50_ms": round(float(np.percentile(warm_latencies_ms, 50)), 2),
            "warm_latency_p95_ms": round(float(np.percentile(warm_latencies_ms, 95)), 2),
            "warm_latency_p99_ms": round(float(np.percentile(warm_latencies_ms, 99)), 2),
            "cold_cache_throughput_qps": round(cold_qps, 1),
            "cold_latency_mean_ms": round(float(np.mean(cold_latencies_ms)), 2),
            "cold_latency_p95_ms": round(float(np.percentile(cold_latencies_ms, 95)), 2),
            "peak_rss_mb": round(rss_after, 1),
            "memory_delta_mb": round(rss_after - rss_before, 1)
        })

    return results


def run_scalability_benchmark(
    reports_dir: str | Path = "reports",
    corpus_tiers: List[int] = [30, 100, 500],
    concurrency_levels: List[int] = [1, 5, 10, 20]
) -> Dict[str, Any]:
    """Execute end-to-end scalability benchmark orchestrating isolated subprocesses per tier."""
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    base_path = Path("data/benchmarks/rag_benchmark.json")
    with open(base_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [q["query"] for q in data["benchmark_queries"]]

    # 1. Benchmark Each Corpus Size Tier in an Isolated Subprocess
    tier_results = []
    for doc_count in corpus_tiers:
        logger.info(f"Spawning isolated subprocess for Corpus Tier: {doc_count} Documents...")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single-tier", str(doc_count)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        tier_data = None
        for line in proc.stdout.splitlines():
            if line.startswith("__TIER_JSON__"):
                tier_data = json.loads(line.replace("__TIER_JSON__", "").strip())
                break
        if not tier_data:
            raise RuntimeError(f"Subprocess for tier {doc_count} failed to return metrics. Stderr: {proc.stderr}")
        tier_results.append(tier_data)
        logger.info(f"Tier {doc_count} completed: chunks={tier_data['total_chunks']}, delta_rss={tier_data['index_memory_delta_mb']} MB")

    # 2. Build pipeline for largest tier to evaluate concurrency and throughput
    largest_doc_count = max(corpus_tiers)
    logger.info(f"Preparing pipeline for {largest_doc_count}-doc concurrency stress test...")
    embedder = get_embedder(model_name="all-MiniLM-L6-v2", enable_cache=True)
    docs = generate_scaled_corpus(largest_doc_count)
    chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(docs)

    sparse = SparseRetriever()
    sparse.index_chunks(chunks)

    dense = DenseRetriever(embedder=embedder)
    dense.index_chunks(chunks)

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, fusion_method="rrf", rrf_k=60)

    concurrency_results = evaluate_concurrency_and_throughput(
        hybrid_pipeline=hybrid,
        queries=queries,
        concurrency_levels=concurrency_levels,
        requests_per_test=100
    )

    summary_data = {
        "benchmark_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluated_corpus_tiers": tier_results,
        "concurrency_stress_test": concurrency_results
    }

    # Save JSON Report
    json_path = reports / "scalability_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Dynamic extraction of actual measured throughput values to prevent outdated claims
    warm_qps_min = min(c["warm_cache_throughput_qps"] for c in concurrency_results)
    warm_qps_max = max(c["warm_cache_throughput_qps"] for c in concurrency_results)
    warm_p95_min = min(c["warm_latency_p95_ms"] for c in concurrency_results)
    warm_p95_max = max(c["warm_latency_p95_ms"] for c in concurrency_results)

    cold_qps_min = min(c["cold_cache_throughput_qps"] for c in concurrency_results)
    cold_qps_max = max(c["cold_cache_throughput_qps"] for c in concurrency_results)
    cold_p95_min = min(c["cold_latency_p95_ms"] for c in concurrency_results)
    cold_p95_max = max(c["cold_latency_p95_ms"] for c in concurrency_results)

    # Save Markdown Report
    md_path = reports / "scalability_benchmark_report.md"
    lines = [
        "# Production Scalability, Concurrency & Throughput Benchmark Report",
        f"\n- **Evaluation Timestamp**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- **Profiling Tooling**: `psutil` Process Resident Set Size (RSS), `concurrent.futures.ThreadPoolExecutor`",
        f"- **Tested Corpus Tiers**: {corpus_tiers} documents ({tier_results[-1]['total_chunks']} total chunks on largest tier)",
        f"- **Tested Concurrency**: {concurrency_levels} concurrent workers\n",
        "## 1. Corpus Scaling & Subprocess-Isolated Memory Footprint Table\n",
        "| Corpus Size | Total Chunks | Indexing Time (s) | Baseline RSS | Post-Index RSS | Index Memory Delta | Query Latency (Mean) | Latency (p95) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for t in tier_results:
        lines.append(
            f"| **{t['doc_count']} docs** | **{t['total_chunks']} chunks** | {t['total_index_time_s']}s | "
            f"{t['baseline_rss_mb']} MB | {t['post_index_rss_mb']} MB | **+{t['index_memory_delta_mb']} MB** | "
            f"**{t['query_latency_mean_ms']} ms** | **{t['query_latency_p95_ms']} ms** |"
        )

    lines.extend([
        "\n## 2. In-Process Concurrency, Sustained Throughput & Latency Scaling Table\n",
        f"*(Evaluated on large corpus tier: **{tier_results[-1]['doc_count']} documents**, **{tier_results[-1]['total_chunks']} chunks** across 100 requests)*\n",
        "| Concurrency Level | Warm-Cache QPS | Cold-Cache QPS | Warm Latency (Mean) | Warm Latency (p95) | Cold Latency (p95) | Peak RSS | Memory Delta |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for c in concurrency_results:
        lines.append(
            f"| **{c['concurrency_level']} workers** | **{c['warm_cache_throughput_qps']} req/s** | "
            f"**{c['cold_cache_throughput_qps']} req/s** | {c['warm_latency_mean_ms']} ms | "
            f"**{c['warm_latency_p95_ms']} ms** | {c['cold_latency_p95_ms']} ms | {c['peak_rss_mb']} MB | {c['memory_delta_mb']:+.1f} MB |"
        )

    lines.extend([
        "\n## 3. Engineering Analysis & Production Observations",
        f"1. **Corpus Volume & Chunk Alignment**: Each technical document is structured into multiple substantive sections (1,200–1,600 chars), producing **3 to 4 chunks per document**. 500 documents yield **{tier_results[-1]['total_chunks']} discrete chunks**, establishing genuine multi-chunk retrieval at scale.",
        "2. **Subprocess-Isolated Memory Overhead**: Each corpus tier is evaluated in a dedicated fresh subprocess. Baseline RSS is recorded *after* document and chunk construction, and vector caching is disabled (`enable_cache=False`). The reported Index Memory Delta isolates pure index overhead (sparse BM25 inverted index + dense FAISS vector matrix) without cross-tier memory reuse or cache retention.",
        "3. **Empirical Throughput Scaling (Warm-Cache vs Cold-Cache)**:",
        f"   - **In-Process Warm-Cache Retrieval**: When query embeddings are cached, in-process hybrid search on the 500-doc ({tier_results[-1]['total_chunks']}-chunk) tier sustains **{warm_qps_min:.1f} to {warm_qps_max:.1f} queries per second**, with p95 latencies ranging from {warm_p95_min:.1f} ms (1 worker) to {warm_p95_max:.1f} ms under high concurrency (20 workers).",
        f"   - **Cold-Cache Neural Inference**: For unseen queries requiring live transformer forward passes on CPU (`all-MiniLM-L6-v2`), sustained throughput reaches **{cold_qps_min:.1f} to {cold_qps_max:.1f} queries per second** with p95 latencies between {cold_p95_min:.1f} ms and {cold_p95_max:.1f} ms.",
        "   - **Scope Qualification**: This benchmark measures in-process retriever throughput; full end-to-end FastAPI HTTP requests will include ASGI framework overhead and network serialization.",
        "4. **Controlled Laboratory Scope**: These results establish rigorous empirical bounds for medium-scale retrieval. Enterprise deployments scaling to millions of documents require distributed sharding (e.g. Qdrant/Milvus cluster) and GPU-accelerated batch reranking."
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved scalability report to {json_path} and {md_path}")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline scalability and concurrency benchmark.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--corpus-tiers", nargs="+", type=int, default=[30, 100, 500])
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--single-tier", type=int, default=None, help="Execute single isolated tier subprocess")
    args = parser.parse_args()

    if args.single_tier is not None:
        tier_metrics = evaluate_single_tier_isolated(args.single_tier)
        print(f"__TIER_JSON__{json.dumps(tier_metrics)}")
        sys.exit(0)

    run_scalability_benchmark(
        reports_dir=args.reports_dir,
        corpus_tiers=args.corpus_tiers,
        concurrency_levels=args.concurrency_levels
    )
