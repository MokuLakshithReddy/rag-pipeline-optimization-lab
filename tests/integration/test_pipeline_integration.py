"""End-to-end integration tests for RAG pipeline workflows."""

from api.pipeline_state import PipelineState
from src.evaluation.evaluator import RAGEvaluator
from src.experiments.runner import load_benchmark_data


def test_full_pipeline_ingest_index_query():
    state = PipelineState(auto_restore=False)

    doc1 = b"FastAPI is a modern, fast web framework for building APIs with Python 3.8+."
    doc2 = b"Streamlit turns data scripts into shareable web apps in minutes with pure Python."

    state.ingest_bytes(doc1, filename="fastapi.txt")
    state.ingest_bytes(doc2, filename="streamlit.txt")

    assert len(state.documents) == 2

    # Build Index
    chunk_count = state.build_index(strategy="recursive", chunk_size=256, chunk_overlap=32)
    assert chunk_count >= 2

    # Query with Hybrid
    resp_hybrid = state.query("What is FastAPI?", retrieval_strategy="hybrid", top_k=2)
    assert resp_hybrid.is_abstained is False
    assert any(term in resp_hybrid.answer.lower() for term in ["fastapi", "web framework", "python", "apis"])
    assert resp_hybrid.latency_ms > 0

    # Query with Dense
    resp_dense = state.query("What does Streamlit do?", retrieval_strategy="dense", top_k=2)
    assert resp_dense.is_abstained is False
    assert any(term in resp_dense.answer.lower() for term in ["streamlit", "shareable", "web apps", "scripts"])

    # Query with Sparse
    resp_sparse = state.query("web apps minutes pure Python", retrieval_strategy="sparse", top_k=2)
    assert resp_sparse.is_abstained is False
    assert any(term in resp_sparse.answer.lower() for term in ["streamlit", "shareable", "web apps", "scripts"])


def test_regression_benchmark_evaluation():
    data = load_benchmark_data("data/benchmarks/rag_benchmark.json")
    state = PipelineState(auto_restore=False)

    for item in data["corpus"]:
        state.ingest_bytes(item["text"].encode("utf-8"), filename=item["source"])

    state.build_index(strategy="recursive", chunk_size=512, chunk_overlap=64)

    evaluator = RAGEvaluator(
        retriever=state.hybrid_retriever,
        generator=state.generator,
        reranker=state.reranker,
        top_k=5
    )

    metrics = evaluator.evaluate_dataset(data["benchmark_queries"])
    assert metrics["hit_at_k"] >= 0.85
    assert metrics["mrr"] >= 0.80
    assert metrics["ndcg_at_k"] >= 0.80
    assert metrics["total_latency_mean_ms"] < 2500.0

