"""Interactive Streamlit Demonstration Application for RAG Pipeline Optimization."""

import os
import time
import pandas as pd
import streamlit as st

from api.pipeline_state import pipeline_state
from src.chunking.registry import get_chunker
from src.evaluation.evaluator import RAGEvaluator
from src.experiments.tracker import ExperimentTracker
from src.retrieval.context_optimizer import ContextOptimizer

st.set_page_config(
    page_title="RAG Pipeline Optimization & Experimentation Lab",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 RAG Pipeline Optimization Lab")
st.caption("Production-grade Retrieval-Augmented Generation benchmark and optimization engineering platform.")

# Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Configuration")

retrieval_strategy = st.sidebar.selectbox(
    "Retrieval Strategy",
    options=["hybrid", "dense", "sparse"],
    index=0,
    help="Dense (Semantic Vector), Sparse (BM25 Keyword), or Hybrid (Reciprocal Rank Fusion)."
)

enable_reranking = st.sidebar.checkbox(
    "Enable Cross-Encoder Reranker",
    value=True,
    help="Re-score retrieved candidate pool with a Cross-Encoder."
)

enable_query_enhancement = st.sidebar.checkbox(
    "Enable Query Enhancement",
    value=False,
    help="Expand queries with domain synonyms and normalize casing/formatting."
)

top_k = st.sidebar.slider("Top-K Retrieved Chunks", min_value=1, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Chunking Settings")
chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    options=["recursive", "fixed", "semantic"],
    index=0
)
chunk_size = st.sidebar.number_input("Chunk Size (characters)", min_value=64, max_value=2048, value=512, step=64)
chunk_overlap = st.sidebar.number_input("Chunk Overlap (characters)", min_value=0, max_value=512, value=64, step=16)

# Main Navigation Tabs
tab_query, tab_docs, tab_experiments, tab_analysis = st.tabs([
    "💬 Query & Inspection",
    "📁 Document Management",
    "📊 Experiment Benchmarks",
    "📑 Systems Analysis"
])

# TAB 1: Query & Inspection
with tab_query:
    st.subheader("Query Pipeline")

    if not pipeline_state.chunks:
        st.info("ℹ️ The index is currently empty. Head over to the **Document Management** tab to upload documents or load the benchmark corpus!")

    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g. How does Raft leader election handle an election timeout?"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        submit_btn = st.button("🚀 Retrieve & Generate", type="primary")

    if submit_btn and user_query:
        if not pipeline_state.chunks:
            st.error("Cannot query: No documents have been indexed yet. Please ingest and index documents first.")
        else:
            with st.spinner("Processing retrieval and generating grounded response..."):
                start_t = time.perf_counter()
                response = pipeline_state.query(
                    raw_query=user_query,
                    retrieval_strategy=retrieval_strategy,
                    top_k=top_k,
                    enable_reranking=enable_reranking,
                    enable_query_enhancement=enable_query_enhancement
                )
                latency = (time.perf_counter() - start_t) * 1000.0

            # Answer Section
            st.markdown("### Grounded Response")
            if response.is_abstained:
                st.warning(f"⚠️ **Abstention Triggered**: {response.answer}")
            else:
                st.success(response.answer)

            # Metadata Metrics Row
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("End-to-End Latency", f"{response.latency_ms:.1f} ms")
            col_m2.metric("Retrieval Strategy", response.retrieval_strategy)
            col_m3.metric("Reranker Active", "Yes" if enable_reranking else "No")
            col_m4.metric("Retrieved Chunks", len(response.retrieved_chunks))

            # Citations Section
            st.markdown("### 📚 Source Citations")
            if response.citations:
                for idx, cit in enumerate(response.citations, start=1):
                    page_str = f" (Page {cit.page_number})" if cit.page_number else ""
                    st.markdown(f"**[{idx}] {cit.source}{page_str}** — Chunk `{cit.chunk_id}`")
                    st.caption(f"*\"{cit.snippet}\"*")
            else:
                st.write("No source citations returned.")

            # Inspection Accordion
            st.markdown("### 🔍 Retrieved Chunks Deep Inspection")
            with st.expander("Inspect Raw Retrieved Chunks & Scores", expanded=False):
                for idx, item in enumerate(response.retrieved_chunks, start=1):
                    st.markdown(
                        f"**Rank {item.rank}** | Type: `{item.retriever_type}` | Score: `{item.score:.4f}` | "
                        f"Source: `{item.chunk.metadata.source}` (Chunk `{item.chunk.chunk_id}`)"
                    )
                    st.code(item.chunk.text, language="markdown")
                    st.markdown("---")

# TAB 2: Document Management
with tab_docs:
    st.subheader("Knowledge Base Ingestion & Indexing")

    col_upload, col_bench = st.columns(2)
    with col_upload:
        st.markdown("#### Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files (.pdf, .txt, .md, .json, .csv)",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "json", "csv"]
        )
        if st.button("📥 Process Uploaded Files"):
            if uploaded_files:
                ingested = 0
                for f in uploaded_files:
                    content = f.read()
                    docs = pipeline_state.ingest_bytes(content, filename=f.name)
                    ingested += len(docs)
                st.success(f"Ingested {ingested} document section(s) from {len(uploaded_files)} file(s).")
            else:
                st.warning("Please select files to upload.")

    with col_bench:
        st.markdown("#### Load Standardized Benchmark Corpus")
        st.write("Quickly populate the knowledge base with the verified 7-document technical benchmark corpus.")
        if st.button("⚡ Load Benchmark Corpus"):
            import json
            bench_file = "data/benchmarks/rag_benchmark.json"
            if os.path.exists(bench_file):
                with open(bench_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data["corpus"]:
                    pipeline_state.ingest_bytes(
                        item["text"].encode("utf-8"),
                        filename=item["source"]
                    )
                st.success(f"Loaded {len(data['corpus'])} benchmark documents into knowledge base.")
            else:
                st.error("Benchmark file data/benchmarks/rag_benchmark.json not found.")

    st.markdown("---")
    st.subheader("🔨 Index Construction")
    st.write(
        f"Active configuration: **{chunking_strategy}** chunking (Size: {chunk_size}, Overlap: {chunk_overlap}), "
        f"Embedder: `{pipeline_state.embedding_model_name}`."
    )

    if st.button("⚡ Rebuild Knowledge Base Index", type="primary"):
        with st.spinner("Chunking documents and building dense + sparse indices..."):
            total_chunks = pipeline_state.build_index(
                strategy=chunking_strategy,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )
            st.success(f"Successfully indexed {total_chunks} chunk(s) across all retrievers!")

    # Current Document Table
    st.markdown("#### Currently Ingested Documents")
    if pipeline_state.documents:
        doc_data = [
            {
                "Doc ID": d.id,
                "Source": d.metadata.source,
                "Format": d.metadata.file_type,
                "Page": d.metadata.page_number or "-",
                "Characters": d.metadata.character_count,
            }
            for d in pipeline_state.documents
        ]
        st.dataframe(pd.DataFrame(doc_data), use_container_width=True)
    else:
        st.write("No documents ingested yet.")

# TAB 3: Experiment Benchmarks
with tab_experiments:
    st.subheader("📊 Empirical Benchmark Results")
    st.write("Results from controlled single-variable experiments on verified ground-truth evaluation datasets.")

    csv_path = "reports/benchmark_results.csv"
    if os.path.exists(csv_path):
        df_bench = pd.read_csv(csv_path)
        st.dataframe(df_bench, use_container_width=True)

        # Plot charts if columns exist
        if "retrieval_strategy" in df_bench.columns and "mrr" in df_bench.columns:
            st.markdown("#### Performance Metric Comparison")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.bar_chart(df_bench.set_index("experiment_id")[["mrr", "ndcg_at_k"]])
            with col_c2:
                st.bar_chart(df_bench.set_index("experiment_id")[["latency_p95_ms"]])
    else:
        st.info("No benchmark results yet. Run `python src/experiments/runner.py` to execute experiments.")

# TAB 4: Systems Analysis
with tab_analysis:
    st.subheader("Technical Retrieval Systems Analysis")
    analysis_file = "reports/retrieval_analysis.md"
    if os.path.exists(analysis_file):
        with open(analysis_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python src/experiments/runner.py` to generate the systems analysis report.")

