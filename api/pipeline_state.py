"""Application pipeline state manager for API and Interactive UI."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunking.base import BaseChunker
from src.chunking.registry import get_chunker
from src.embeddings.base import BaseEmbedder
from src.embeddings.registry import get_embedder
from src.generation.llm_generator import LLMGenerator
from src.ingestion.document_loader import UniversalDocumentLoader
from src.ingestion.models import Document, DocumentChunk, GenerationResponse
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.context_optimizer import ContextOptimizer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_enhancement import QueryEnhancer
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger("rag_pipeline")


class PipelineState:
    """Encapsulates active knowledge base, indices, and RAG components."""

    def __init__(
        self,
        upload_dir: str = "data/uploads",
        index_dir: str = "data/indices",
        default_embedder_model: str = "all-MiniLM-L6-v2",
        default_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.upload_dir = Path(upload_dir)
        self.index_dir = Path(index_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.loader = UniversalDocumentLoader(deduplicate=True)
        self.documents: List[Document] = []
        self.chunks: List[DocumentChunk] = []

        # Default components
        self.chunking_strategy = "recursive"
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.embedding_model_name = default_embedder_model
        self.reranker_model_name = default_reranker_model

        self.embedder: BaseEmbedder = get_embedder(model_name=self.embedding_model_name)
        self.dense_retriever = DenseRetriever(embedder=self.embedder)
        self.sparse_retriever = SparseRetriever()
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            fusion_method="rrf",
            rrf_k=60
        )
        self.reranker = CrossEncoderReranker(model_name=self.reranker_model_name, top_n=3)
        self.generator = LLMGenerator(provider="mock")
        self.context_optimizer = ContextOptimizer(max_context_chars=4000)

    def ingest_bytes(self, content_bytes: bytes, filename: str) -> List[Document]:
        """Ingest a document from raw bytes."""
        new_docs = self.loader.load_from_bytes(content_bytes, filename=filename)
        self.documents.extend(new_docs)
        return new_docs

    def build_index(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: Optional[str] = None
    ) -> int:
        """Chunk all loaded documents and build dense, sparse, and hybrid indices."""
        start_t = time.perf_counter()

        self.chunking_strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if embedding_model and embedding_model != self.embedding_model_name:
            self.embedding_model_name = embedding_model
            self.embedder = get_embedder(model_name=embedding_model)

        chunker = get_chunker(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = chunker.chunk_documents(self.documents)

        # Clear existing indices and re-index
        self.dense_retriever = DenseRetriever(embedder=self.embedder)
        self.sparse_retriever = SparseRetriever()

        if self.chunks:
            self.dense_retriever.index_chunks(self.chunks)
            self.sparse_retriever.index_chunks(self.chunks)

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            fusion_method="rrf"
        )

        logger.info(
            f"Indexed {len(self.chunks)} chunks in {(time.perf_counter() - start_t) * 1000.0:.1f}ms"
        )
        return len(self.chunks)

    def query(
        self,
        raw_query: str,
        retrieval_strategy: str = "hybrid",
        top_k: int = 5,
        enable_reranking: bool = True,
        enable_query_enhancement: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Process query end-to-end through retrieval, reranking, and generation."""
        query_text = raw_query.strip()
        if enable_query_enhancement:
            query_text, _ = QueryEnhancer.expand_query(query_text)

        # 1. Select Retriever
        if retrieval_strategy == "dense":
            results = self.dense_retriever.retrieve(query_text, top_k=top_k, filters=filters)
        elif retrieval_strategy == "sparse":
            results = self.sparse_retriever.retrieve(query_text, top_k=top_k, filters=filters)
        else:
            results = self.hybrid_retriever.retrieve(query_text, top_k=top_k, filters=filters)

        # 2. Rerank if enabled and results exist
        if enable_reranking and results:
            results = self.reranker.rerank(query_text, results)

        # 3. Optimize Context
        context_str, citations, accepted_chunks = self.context_optimizer.optimize_context(results)

        # 4. Generate Grounded Response
        response = self.generator.generate(
            query=raw_query,
            context=context_str,
            citations=citations,
            retrieved_chunks=accepted_chunks,
            retrieval_strategy=retrieval_strategy
        )
        return response


# Global pipeline state instance
pipeline_state = PipelineState()

