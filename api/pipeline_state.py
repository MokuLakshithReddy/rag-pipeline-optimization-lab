"""Thread-safe, lazy-loading application pipeline state manager with persistent storage recovery."""

from __future__ import annotations
import json
import logging
import os
import threading
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
    """Thread-safe state manager with lazy model loading and persistent index recovery."""

    def __init__(
        self,
        upload_dir: str = "data/uploads",
        index_dir: str = "data/indices",
        default_embedder_model: str = "all-MiniLM-L6-v2",
        default_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        auto_restore: bool = True
    ):
        self._lock = threading.RLock()
        self.upload_dir = Path(upload_dir)
        self.index_dir = Path(index_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.loader = UniversalDocumentLoader(deduplicate=True)
        self.documents: List[Document] = []
        self.chunks: List[DocumentChunk] = []

        # Configuration
        self.chunking_strategy = "recursive"
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.embedding_model_name = default_embedder_model
        self.reranker_model_name = default_reranker_model

        # Lazy component references
        self._embedder: Optional[BaseEmbedder] = None
        self._dense_retriever: Optional[DenseRetriever] = None
        self._sparse_retriever: Optional[SparseRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._generator: Optional[LLMGenerator] = None
        self.context_optimizer = ContextOptimizer(max_context_chars=4000)

        if auto_restore:
            self.restore_state()

    @property
    def embedder(self) -> BaseEmbedder:
        with self._lock:
            if self._embedder is None:
                logger.info(f"Lazy-initializing embedder: {self.embedding_model_name}")
                self._embedder = get_embedder(model_name=self.embedding_model_name)
            return self._embedder

    @property
    def dense_retriever(self) -> DenseRetriever:
        with self._lock:
            if self._dense_retriever is None:
                self._dense_retriever = DenseRetriever(embedder=self.embedder)
                if self.chunks:
                    self._dense_retriever.index_chunks(self.chunks)
            return self._dense_retriever

    @property
    def sparse_retriever(self) -> SparseRetriever:
        with self._lock:
            if self._sparse_retriever is None:
                self._sparse_retriever = SparseRetriever()
                if self.chunks:
                    self._sparse_retriever.index_chunks(self.chunks)
            return self._sparse_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        with self._lock:
            if self._hybrid_retriever is None:
                self._hybrid_retriever = HybridRetriever(
                    dense_retriever=self.dense_retriever,
                    sparse_retriever=self.sparse_retriever,
                    fusion_method="rrf",
                    rrf_k=60
                )
            return self._hybrid_retriever

    @property
    def reranker(self) -> CrossEncoderReranker:
        with self._lock:
            if self._reranker is None:
                logger.info(f"Lazy-initializing reranker: {self.reranker_model_name}")
                self._reranker = CrossEncoderReranker(model_name=self.reranker_model_name, top_n=3)
            return self._reranker

    @property
    def generator(self) -> LLMGenerator:
        with self._lock:
            if self._generator is None:
                provider = os.getenv("LLM_PROVIDER")
                logger.info(f"Initializing generator with provider: {provider}")
                self._generator = LLMGenerator(provider=provider)
            return self._generator

    def ingest_bytes(self, content_bytes: bytes, filename: str, persist: bool = True) -> List[Document]:
        """Ingest document from raw bytes with persistence option."""
        with self._lock:
            new_docs = self.loader.load_from_bytes(content_bytes, filename=filename)
            self.documents.extend(new_docs)

            if persist and new_docs:
                save_path = self.upload_dir / filename
                save_path.write_bytes(content_bytes)
                self._save_documents_manifest()

            return new_docs

    def build_index(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: Optional[str] = None
    ) -> int:
        """Chunk all loaded documents, build dense, sparse, and hybrid indices, and persist."""
        with self._lock:
            start_t = time.perf_counter()

            self.chunking_strategy = strategy
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

            if embedding_model and embedding_model != self.embedding_model_name:
                self.embedding_model_name = embedding_model
                self._embedder = None
                self._dense_retriever = None

            chunker = get_chunker(
                strategy=self.chunking_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.chunks = chunker.chunk_documents(self.documents)

            # Re-index
            self._dense_retriever = DenseRetriever(embedder=self.embedder)
            self._sparse_retriever = SparseRetriever()

            if self.chunks:
                self._dense_retriever.index_chunks(self.chunks)
                self._sparse_retriever.index_chunks(self.chunks)

            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self._dense_retriever,
                sparse_retriever=self._sparse_retriever,
                fusion_method="rrf"
            )

            # Persist indices to disk
            self.save_state()

            elapsed = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"Indexed {len(self.chunks)} chunks in {elapsed:.1f}ms and persisted state.")
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
        with self._lock:
            query_text = raw_query.strip()
            if enable_query_enhancement:
                query_text, _ = QueryEnhancer.expand_query(query_text)

            # 1. Retrieval
            if retrieval_strategy == "dense":
                results = self.dense_retriever.retrieve(query_text, top_k=top_k, filters=filters)
            elif retrieval_strategy == "sparse":
                results = self.sparse_retriever.retrieve(query_text, top_k=top_k, filters=filters)
            else:
                results = self.hybrid_retriever.retrieve(query_text, top_k=top_k, filters=filters)

            # 2. Rerank
            if enable_reranking and results:
                results = self.reranker.rerank(query_text, results)

            # 3. Context Optimization
            context_str, citations, accepted_chunks = self.context_optimizer.optimize_context(results)

            # 4. Grounded Generation
            response = self.generator.generate(
                query=raw_query,
                context=context_str,
                citations=citations,
                retrieved_chunks=accepted_chunks,
                retrieval_strategy=retrieval_strategy
            )
            return response

    def _save_documents_manifest(self) -> None:
        """Save ingested documents metadata manifest."""
        manifest_file = self.upload_dir / "documents_manifest.json"
        data = [d.model_dump() for d in self.documents]
        manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_state(self) -> None:
        """Persist indices, chunks, and metadata to disk."""
        with self._lock:
            self._save_documents_manifest()
            if self._dense_retriever and self.chunks:
                self._dense_retriever.save(self.index_dir)
            if self._sparse_retriever and self.chunks:
                self._sparse_retriever.save(self.index_dir)

    def restore_state(self) -> bool:
        """Restore documents and indices from disk after application restart."""
        with self._lock:
            manifest_file = self.upload_dir / "documents_manifest.json"
            if manifest_file.exists():
                try:
                    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
                    self.documents = [Document(**item) for item in payload]
                    logger.info(f"Restored {len(self.documents)} documents from {manifest_file}")
                except Exception as e:
                    logger.warning(f"Failed to restore documents manifest: {e}")

            dense_meta = self.index_dir / "dense_metadata.json"
            if dense_meta.exists() and (self.index_dir / "dense_embeddings.npy").exists():
                try:
                    self._dense_retriever = DenseRetriever(embedder=self.embedder)
                    self._dense_retriever.load(self.index_dir)
                    self.chunks = self._dense_retriever.chunks

                    self._sparse_retriever = SparseRetriever()
                    self._sparse_retriever.load(self.index_dir)

                    self._hybrid_retriever = HybridRetriever(
                        dense_retriever=self._dense_retriever,
                        sparse_retriever=self._sparse_retriever,
                        fusion_method="rrf"
                    )
                    logger.info(f"Restored {len(self.chunks)} indexed chunks from persistent storage.")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to restore indices: {e}")

            return False


# Global pipeline state instance
pipeline_state = PipelineState()
