"""Thread-safe, multi-tenant application pipeline state manager with persistent storage recovery."""

from __future__ import annotations
import json
import logging
import os
import shutil
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
    """Thread-safe tenant-isolated state manager with lazy model loading and persistent recovery."""

    def __init__(
        self,
        tenant_id: str = "default",
        upload_dir: Optional[str | Path] = None,
        index_dir: Optional[str | Path] = None,
        default_embedder_model: str = "all-MiniLM-L6-v2",
        default_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        auto_restore: bool = True,
        shared_embedder: Optional[BaseEmbedder] = None,
        shared_reranker: Optional[CrossEncoderReranker] = None,
        shared_generator: Optional[LLMGenerator] = None
    ):
        self.tenant_id = tenant_id
        self._lock = threading.RLock()

        if upload_dir is not None:
            self.upload_dir = Path(upload_dir)
        else:
            self.upload_dir = Path(f"data/tenants/{tenant_id}/uploads")

        if index_dir is not None:
            self.index_dir = Path(index_dir)
        else:
            self.index_dir = Path(f"data/tenants/{tenant_id}/indices")

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

        # Shared or lazy model references
        self._embedder: Optional[BaseEmbedder] = shared_embedder
        self._reranker: Optional[CrossEncoderReranker] = shared_reranker
        self._generator: Optional[LLMGenerator] = shared_generator

        # Tenant-specific retriever indices
        self._dense_retriever: Optional[DenseRetriever] = None
        self._sparse_retriever: Optional[SparseRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self.context_optimizer = ContextOptimizer(max_context_chars=4000)

        if auto_restore:
            self.restore_state()

    @property
    def embedder(self) -> BaseEmbedder:
        with self._lock:
            if self._embedder is None:
                logger.info(f"[{self.tenant_id}] Lazy-initializing embedder: {self.embedding_model_name}")
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
                logger.info(f"[{self.tenant_id}] Lazy-initializing reranker: {self.reranker_model_name}")
                self._reranker = CrossEncoderReranker(model_name=self.reranker_model_name, top_n=3)
            return self._reranker

    @property
    def generator(self) -> LLMGenerator:
        with self._lock:
            if self._generator is None:
                provider = os.getenv("LLM_PROVIDER")
                logger.info(f"[{self.tenant_id}] Initializing generator with provider: {provider}")
                self._generator = LLMGenerator(provider=provider)
            return self._generator

    def ingest_bytes(self, content_bytes: bytes, filename: str, persist: bool = True) -> List[Document]:
        """Ingest document from raw bytes into this tenant's isolated repository."""
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
        chunk_overlap: int = 64
    ) -> int:
        """Chunk all tenant documents and build dense, sparse, and hybrid indices."""
        with self._lock:
            if not self.documents:
                return 0

            self.chunking_strategy = strategy
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

            chunker = get_chunker(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedder=self.embedder if strategy == "semantic" else None
            )

            self.chunks = chunker.chunk_documents(self.documents)
            logger.info(f"[{self.tenant_id}] Chunked {len(self.documents)} documents into {len(self.chunks)} chunks.")

            # Build and synchronize indices
            self._dense_retriever = DenseRetriever(embedder=self.embedder)
            self._dense_retriever.index_chunks(self.chunks)

            self._sparse_retriever = SparseRetriever()
            self._sparse_retriever.index_chunks(self.chunks)

            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self._dense_retriever,
                sparse_retriever=self._sparse_retriever,
                fusion_method="rrf",
                rrf_k=60
            )

            self.save_state()
            return len(self.chunks)

    def query(
        self,
        question: str,
        retrieval_strategy: str = "hybrid",
        top_k: int = 5,
        use_reranker: bool = False,
        use_query_expansion: bool = False
    ) -> GenerationResponse:
        """Execute end-to-end RAG query against tenant's isolated index."""
        with self._lock:
            if not self.chunks:
                return GenerationResponse(
                    answer="No documents indexed in your tenant workspace. Please upload and index documents first.",
                    citations=[],
                    retrieved_chunks=[],
                    confidence_score=0.0,
                    latency_ms=0.0,
                    is_abstained=True
                )

            start_t = time.perf_counter()
            processed_query = question

            if use_query_expansion:
                processed_query, _ = QueryEnhancer.expand_query(question)

            # Route retrieval
            if retrieval_strategy == "dense":
                retriever = self.dense_retriever
            elif retrieval_strategy == "sparse":
                retriever = self.sparse_retriever
            else:
                retriever = self.hybrid_retriever

            results = retriever.retrieve(processed_query, top_k=top_k)

            # Optional reranking
            if use_reranker and results:
                results = self.reranker.rerank(question, results)

            # Context optimization
            context_str, citations, accepted_results = self.context_optimizer.optimize_context(results)

            # Grounded generation
            response = self.generator.generate(
                query=question,
                context=context_str,
                citations=citations,
                retrieved_chunks=accepted_results
            )

            total_ms = (time.perf_counter() - start_t) * 1000.0
            response.latency_ms = total_ms
            return response

    def _save_documents_manifest(self) -> None:
        """Persist document metadata manifest to tenant directory."""
        manifest_file = self.upload_dir / "documents_manifest.json"
        data = [
            {
                "id": d.id,
                "text": d.text,
                "metadata": {
                    "source": d.metadata.source,
                    "doc_id": d.metadata.doc_id,
                    "file_type": d.metadata.file_type,
                    "page_number": d.metadata.page_number,
                    "character_count": d.metadata.character_count,
                    "token_estimate": d.metadata.token_estimate,
                    "hash": d.metadata.hash,
                    "extra": d.metadata.extra,
                }
            }
            for d in self.documents
        ]
        manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_state(self) -> None:
        """Persist indices, chunks, and metadata to tenant disk."""
        with self._lock:
            self._save_documents_manifest()
            if self._dense_retriever and self.chunks:
                self._dense_retriever.save(self.index_dir)
            if self._sparse_retriever and self.chunks:
                self._sparse_retriever.save(self.index_dir)

    def restore_state(self) -> bool:
        """Restore tenant documents and indices from persistent disk storage."""
        with self._lock:
            manifest_file = self.upload_dir / "documents_manifest.json"
            if manifest_file.exists():
                try:
                    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
                    self.documents = [Document(**item) for item in payload]
                    logger.info(f"[{self.tenant_id}] Restored {len(self.documents)} documents from manifest.")
                except Exception as e:
                    logger.warning(f"[{self.tenant_id}] Failed to restore documents manifest: {e}")

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
                    logger.info(f"[{self.tenant_id}] Restored {len(self.chunks)} indexed chunks from storage.")
                    return True
                except Exception as e:
                    logger.warning(f"[{self.tenant_id}] Failed to restore indices: {e}")

            return False

    def reset(self) -> None:
        """Clear all tenant documents, chunks, and persisted indices."""
        with self._lock:
            self.documents.clear()
            self.chunks.clear()
            self._dense_retriever = None
            self._sparse_retriever = None
            self._hybrid_retriever = None
            if self.upload_dir.exists():
                shutil.rmtree(self.upload_dir, ignore_errors=True)
                self.upload_dir.mkdir(parents=True, exist_ok=True)
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir, ignore_errors=True)
                self.index_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{self.tenant_id}] Workspace reset cleanly.")


class MultiTenantStateManager:
    """Manages multi-tenant pipeline states with shared transformer model weights."""

    def __init__(
        self,
        base_tenants_dir: str | Path = "data/tenants",
        default_embedder_model: str = "all-MiniLM-L6-v2",
        default_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self._lock = threading.RLock()
        self.base_dir = Path(base_tenants_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_embedder_model = default_embedder_model
        self.default_reranker_model = default_reranker_model

        # Shared models across tenants to conserve RAM and avoid duplicate GPU allocations
        self._shared_embedder: Optional[BaseEmbedder] = None
        self._shared_reranker: Optional[CrossEncoderReranker] = None
        self._shared_generator: Optional[LLMGenerator] = None

        self._tenants: Dict[str, PipelineState] = {}

    @property
    def shared_embedder(self) -> BaseEmbedder:
        with self._lock:
            if self._shared_embedder is None:
                self._shared_embedder = get_embedder(model_name=self.default_embedder_model)
            return self._shared_embedder

    @property
    def shared_reranker(self) -> CrossEncoderReranker:
        with self._lock:
            if self._shared_reranker is None:
                self._shared_reranker = CrossEncoderReranker(model_name=self.default_reranker_model, top_n=3)
            return self._shared_reranker

    @property
    def shared_generator(self) -> LLMGenerator:
        with self._lock:
            if self._shared_generator is None:
                self._shared_generator = LLMGenerator(provider=os.getenv("LLM_PROVIDER"))
            return self._shared_generator

    def get_state(self, tenant_id: str = "default", auto_restore: bool = True) -> PipelineState:
        """Get or initialize isolated PipelineState for a specific tenant."""
        safe_tenant = "".join(c for c in tenant_id if c.isalnum() or c in {"-", "_"}).strip() or "default"
        with self._lock:
            if safe_tenant not in self._tenants:
                state = PipelineState(
                    tenant_id=safe_tenant,
                    upload_dir=self.base_dir / safe_tenant / "uploads",
                    index_dir=self.base_dir / safe_tenant / "indices",
                    default_embedder_model=self.default_embedder_model,
                    default_reranker_model=self.default_reranker_model,
                    auto_restore=auto_restore,
                    shared_embedder=self.shared_embedder,
                    shared_reranker=self.shared_reranker,
                    shared_generator=self.shared_generator
                )
                self._tenants[safe_tenant] = state
            return self._tenants[safe_tenant]

    def restore_all_tenants(self) -> int:
        """Scan tenant directories and restore persistent states."""
        with self._lock:
            restored_count = 0
            if self.base_dir.exists():
                for tenant_dir in self.base_dir.iterdir():
                    if tenant_dir.is_dir():
                        tenant_id = tenant_dir.name
                        state = self.get_state(tenant_id, auto_restore=True)
                        if state.documents or state.chunks:
                            restored_count += 1
            return restored_count


# Global multi-tenant manager instance and default tenant state for backwards compatibility
tenant_manager = MultiTenantStateManager()
pipeline_state = tenant_manager.get_state("default")
