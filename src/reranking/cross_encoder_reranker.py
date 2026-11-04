"""Cross-Encoder reranking implementation with transparent status tracking and strict mode."""

from __future__ import annotations
import logging
import re
from typing import List, Optional
import numpy as np

from src.ingestion.models import RetrievalResult
from src.reranking.base import BaseReranker

logger = logging.getLogger("reranker")


class CrossEncoderReranker(BaseReranker):
    """Reranks candidate retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3,
        device: str = "cpu",
        lazy_load: bool = True,
        strict_mode: bool = False
    ):
        super().__init__(model_name=model_name, top_n=top_n)
        self.device = device
        self.strict_mode = strict_mode
        self._model = None
        self._load_failed = False
        self._load_error: Optional[str] = None
        if not lazy_load:
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded CrossEncoder model: {self.model_name}")
            except Exception as e:
                self._load_failed = True
                self._load_error = str(e)
                logger.warning(f"Failed to load CrossEncoder ({self.model_name}): {e}")
                if self.strict_mode:
                    raise RuntimeError(
                        f"CrossEncoderReranker strict mode enabled: Failed to load '{self.model_name}': {e}"
                    )

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return []

        self._ensure_model()

        if self._model is not None and not self._load_failed:
            pairs = [[query, r.chunk.text] for r in results]
            try:
                scores = self._model.predict(pairs, show_progress_bar=False)
                sorted_indices = np.argsort(scores)[::-1]
                effective_n = min(self.top_n, len(results))
                reranked: List[RetrievalResult] = []

                for rank, idx in enumerate(sorted_indices[:effective_n], start=1):
                    orig = results[idx]
                    reranked.append(
                        RetrievalResult(
                            chunk=orig.chunk,
                            score=float(scores[idx]),
                            rank=rank,
                            retriever_type="neural_cross_encoder"
                        )
                    )
                return reranked
            except Exception as e:
                if self.strict_mode:
                    raise RuntimeError(f"Cross-encoder inference failed in strict mode: {e}")
                logger.warning(f"Cross-encoder inference failed ({e}), using lexical fallback.")

        return self._lexical_fallback_rerank(query, results)

    def _lexical_fallback_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deterministic lexical overlap fallback for offline or low-resource execution."""
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        scored: List[tuple[float, RetrievalResult]] = []

        for r in results:
            c_tokens = re.findall(r'\b\w+\b', r.chunk.text.lower())
            c_set = set(c_tokens)
            overlap = len(q_tokens & c_set)
            bonus = 0.5 if any(t in c_tokens[:20] for t in q_tokens) else 0.0
            lex_score = overlap + bonus + (0.1 * r.score)
            scored.append((lex_score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        effective_n = min(self.top_n, len(results))

        reranked: List[RetrievalResult] = []
        for rank, (score, orig) in enumerate(scored[:effective_n], start=1):
            reranked.append(
                RetrievalResult(
                    chunk=orig.chunk,
                    score=float(score),
                    rank=rank,
                    retriever_type="fallback_lexical"
                )
            )
        return reranked
