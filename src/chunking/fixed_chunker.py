"""Fixed-size character chunker with configurable overlap."""

from __future__ import annotations
from typing import List
from src.chunking.base import BaseChunker
from src.ingestion.models import Document, DocumentChunk


class FixedSizeChunker(BaseChunker):
    """Chunks documents into fixed character windows with designated overlap."""

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        text = document.text
        if not text:
            return []

        chunks: List[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        total_len = len(text)
        start = 0
        chunk_idx = 0

        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            chunk_text = text[start:end]

            chunk_id = f"{document.id}_c{chunk_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.id,
                text=chunk_text,
                chunk_index=chunk_idx,
                start_char=start,
                end_char=end,
                metadata=document.metadata
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end == total_len:
                break
            start += step

        return chunks

