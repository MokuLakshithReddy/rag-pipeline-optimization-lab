"""Recursive character chunker that preserves natural semantic text boundaries."""

from __future__ import annotations
import re
from typing import List, Optional
from src.chunking.base import BaseChunker
from src.ingestion.models import Document, DocumentChunk


class RecursiveCharacterChunker(BaseChunker):
    """Hierarchical text splitter using paragraphs, sentences, and words before characters."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        text = document.text
        if not text:
            return []

        raw_splits = self._split_text(text, self.separators)
        merged_texts = self._merge_splits(raw_splits)

        chunks: List[DocumentChunk] = []
        current_search_idx = 0

        for chunk_idx, chunk_text in enumerate(merged_texts):
            # Locate character span in source document
            found_idx = text.find(chunk_text, current_search_idx)
            if found_idx == -1:
                found_idx = text.find(chunk_text)
            start_char = found_idx if found_idx != -1 else current_search_idx
            end_char = start_char + len(chunk_text)
            current_search_idx = max(current_search_idx, start_char + 1)

            chunk = DocumentChunk(
                chunk_id=f"{document.id}_rc{chunk_idx}",
                doc_id=document.id,
                text=chunk_text,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                metadata=document.metadata
            )
            chunks.append(chunk)

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using the most meaningful separator that succeeds."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size or not new_separators:
                final_chunks.append(s)
            else:
                final_chunks.extend(self._split_text(s, new_separators))

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Combine smaller segments into chunks obeying chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for piece in splits:
            piece_len = len(piece)
            if total_len + piece_len > self.chunk_size and current_doc:
                merged = " ".join(current_doc).strip()
                if merged:
                    docs.append(merged)
                # Keep overlap items
                while total_len > self.chunk_overlap and len(current_doc) > 1:
                    popped = current_doc.pop(0)
                    total_len -= len(popped) + 1

            current_doc.append(piece)
            total_len += piece_len + 1

        if current_doc:
            merged = " ".join(current_doc).strip()
            if merged:
                docs.append(merged)

        return docs

