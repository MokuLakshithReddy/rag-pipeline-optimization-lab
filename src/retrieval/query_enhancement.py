"""Query enhancement techniques: normalization, expansion, and multi-query decomposition."""

from __future__ import annotations
import re
from typing import Dict, List, Tuple


class QueryEnhancer:
    """Enhances raw search queries through normalization, expansion, and multi-querying."""

    # Common domain synonym dictionary for expansion
    EXPANSION_SYNONYMS: Dict[str, List[str]] = {
        "rag": ["retrieval-augmented generation", "vector search", "retrieval"],
        "hybrid": ["dense and sparse", "bm25 and vector", "reciprocal rank fusion"],
        "vector": ["dense embedding", "similarity index"],
        "chunking": ["text splitting", "segmentation", "chunk size"],
        "metrics": ["precision", "recall", "mrr", "ndcg", "evaluation"],
        "latency": ["response time", "execution speed", "inference delay"],
        "abstention": ["refusal", "no information", "insufficient evidence"],
    }

    @classmethod
    def normalize_query(cls, query: str) -> str:
        """Clean whitespace, remove trailing punctuation, and normalize casing."""
        q = query.strip()
        # Collapse multiple spaces
        q = re.sub(r'\s+', ' ', q)
        # Normalize smart quotes and dashes
        q = q.replace('“', '"').replace('”', '"').replace('’', "'").replace('—', '-')
        return q

    @classmethod
    def expand_query(cls, query: str) -> Tuple[str, List[str]]:
        """Expand query with relevant domain terms without altering primary intent."""
        normalized = cls.normalize_query(query)
        words = re.findall(r'\b\w+\b', normalized.lower())
        added_terms: List[str] = []

        for word in words:
            if word in cls.EXPANSION_SYNONYMS:
                for term in cls.EXPANSION_SYNONYMS[word]:
                    if term not in normalized.lower() and term not in added_terms:
                        added_terms.append(term)

        if added_terms:
            expanded = f"{normalized} ({' '.join(added_terms)})"
            return expanded, added_terms

        return normalized, []

    @classmethod
    def generate_multi_queries(cls, query: str) -> List[str]:
        """Generate alternative query formulations for comprehensive retrieval coverage."""
        normalized = cls.normalize_query(query)
        queries = [normalized]

        # If query contains conjunctions (and, or, vs), decompose into sub-queries
        conjunction_match = re.split(r'\b(?:and|vs|versus|as well as)\b', normalized, flags=re.IGNORECASE)
        if len(conjunction_match) > 1:
            for part in conjunction_match:
                sub = part.strip()
                if len(sub) > 5 and sub not in queries:
                    queries.append(sub)

        # Keyword-focused query
        tokens = re.findall(r'\b\w+\b', normalized)
        stopwords = {"what", "is", "the", "how", "to", "does", "in", "for", "a", "an", "of", "with"}
        keywords = [t for t in tokens if t.lower() not in stopwords]
        if keywords and len(keywords) >= 2:
            keyword_q = " ".join(keywords)
            if keyword_q != normalized and keyword_q not in queries:
                queries.append(keyword_q)

        return queries

