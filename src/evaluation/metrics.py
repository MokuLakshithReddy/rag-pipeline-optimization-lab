"""Quantitative evaluation metrics for retrieval and grounded generation."""

from __future__ import annotations
import math
import re
from typing import Dict, List, Set


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def compute_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute Reciprocal Rank (RR) for MRR calculation."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    seen_relevant: Set[str] = set()

    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids and doc_id not in seen_relevant:
            dcg += 1.0 / math.log2(rank + 1)
            seen_relevant.add(doc_id)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
    if idcg == 0.0:
        return 0.0

    return min(1.0, dcg / idcg)


def compute_hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Hit@K (1.0 if at least one relevant document is in top_k, else 0.0)."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between predicted answer and ground truth."""
    pred_tokens = re.findall(r'\b\w+\b', prediction.lower())
    gt_tokens = re.findall(r'\b\w+\b', ground_truth.lower())

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    recall = sum(1 for t in gt_tokens if t in common) / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_faithfulness(prediction: str, context: str) -> float:
    """Estimate faithfulness by measuring proportion of prediction content words present in context."""
    pred_tokens = re.findall(r'\b\w+\b', prediction.lower())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "with", "source", "page"}
    meaningful_pred = [t for t in pred_tokens if t not in stopwords]

    if not meaningful_pred:
        return 1.0

    context_lower = context.lower()
    supported = sum(1 for t in meaningful_pred if t in context_lower)
    return supported / len(meaningful_pred)
