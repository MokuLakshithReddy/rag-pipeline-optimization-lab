"""Quantitative evaluation metrics for retrieval and grounded generation.

All metrics are mathematically bounded in the range [0.0, 1.0].
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Set


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Precision@K evaluating unique relevant document hits in top-K.
    
    Precision@K = (Unique relevant documents retrieved in top-K) / K
    Mathematically bounded in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    # Deduplicate to count unique relevant documents retrieved
    unique_relevant_hits = set(doc_id for doc_id in top_k if doc_id in relevant_ids and doc_id != "NON_RELEVANT")
    precision = len(unique_relevant_hits) / k
    return float(min(1.0, max(0.0, precision)))


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Recall@K evaluating unique relevant document hits against total ground truth.
    
    Recall@K = (Unique relevant documents retrieved in top-K) / |Relevant Documents|
    Guaranteed strictly bounded in [0.0, 1.0].
    """
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    # Deduplicate to prevent duplicate chunks from the same document inflating recall beyond 1.0
    unique_relevant_hits = set(doc_id for doc_id in top_k if doc_id in relevant_ids and doc_id != "NON_RELEVANT")
    recall = len(unique_relevant_hits) / len(relevant_ids)
    return float(min(1.0, max(0.0, recall)))


def compute_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute Reciprocal Rank (RR) for MRR calculation.
    
    RR = 1 / rank_first_relevant_hit, bounded in [0.0, 1.0].
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids and doc_id != "NON_RELEVANT":
            return float(min(1.0, max(0.0, 1.0 / rank)))
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    
    Deduplicates relevant document appearances so repeated chunks do not inflate DCG.
    Mathematically bounded in [0.0, 1.0].
    """
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    seen_relevant: Set[str] = set()

    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids and doc_id != "NON_RELEVANT" and doc_id not in seen_relevant:
            dcg += 1.0 / math.log2(rank + 1)
            seen_relevant.add(doc_id)

    # Ideal DCG for the number of unique relevant documents available (up to K)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0

    return float(min(1.0, max(0.0, dcg / idcg)))


def compute_hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Hit@K (1.0 if at least one unique relevant document is in top_k, else 0.0)."""
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    has_hit = any(doc_id in relevant_ids and doc_id != "NON_RELEVANT" for doc_id in top_k)
    return 1.0 if has_hit else 0.0


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
    if precision + recall == 0.0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return float(min(1.0, max(0.0, f1)))


def compute_faithfulness(prediction: str, context: str) -> float:
    """Estimate faithfulness by measuring proportion of prediction propositions and content tokens in context.
    
    Guaranteed bounded in [0.0, 1.0].
    """
    if not context.strip():
        return 0.0

    pred_tokens = re.findall(r'\b\w+\b', prediction.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
        "for", "of", "with", "source", "page", "chunk", "and", "that", "this"
    }
    meaningful_pred = [t for t in pred_tokens if t not in stopwords]

    if not meaningful_pred:
        return 1.0

    context_lower = context.lower()
    supported = sum(1 for t in meaningful_pred if t in context_lower)
    score = supported / len(meaningful_pred)
    return float(min(1.0, max(0.0, score)))
