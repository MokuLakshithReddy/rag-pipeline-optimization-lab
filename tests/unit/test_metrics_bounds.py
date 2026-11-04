"""Unit tests asserting strict mathematical bounds for all evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    compute_faithfulness,
    compute_hit_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_token_f1,
)


def test_recall_at_k_never_exceeds_one_with_duplicate_chunks():
    """Verify that multiple chunks originating from the same document do NOT inflate recall."""
    # Scenario: The ground truth has 1 relevant document 'doc_A'.
    # Top-5 retrieved results contain 4 chunks all originating from 'doc_A' and 1 from 'doc_B'.
    retrieved_chunks_from_docs = ["doc_A", "doc_A", "doc_A", "doc_A", "doc_B"]
    relevant_docs = {"doc_A"}

    recall = compute_recall_at_k(retrieved_chunks_from_docs, relevant_docs, k=5)
    # MUST be strictly 1.0, NEVER 4.0 or > 1.0
    assert recall == 1.0
    assert 0.0 <= recall <= 1.0


def test_precision_at_k_bounds():
    retrieved = ["doc_A", "doc_A", "doc_B", "doc_C", "doc_D"]
    relevant = {"doc_A", "doc_B"}

    precision = compute_precision_at_k(retrieved, relevant, k=5)
    # Unique relevant hits in top 5 are doc_A and doc_B (2 unique hits / 5) = 0.4
    assert precision == 0.4
    assert 0.0 <= precision <= 1.0


def test_ndcg_at_k_never_exceeds_one_with_duplicate_chunks():
    """Verify that repeated chunks from the same document do not accumulate duplicate DCG gains."""
    retrieved = ["doc_A", "doc_A", "doc_A", "doc_B"]
    relevant = {"doc_A"}

    ndcg = compute_ndcg_at_k(retrieved, relevant, k=4)
    assert ndcg == 1.0
    assert 0.0 <= ndcg <= 1.0


def test_mrr_bounds():
    assert compute_reciprocal_rank(["doc_A", "doc_B"], {"doc_A"}) == 1.0
    assert compute_reciprocal_rank(["doc_B", "doc_A"], {"doc_A"}) == 0.5
    assert compute_reciprocal_rank(["doc_B", "doc_C"], {"doc_A"}) == 0.0
    assert compute_reciprocal_rank([], {"doc_A"}) == 0.0


def test_hit_at_k_bounds():
    assert compute_hit_at_k(["doc_A", "doc_B"], {"doc_A"}, k=1) == 1.0
    assert compute_hit_at_k(["doc_B", "doc_A"], {"doc_A"}, k=1) == 0.0
    assert compute_hit_at_k(["doc_B", "doc_A"], {"doc_A"}, k=2) == 1.0


def test_token_f1_bounds():
    assert compute_token_f1("exact match text", "exact match text") == 1.0
    assert compute_token_f1("completely different", "unrelated words") == 0.0
    f1 = compute_token_f1("partial match of words", "partial match with different words")
    assert 0.0 <= f1 <= 1.0


def test_faithfulness_bounds():
    context = "The Raft consensus algorithm relies on leader election and log replication."
    pred_supported = "Raft uses leader election and log replication."
    pred_unsupported = "Quantum computers break AES-256 via Shor algorithm."

    f_high = compute_faithfulness(pred_supported, context)
    f_low = compute_faithfulness(pred_unsupported, context)

    assert 0.0 <= f_high <= 1.0
    assert 0.0 <= f_low <= 1.0
    assert f_high > f_low

