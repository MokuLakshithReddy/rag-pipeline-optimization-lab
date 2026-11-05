"""Integration tests for Kaggle competition submission validation and schema enforcement."""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.evaluation.kaggle_validator import KaggleSubmissionValidator, KaggleValidationError


@pytest.fixture
def test_files_dir(tmp_path):
    """Fixture providing temporary test directory with dummy submission and test sets."""
    test_set_path = tmp_path / "test_set.csv"
    test_data = pd.DataFrame([
        {"query_id": "q1", "query": "What is MVCC?", "ground_truth_answer": "Multi-version concurrency control."},
        {"query_id": "q2", "query": "What is PBFT?", "ground_truth_answer": "Practical Byzantine Fault Tolerance."},
        {"query_id": "q3", "query": "What is WAL?", "ground_truth_answer": "Write ahead logging."}
    ])
    test_data.to_csv(test_set_path, index=False)

    return {
        "dir": tmp_path,
        "test_set_path": test_set_path,
        "test_df": test_data
    }


def test_valid_kaggle_submission_passes(test_files_dir):
    """Verify that a compliant submission passes schema and alignment validation."""
    sub_path = test_files_dir["dir"] / "submission.csv"
    valid_sub = pd.DataFrame([
        {"query_id": "q1", "prediction": "Multi-version concurrency control enables concurrent reads."},
        {"query_id": "q2", "prediction": "Practical Byzantine Fault Tolerance provides consensus."},
        {"query_id": "q3", "prediction": "Write ahead logging guarantees durability."}
    ])
    valid_sub.to_csv(sub_path, index=False)

    validator = KaggleSubmissionValidator()
    report = validator.validate(sub_path, test_files_dir["test_set_path"])

    assert report["validation_status"] == "PASSED"
    assert report["integrity"]["total_rows"] == 3
    assert report["integrity"]["id_column"] == "query_id"
    assert report["integrity"]["prediction_column"] == "prediction"
    assert "offline_score" in report
    assert report["offline_score"]["answer_token_f1_mean"] > 0.0


def test_kaggle_submission_rejects_missing_column(test_files_dir):
    """Verify validator fails fast when mandatory prediction column is missing."""
    sub_path = test_files_dir["dir"] / "missing_col.csv"
    sub_df = pd.DataFrame([
        {"query_id": "q1", "wrong_column": "some answer"}
    ])
    sub_df.to_csv(sub_path, index=False)

    validator = KaggleSubmissionValidator()
    with pytest.raises(KaggleValidationError, match="Missing required prediction column"):
        validator.validate(sub_path)


def test_kaggle_submission_rejects_nan_predictions(test_files_dir):
    """Verify validator rejects submissions containing NaN/null values in predictions."""
    sub_path = test_files_dir["dir"] / "nan_sub.csv"
    sub_df = pd.DataFrame([
        {"query_id": "q1", "prediction": "Valid answer"},
        {"query_id": "q2", "prediction": None},
        {"query_id": "q3", "prediction": "Valid answer 3"}
    ])
    sub_df.to_csv(sub_path, index=False)

    validator = KaggleSubmissionValidator()
    with pytest.raises(KaggleValidationError, match="null/NaN values"):
        validator.validate(sub_path)


def test_kaggle_submission_rejects_row_count_mismatch(test_files_dir):
    """Verify validator rejects submission when row count or IDs do not match the competition test set."""
    sub_path = test_files_dir["dir"] / "mismatch_sub.csv"
    sub_df = pd.DataFrame([
        {"query_id": "q1", "prediction": "Answer 1"},
        {"query_id": "q2", "prediction": "Answer 2"}
        # missing q3
    ])
    sub_df.to_csv(sub_path, index=False)

    validator = KaggleSubmissionValidator()
    with pytest.raises(KaggleValidationError, match="missing 1 expected query IDs"):
        validator.validate(sub_path, test_files_dir["test_set_path"])


def test_kaggle_submission_rejects_duplicate_ids(test_files_dir):
    """Verify validator rejects submissions containing duplicate query IDs."""
    sub_path = test_files_dir["dir"] / "duplicate_sub.csv"
    sub_df = pd.DataFrame([
        {"query_id": "q1", "prediction": "Answer 1"},
        {"query_id": "q1", "prediction": "Answer 1 duplicate"},
        {"query_id": "q2", "prediction": "Answer 2"}
    ])
    sub_df.to_csv(sub_path, index=False)

    validator = KaggleSubmissionValidator()
    with pytest.raises(KaggleValidationError, match="duplicate query IDs"):
        validator.validate(sub_path)

