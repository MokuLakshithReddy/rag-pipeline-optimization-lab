"""Kaggle Competition Submission Validator and Schema Verifier.

Validates competition submission files against the official Gen AI Project 4 specifications:
1. Column schema validation (query_id/id, prediction/answer, citations/retrieved_context).
2. Row-count and ID alignment with the competition test set.
3. Content integrity (zero NaN/null values, no empty strings, no duplicate IDs).
4. UTF-8 encoding and file structure conformity.
5. Offline benchmark scoring against ground-truth evaluation sets when available.
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_hit_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_token_f1,
)

logger = logging.getLogger("kaggle_validator")


class KaggleValidationError(Exception):
    """Raised when a competition submission violates schema or integrity rules."""
    pass


class KaggleSubmissionValidator:
    """Rigorous validator for Kaggle competition submissions."""

    # Recognized column mappings for Kaggle test and submission sets
    ID_COLUMNS = {"query_id", "id", "qid", "question_id"}
    PRED_COLUMNS = {"prediction", "answer", "predicted_answer", "response", "generated_answer", "answer_text"}
    CONTEXT_COLUMNS = {"retrieved_context", "context", "retrieved_chunks", "citations", "documents", "cited_docs", "cited_sections"}

    # Official NxtWave Competition schema
    OFFICIAL_COMPETITION_COLUMNS = ["question_id", "answer_text", "cited_docs", "cited_sections"]
    OFFICIAL_ROW_COUNT = 50

    def __init__(
        self,
        allow_empty_predictions: bool = False,
        min_prediction_chars: int = 3,
        max_prediction_chars: int = 10000,
        strict_competition: bool = False
    ):
        self.allow_empty_predictions = allow_empty_predictions
        self.min_prediction_chars = min_prediction_chars
        self.max_prediction_chars = max_prediction_chars
        self.strict_competition = strict_competition

    def load_submission(self, submission_path: str | Path) -> pd.DataFrame:
        """Load submission from CSV, TSV, or Parquet with strict format checks."""
        path = Path(submission_path)
        if not path.exists():
            raise KaggleValidationError(f"Submission file does not exist: {path}")

        if path.stat().st_size == 0:
            raise KaggleValidationError(f"Submission file is empty (0 bytes): {path}")

        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix.lower() in {".tsv", ".tab"}:
                df = pd.read_csv(path, sep="\t", encoding="utf-8")
            else:
                df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError as e:
            raise KaggleValidationError(f"Submission file must be UTF-8 encoded: {e}")
        except Exception as e:
            raise KaggleValidationError(f"Failed to parse submission file: {e}")

        if df.empty:
            raise KaggleValidationError("Submission file contains 0 data rows.")

        return df

    def identify_columns(self, df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
        """Identify query ID column, prediction column, and optional context column."""
        cols = {c.strip().lower(): c for c in df.columns}

        # 1. Locate ID column
        id_col = None
        for candidate in self.ID_COLUMNS:
            if candidate in cols:
                id_col = cols[candidate]
                break

        if not id_col:
            raise KaggleValidationError(
                f"Missing required query ID column in submission. Expected one of {list(self.ID_COLUMNS)}, "
                f"found: {list(df.columns)}"
            )

        # 2. Locate Prediction column
        pred_col = None
        for candidate in self.PRED_COLUMNS:
            if candidate in cols:
                pred_col = cols[candidate]
                break

        if not pred_col:
            raise KaggleValidationError(
                f"Missing required prediction column in submission. Expected one of {list(self.PRED_COLUMNS)}, "
                f"found: {list(df.columns)}"
            )

        # 3. Locate optional Context column
        ctx_col = None
        for candidate in self.CONTEXT_COLUMNS:
            if candidate in cols:
                ctx_col = cols[candidate]
                break

        return id_col, pred_col, ctx_col

    def validate_integrity(
        self,
        df: pd.DataFrame,
        id_col: str,
        pred_col: str,
        expected_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive data integrity and sanity checks."""
        errors: List[str] = []
        warnings: List[str] = []

        total_rows = len(df)

        # 1. Duplicate IDs
        ids_series = df[id_col].astype(str)
        dup_count = ids_series.duplicated().sum()
        if dup_count > 0:
            duplicates = ids_series[ids_series.duplicated()].unique().tolist()[:5]
            errors.append(f"Found {dup_count} duplicate query IDs in submission (examples: {duplicates})")

        # 2. Null or NaN in predictions
        null_count = df[pred_col].isna().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} null/NaN values in prediction column '{pred_col}'")

        # 3. Empty or whitespace-only predictions
        empty_mask = df[pred_col].astype(str).str.strip() == ""
        empty_count = empty_mask.sum()
        if empty_count > 0 and not self.allow_empty_predictions:
            errors.append(f"Found {empty_count} empty or whitespace-only answers in '{pred_col}'")

        # 4. Character length boundaries
        lengths = df[pred_col].astype(str).str.len()
        short_count = (lengths < self.min_prediction_chars).sum()
        long_count = (lengths > self.max_prediction_chars).sum()

        if short_count > 0:
            warnings.append(
                f"{short_count} predictions are shorter than {self.min_prediction_chars} characters."
            )
        if long_count > 0:
            warnings.append(
                f"{long_count} predictions exceed {self.max_prediction_chars} characters."
            )

        # 5. Row-count and ID alignment with competition test set
        if expected_ids is not None:
            submitted_ids = set(ids_series)
            missing_ids = expected_ids - submitted_ids
            extra_ids = submitted_ids - expected_ids

            if missing_ids:
                examples = list(missing_ids)[:5]
                errors.append(
                    f"Submission is missing {len(missing_ids)} expected query IDs from the test set (examples: {examples})"
                )

            if extra_ids:
                examples = list(extra_ids)[:5]
                errors.append(
                    f"Submission contains {len(extra_ids)} unexpected query IDs not in test set (examples: {examples})"
                )

            if len(submitted_ids) != len(expected_ids):
                errors.append(
                    f"Row count mismatch: submission contains {len(submitted_ids)} unique queries, "
                    f"expected {len(expected_ids)}"
                )

        # 6. Official Competition Strict Verification Mode
        if self.strict_competition:
            # Check exact column names
            actual_cols = [c.strip().lower() for c in df.columns]
            for req_col in self.OFFICIAL_COMPETITION_COLUMNS:
                if req_col not in actual_cols:
                    errors.append(
                        f"Competition strict schema violation: missing mandatory column '{req_col}'. "
                        f"Required columns: {self.OFFICIAL_COMPETITION_COLUMNS}, found: {list(df.columns)}"
                    )

            # Check exact row count (must be exactly 50)
            if total_rows != self.OFFICIAL_ROW_COUNT and expected_ids is None:
                errors.append(
                    f"Competition strict rule violation: submission must contain exactly {self.OFFICIAL_ROW_COUNT} rows, "
                    f"found {total_rows} rows."
                )

            # Check citation columns integrity
            for col_name in ["cited_docs", "cited_sections"]:
                matched_col = None
                for c in df.columns:
                    if c.strip().lower() == col_name:
                        matched_col = c
                        break
                if matched_col:
                    null_cites = df[matched_col].isna().sum()
                    if null_cites > 0:
                        errors.append(
                            f"Competition citation violation: found {null_cites} null values in '{matched_col}'"
                        )
                    empty_cites = (df[matched_col].astype(str).str.strip() == "").sum()
                    if empty_cites > 0:
                        errors.append(
                            f"Competition citation violation: found {empty_cites} empty values in '{matched_col}'"
                        )

        if errors:
            raise KaggleValidationError("; ".join(errors))

        return {
            "status": "PASSED",
            "total_rows": total_rows,
            "unique_queries": len(ids_series.unique()),
            "id_column": id_col,
            "prediction_column": pred_col,
            "mean_prediction_length": float(lengths.mean()) if total_rows > 0 else 0.0,
            "min_prediction_length": int(lengths.min()) if total_rows > 0 else 0,
            "max_prediction_length": int(lengths.max()) if total_rows > 0 else 0,
            "warnings": warnings
        }

    def evaluate_ground_truth(
        self,
        submission_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sub_id_col: str,
        sub_pred_col: str,
        test_id_col: str,
        test_gt_col: str
    ) -> Dict[str, float]:
        """Compute offline validation metrics against ground truth labels."""
        merged = pd.merge(
            submission_df[[sub_id_col, sub_pred_col]],
            test_df[[test_id_col, test_gt_col]],
            left_on=sub_id_col,
            right_on=test_id_col,
            how="inner"
        )

        if merged.empty:
            raise KaggleValidationError("Cannot evaluate: zero overlapping query IDs between submission and test set.")

        f1_scores: List[float] = []
        for _, row in merged.iterrows():
            pred = str(row[sub_pred_col])
            gt = str(row[test_gt_col])
            f1_scores.append(compute_token_f1(pred, gt))

        return {
            "evaluated_queries": float(len(merged)),
            "answer_token_f1_mean": float(np.mean(f1_scores)),
            "answer_token_f1_p50": float(np.percentile(f1_scores, 50)),
            "answer_token_f1_p95": float(np.percentile(f1_scores, 95))
        }

    def validate(
        self,
        submission_path: str | Path,
        test_set_path: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """Run full end-to-end validation pipeline on a submission file."""
        df = self.load_submission(submission_path)
        id_col, pred_col, ctx_col = self.identify_columns(df)

        expected_ids: Optional[Set[str]] = None
        eval_metrics: Optional[Dict[str, float]] = None

        if test_set_path:
            test_path = Path(test_set_path)
            if test_path.suffix.lower() == ".parquet":
                test_df = pd.read_parquet(test_path)
            elif test_path.suffix.lower() == ".json":
                with open(test_path, "r", encoding="utf-8") as f:
                    jdata = json.load(f)
                if isinstance(jdata, dict) and "benchmark_queries" in jdata:
                    test_df = pd.DataFrame(jdata["benchmark_queries"])
                elif isinstance(jdata, list):
                    test_df = pd.DataFrame(jdata)
                else:
                    test_df = pd.DataFrame([jdata])
            else:
                test_df = pd.read_csv(test_path, encoding="utf-8")

            t_id_col = None
            for c in self.ID_COLUMNS:
                for col in test_df.columns:
                    if col.lower() == c:
                        t_id_col = col
                        break
                if t_id_col:
                    break

            if not t_id_col:
                raise KaggleValidationError(
                    f"Could not identify a valid query ID column in the provided test set '{test_set_path}'. "
                    f"Expected one of {list(self.ID_COLUMNS)}, found: {list(test_df.columns)}"
                )

            expected_ids = set(test_df[t_id_col].astype(str))

            # Check if ground truth answers are present for evaluation
            gt_col = None
            for candidate in ["ground_truth_answer", "ground_truth", "answer", "target"]:
                for col in test_df.columns:
                    if col.lower() == candidate:
                        gt_col = col
                        break
                if gt_col:
                    break

            if gt_col:
                eval_metrics = self.evaluate_ground_truth(
                    submission_df=df,
                    test_df=test_df,
                    sub_id_col=id_col,
                    sub_pred_col=pred_col,
                    test_id_col=t_id_col,
                    test_gt_col=gt_col
                )

        integrity_result = self.validate_integrity(
            df=df,
            id_col=id_col,
            pred_col=pred_col,
            expected_ids=expected_ids
        )

        report = {
            "submission_path": str(submission_path),
            "test_set_path": str(test_set_path) if test_set_path else None,
            "validation_status": "PASSED",
            "integrity": integrity_result,
            "offline_score": eval_metrics
        }
        return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Kaggle competition submission file.")
    parser.add_argument("--submission", required=True, help="Path to submission file (.csv, .tsv, .parquet)")
    parser.add_argument("--test-set", default=None, help="Path to competition test set or ground truth")
    parser.add_argument("--output-json", default=None, help="Optional path to output validation JSON")
    parser.add_argument(
        "--strict-competition",
        action="store_true",
        help="Enforce official Gen AI Project 4 Kaggle competition rules (exact 50 rows and schema)"
    )
    args = parser.parse_args()

    validator = KaggleSubmissionValidator(strict_competition=args.strict_competition)
    try:
        report = validator.validate(args.submission, args.test_set)
        print("\n=== KAGGLE SUBMISSION VALIDATION PASSED ===")
        print(f"Total Rows: {report['integrity']['total_rows']}")
        print(f"ID Column: {report['integrity']['id_column']}")
        print(f"Prediction Column: {report['integrity']['prediction_column']}")
        if report["integrity"]["warnings"]:
            print("Warnings:", report["integrity"]["warnings"])
        if report.get("offline_score"):
            print(f"Offline Mean Token F1: {report['offline_score']['answer_token_f1_mean']:.4f}")

        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Validation report saved to {args.output_json}")

    except KaggleValidationError as e:
        print(f"\n=== KAGGLE SUBMISSION VALIDATION FAILED ===")
        print(f"Error: {e}")
        exit(1)

