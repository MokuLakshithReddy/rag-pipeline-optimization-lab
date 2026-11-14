#!/usr/bin/env python3
"""Validate a ProctorIQ Kaggle submission against the original competition ZIP.

Checks the submission contract, question coverage, and exact source citations.
It does not estimate Kaggle's hidden semantic score.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

COLUMNS = ["question_id", "answer_text", "cited_docs", "cited_sections"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()

    if not args.archive.is_file() or not args.submission.is_file():
        print("ERROR: archive or submission file does not exist", file=sys.stderr)
        return 2

    with zipfile.ZipFile(args.archive) as archive:
        test_members = [n for n in archive.namelist() if n.endswith("Data/Test CSV/test.csv")]
        if len(test_members) != 1:
            raise ValueError(f"Expected one test.csv in ZIP; found {test_members}")
        test_text = archive.read(test_members[0]).decode("utf-8-sig")
        expected = [r["question_id"] for r in csv.DictReader(io.StringIO(test_text))]
        documents = {
            Path(n).stem: archive.read(n).decode("utf-8")
            for n in archive.namelist()
            if n.startswith("Data/.md Files/") and n.endswith(".md")
        }

    with args.submission.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)

    errors: list[str] = []
    if columns != COLUMNS:
        errors.append(f"Expected columns {COLUMNS}, got {columns}")

    ids = [r.get("question_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate question_id values found")
    if ids != expected:
        errors.append(
            f"Question IDs/order differ; missing={sorted(set(expected)-set(ids))}, "
            f"extra={sorted(set(ids)-set(expected))}"
        )

    pair_count = 0
    word_counts = []
    for row in rows:
        qid = row.get("question_id", "<unknown>")
        answer = (row.get("answer_text") or "").strip()
        docs = [v.strip() for v in (row.get("cited_docs") or "").split("|") if v.strip()]
        sections = [v.strip() for v in (row.get("cited_sections") or "").split("|") if v.strip()]
        if not answer:
            errors.append(f"{qid}: empty answer_text")
        else:
            word_counts.append(len(answer.split()))
        if not docs or not sections:
            errors.append(f"{qid}: citations must not be empty")
        if len(docs) != len(sections):
            errors.append(f"{qid}: document/section citation counts do not align")
            continue
        for doc_id, section in zip(docs, sections):
            pair_count += 1
            if doc_id not in documents:
                errors.append(f"{qid}: unknown source document {doc_id!r}")
                continue
            headings = re.findall(r"^##\s+(.+?)\s*$", documents[doc_id], flags=re.MULTILINE)
            if section not in headings:
                errors.append(f"{qid}: {section!r} is not an exact H2 heading in {doc_id!r}")

    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for error in errors:
            print(f" - {error}")
        return 1

    print("PASSED")
    print(f"Questions: {len(rows)}")
    print(f"Exact source citation pairs: {pair_count}")
    print(f"Average answer length: {sum(word_counts)/max(1, len(word_counts)):.1f} words")
    print("This validates format and citations, not Kaggle's hidden semantic score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
