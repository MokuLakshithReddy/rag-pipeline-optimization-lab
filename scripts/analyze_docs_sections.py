import csv
import glob
import os
from pathlib import Path

# Load test questions
with open('kaggle_competition_data/Data/Test CSV/test.csv', 'r', encoding='utf-8') as f:
    test_rows = {r['question_id']: r['question'] for r in csv.DictReader(f)}

with open('submission_direct_top1.csv', 'r', encoding='utf-8') as f:
    sub = {r['question_id']: r for r in csv.DictReader(f)}

# Load knowledge base documents
kb_dir = 'kaggle_competition_data/Data/.md Files'
docs = {}
for p in sorted(glob.glob(os.path.join(kb_dir, '*.md'))):
    stem = Path(p).stem
    with open(p, 'r', encoding='utf-8') as f:
        docs[stem] = f.read()

# Let's check which questions might have potential improvements
# Let's inspect groups of questions:
# 1. Single doc vs multi doc
# 2. Refusal questions
# 3. Specific wording

print("=== MULTI-DOC QUESTIONS ===")
for qid in sorted(test_rows.keys()):
    if "|" in sub[qid]['cited_docs']:
        print(f"[{qid}] {test_rows[qid]}")
        print(f"   Docs: {sub[qid]['cited_docs']}")
        print(f"   Secs: {sub[qid]['cited_sections']}")
        print(f"   Ans: {sub[qid]['answer_text'][:100]}...\n")

print("\n=== SINGLE-DOC QUESTIONS ===")
for qid in sorted(test_rows.keys()):
    if "|" not in sub[qid]['cited_docs']:
        print(f"[{qid}] {test_rows[qid]}")
        print(f"   Doc: {sub[qid]['cited_docs']}")
        print(f"   Sec: {sub[qid]['cited_sections']}\n")

