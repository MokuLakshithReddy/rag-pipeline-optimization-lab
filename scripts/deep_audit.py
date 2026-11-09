import csv
import glob
import os
from pathlib import Path

# Load test questions
with open('kaggle_competition_data/Data/Test CSV/test.csv', 'r', encoding='utf-8') as f:
    test_rows = {r['question_id']: r['question'] for r in csv.DictReader(f)}

# Load current best submission (73.57000)
with open('submission_direct_top1.csv', 'r', encoding='utf-8') as f:
    sub_direct = {r['question_id']: r for r in csv.DictReader(f)}

# Load knowledge base documents
kb_dir = 'kaggle_competition_data/Data/.md Files'
docs = {}
for p in sorted(glob.glob(os.path.join(kb_dir, '*.md'))):
    stem = Path(p).stem
    with open(p, 'r', encoding='utf-8') as f:
        docs[stem] = f.read()

print(f"Total test questions: {len(test_rows)}")
print(f"Total KB docs: {len(docs)}")

# Let's inspect each question
for qid in sorted(test_rows.keys()):
    q = test_rows[qid]
    sub = sub_direct[qid]
    print(f"==================================================")
    print(f"[{qid}] QUESTION: {q}")
    print(f"DOCS: {sub['cited_docs']}")
    print(f"SECTIONS: {sub['cited_sections']}")
    print(f"ANSWER: {sub['answer_text']}")

