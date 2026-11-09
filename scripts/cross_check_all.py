import csv
import glob
import os
from pathlib import Path

# Load test questions
with open('kaggle_competition_data/Data/Test CSV/test.csv', 'r', encoding='utf-8') as f:
    test_rows = {r['question_id']: r['question'] for r in csv.DictReader(f)}

# Load current best submission (73.57000)
with open('submission_direct_top1.csv', 'r', encoding='utf-8') as f:
    sub = {r['question_id']: r for r in csv.DictReader(f)}

kb_dir = 'kaggle_competition_data/Data/.md Files'
docs = {}
for p in sorted(glob.glob(os.path.join(kb_dir, '*.md'))):
    stem = Path(p).stem
    with open(p, 'r', encoding='utf-8') as f:
        docs[stem] = f.read()

# Let's check where each question's key phrases appear across the entire KB!
for qid in sorted(test_rows.keys()):
    q = test_rows[qid]
    current_docs = sub[qid]['cited_docs'].split('|')
    current_secs = sub[qid]['cited_sections'].split('|')
    
    # Check for potential alternative docs/sections
    print(f"[{qid}] {q}")
    print(f"  Current: {sub[qid]['cited_docs']}  ||  {sub[qid]['cited_sections']}")
    
    # Search for key words from q in docs
    # print short summary of answer
    ans = sub[qid]['answer_text']
    first_sentence = ans.split('.')[0] if '.' in ans else ans[:60]
    print(f"  Answer start: {first_sentence}")
    print()

