import csv
import glob
import os
from pathlib import Path

# Load knowledge base documents
kb_dir = 'kaggle_competition_data/Data/.md Files'
docs = {}
for p in sorted(glob.glob(os.path.join(kb_dir, '*.md'))):
    stem = Path(p).stem
    with open(p, 'r', encoding='utf-8') as f:
        docs[stem] = f.read()

# Load current submission
with open('submission_flawless_top1.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Let's inspect each answer and find sentences that might not be in the source document
for r in rows:
    qid = r['question_id']
    ans = r['answer_text']
    cited_docs = r['cited_docs'].split('|')
    
    combined_doc_text = " ".join([docs[d] for d in cited_docs])
    
    # Split answer into sentences
    sentences = [s.strip() for s in ans.split('.') if len(s.strip()) > 5]
    for s in sentences:
        # Check if key words of s are in combined_doc_text
        words = [w.lower().strip(".,!?:;\"'()[]{}") for w in s.split() if len(w) > 4]
        missing = [w for w in words if w not in combined_doc_text.lower()]
        if len(missing) > 2:
            print(f"[{qid}] Missing words in doc: {missing}")
            print(f"   Sentence: {s}")

