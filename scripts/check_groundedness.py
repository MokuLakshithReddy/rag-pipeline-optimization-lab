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

# For each section in docs, extract text
sections_dict = {}
for doc_id, text in docs.items():
    for part in text.split('\n## '):
        if not part.strip():
            continue
        lines = part.strip().split('\n', 1)
        sec_title = lines[0].strip().lstrip('#').strip()
        sec_text = lines[1] if len(lines) > 1 else ''
        sections_dict[(doc_id, sec_title)] = sec_text

print(f"Extracted {len(sections_dict)} sections.")

# Check overlap and faithfulness
for qid in sorted(test_rows.keys()):
    q = test_rows[qid]
    r = sub[qid]
    c_docs = r['cited_docs'].split('|')
    c_secs = r['cited_sections'].split('|')
    
    # Combined section texts
    source_texts = []
    for d, s in zip(c_docs, c_secs):
        st = sections_dict.get((d, s), '')
        if not st:
            # find closest
            for (kd, ks), v in sections_dict.items():
                if kd == d and s in ks:
                    st = v
                    break
        source_texts.append(st)
    combined_source = " ".join(source_texts)
    
    ans = r['answer_text']
    # Calculate word overlap
    ans_words = set(w.lower().strip(".,!?:;\"'()[]{}") for w in ans.split())
    source_words = set(w.lower().strip(".,!?:;\"'()[]{}") for w in combined_source.split())
    
    hallucinated_words = ans_words - source_words - {'yes', 'no', 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'of', 'and', 'or', 'for', 'it', 'its', 'you', 'your', 'my', 'i', 'this', 'that', 'not', 'do', 'does', 'did', 'cannot', 'can', 'be', 'by', 'as', 'at', 'with', 'from', 'if'}
    
    if len(hallucinated_words) > 5:
        print(f"[{qid}] {q}")
        print(f"   Cited: {r['cited_docs']} | {r['cited_sections']}")
        print(f"   Hallucinated/External words ({len(hallucinated_words)}): {hallucinated_words}")
        print(f"   Ans: {ans}")
        print("-" * 50)

