import os, glob, re, json
import pandas as pd

df = pd.read_csv('submission.csv')

# Load the 10 documents
docs = {}
for path in sorted(glob.glob('kaggle_competition_data/Data/.md Files/*.md')):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, 'r', encoding='utf-8') as f:
        docs[stem] = f.read()

# Let's inspect questions Q36 through Q43
for qid in ["Q36", "Q37", "Q38", "Q39", "Q40", "Q41", "Q42", "Q43"]:
    r = df[df['question_id'] == qid].iloc[0]
    print(f"=== {qid} ===")
    print("SUBMISSION:", r['answer_text'])
    d = r['cited_docs']
    s = r['cited_sections']
    print(f"CITED: {d} -> {s}")
    if d in docs:
        content = docs[d]
        if s in content:
            sec_part = content.split(f"## {s}")[1].split("\n## ")[0]
            print("SOURCE SEC EXCERPT:\n", sec_part[:300].strip())
    print("-" * 50)

