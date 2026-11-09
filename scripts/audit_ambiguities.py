import csv

with open('submission_flawless_top1.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

with open('kaggle_competition_data/Data/Test CSV/test.csv', 'r', encoding='utf-8') as f:
    q_dict = {r['question_id']: r['question'] for r in csv.DictReader(f)}

# Let's inspect each row and see potential ambiguities
for r in rows:
    qid = r['question_id']
    q = q_dict[qid]
    docs = r['cited_docs']
    secs = r['cited_sections']
    ans = r['answer_text']
    
    # Let's highlight multi-doc or questions with potential alternative sections
    print(f"[{qid}] {q}")
    print(f"  Docs: {docs}")
    print(f"  Secs: {secs}")
    print(f"  Ans:  {ans[:100]}...")
    print()

