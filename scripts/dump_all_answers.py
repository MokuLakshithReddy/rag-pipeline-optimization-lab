import csv

with open('submission_direct_top1.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    qid = r['question_id']
    ans = r['answer_text']
    docs = r['cited_docs']
    secs = r['cited_sections']
    print(f"=== {qid} ===")
    print(f"Docs: {docs}")
    print(f"Secs: {secs}")
    print(f"Ans: {ans}\n")

