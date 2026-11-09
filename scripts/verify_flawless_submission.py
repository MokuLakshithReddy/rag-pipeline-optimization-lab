import csv

with open('submission_flawless_top1.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}")
assert len(rows) == 50, f"Expected 50 rows, got {len(rows)}"

fieldnames = ["question_id", "answer_text", "cited_docs", "cited_sections"]
assert list(rows[0].keys()) == fieldnames, f"Headers mismatch: {rows[0].keys()}"

for i, r in enumerate(rows, 1):
    qid = r['question_id']
    expected_qid = f"Q{i:02d}"
    assert qid == expected_qid, f"Row {i} has qid {qid}, expected {expected_qid}"
    assert len(r['answer_text'].strip()) > 10, f"Row {qid} has too short answer"
    assert len(r['cited_docs'].strip()) > 0, f"Row {qid} has empty cited_docs"
    assert len(r['cited_sections'].strip()) > 0, f"Row {qid} has empty cited_sections"
    
    docs = r['cited_docs'].split('|')
    secs = r['cited_sections'].split('|')
    assert len(docs) == len(secs), f"Row {qid} alignment mismatch: {len(docs)} docs vs {len(secs)} secs"
    for d in docs:
        assert not d.endswith('.md'), f"Doc {d} has .md extension"
    for s in secs:
        assert s.startswith('Section '), f"Section {s} does not start with Section"

print("All 50 rows perfectly formatted, aligned, and verified!")

# Compare with submission_direct_top1.csv
with open('submission_direct_top1.csv', 'r', encoding='utf-8') as f:
    old_rows = {r['question_id']: r for r in csv.DictReader(f)}

print("\n--- Changes from submission_direct_top1.csv ---")
changed_count = 0
for r in rows:
    qid = r['question_id']
    old = old_rows[qid]
    diffs = []
    if r['cited_docs'] != old['cited_docs']:
        diffs.append(f"Docs: {old['cited_docs']} -> {r['cited_docs']}")
    if r['cited_sections'] != old['cited_sections']:
        diffs.append(f"Secs: {old['cited_sections']} -> {r['cited_sections']}")
    if r['answer_text'] != old['answer_text']:
        diffs.append(f"Ans changed")
    if diffs:
        changed_count += 1
        print(f"[{qid}]: {', '.join(diffs)}")

print(f"\nTotal questions refined: {changed_count}")

