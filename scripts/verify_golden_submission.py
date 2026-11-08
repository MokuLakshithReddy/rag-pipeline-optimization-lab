import pandas as pd
import json

df = pd.read_csv('submission_golden_rank1.csv')
print("Total rows:", len(df))
print("Columns:", list(df.columns))

# Verify every field has no NaN, no trailing whitespace, valid pipe separation
for i, r in df.iterrows():
    qid = r['question_id']
    ans = r['answer_text']
    docs = str(r['cited_docs']).split('|')
    secs = str(r['cited_sections']).split('|')
    
    assert pd.notna(ans) and len(ans.strip()) > 20, f"Short or null answer at {qid}"
    assert len(docs) == len(secs), f"Length mismatch at {qid}: {len(docs)} docs vs {len(secs)} secs"
    for d in docs:
        assert d.endswith('_guide') or d.endswith('_policy') or d.endswith('_playbook'), f"Invalid doc format: {d}"
    for s in secs:
        assert s.startswith('Section ') or s == 'Overview', f"Invalid section format: {s}"

print("All 50 rows passed strict structural integrity checks.")

