import pandas as pd

df1 = pd.read_csv('temp_kernel/submission_optimized.csv')
df2 = pd.read_csv('submission.csv')

for i in range(50):
    r1 = df1.iloc[i]
    r2 = df2.iloc[i]
    diffs = []
    if r1['cited_docs'] != r2['cited_docs']:
        diffs.append(f"docs: '{r1['cited_docs']}' -> '{r2['cited_docs']}'")
    if r1['cited_sections'] != r2['cited_sections']:
        diffs.append(f"secs: '{r1['cited_sections']}' -> '{r2['cited_sections']}'")
    if r1['answer_text'] != r2['answer_text']:
        diffs.append("answer modified")
    if diffs:
        print(f"{r1['question_id']}: {', '.join(diffs)}")

