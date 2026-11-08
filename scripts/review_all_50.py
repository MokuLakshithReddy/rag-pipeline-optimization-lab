import pandas as pd

df = pd.read_csv('submission.csv')
test_df = pd.read_csv('kaggle_competition_data/Data/Test CSV/test.csv')

for i in range(len(df)):
    r = df.iloc[i]
    q = test_df.iloc[i]['question']
    print(f"[{r['question_id']}] {q}")
    print(f"   DOCS: {r['cited_docs']}")
    print(f"   SECS: {r['cited_sections']}")
    print(f"   ANS : {r['answer_text'][:120]}...")
    print()

