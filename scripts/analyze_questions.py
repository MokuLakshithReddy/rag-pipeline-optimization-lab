import pandas as pd
import json

df = pd.read_csv('kaggle_competition_data/Data/Test CSV/test.csv')
for _, row in df.iterrows():
    print(f"{row['question_id']}: {row['question']}")
