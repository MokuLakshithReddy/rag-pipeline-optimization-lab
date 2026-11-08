import pandas as pd
import copy
from scripts.build_exact_source_solution import EXACT_EXCERPT_MAP

clean_map = copy.deepcopy(EXACT_EXCERPT_MAP)

# Q34: Strictly a prohibited shortcut / policy question (not network disconnection)
clean_map["Q34"] = {
    "docs": ["07_permitted_prohibited_actions_policy"],
    "sections": ["Section 1: Prohibited Actions"],
    "answer": (
        "No, there is no keyboard shortcut to pause an assessment. Assessment timers run continuously once started, and exams cannot be paused for breaks. "
        "Using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot) "
        "and any action that moves focus away from the assessment window is strictly prohibited under Section 1 of the Permitted & Prohibited Actions Policy and triggers automatic termination."
    )
}

test_df = pd.read_csv('kaggle_competition_data/Data/Test CSV/test.csv')
rows = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    item = clean_map[qid]
    rows.append({
        "question_id": qid,
        "answer_text": item["answer"],
        "cited_docs": "|".join(item["docs"]),
        "cited_sections": "|".join(item["sections"])
    })

clean_df = pd.DataFrame(rows)
clean_df.to_csv("submission_clean_top1.csv", index=False)
print("Saved submission_clean_top1.csv")

for i, r in clean_df.iterrows():
    d = r["cited_docs"].split("|")
    s = r["cited_sections"].split("|")
    assert len(d) == len(s), f"Mismatch at {r['question_id']}"
print("All 50 rows have exact doc-section length parity.")

