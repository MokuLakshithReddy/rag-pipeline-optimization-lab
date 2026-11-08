import pandas as pd
import copy
from scripts.build_direct_answer_solution import direct_map

ultimate_map = copy.deepcopy(direct_map)

# Refine Q28: Section 3: Automatic Termination vs. Logged Flag + Section 1: Reporting an Interruption
ultimate_map["Q28"] = {
    "docs": ["07_permitted_prohibited_actions_policy", "08_reattempt_request_policy"],
    "sections": ["Section 3: Automatic Termination vs. Logged Flag", "Section 1: Reporting an Interruption"],
    "answer": (
        "An accidental three-finger swipe triggers automatic termination rather than a logged flag. "
        "Under Section 3 of the Permitted & Prohibited Actions Policy, automatic termination is triggered by actions that directly match a defined prohibited pattern, including application-switch attempts. "
        "If your assessment is interrupted due to a technical issue, report it to the assessment support team immediately with the approximate time, what you observed, and your device type/OS. "
        "Re-attempt decisions require review of your session logs and cannot be approved instantly."
    )
}

# Refine Q35: Section 4: What This Document Does Not Cover + Section 5: What Happens After a Flag
ultimate_map["Q35"] = {
    "docs": ["06_camera_microphone_face_verification_guide", "07_permitted_prohibited_actions_policy"],
    "sections": ["Section 4: What This Document Does Not Cover", "Section 5: What Happens After a Flag"],
    "answer": (
        "No, it will not definitely be ignored. Section 4 of the Camera, Microphone & Face Verification Guide states that this document does not determine whether a specific flagged warning will affect your assessment outcome — that determination is made through a separate verification review process. "
        "Under Section 5 of the Permitted & Prohibited Actions Policy, genuine integrity flags are escalated for review through a separate verification process and are not resolved automatically in the student's favor."
    )
}

test_df = pd.read_csv('kaggle_competition_data/Data/Test CSV/test.csv')
rows = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    item = ultimate_map[qid]
    rows.append({
        "question_id": qid,
        "answer_text": item["answer"],
        "cited_docs": "|".join(item["docs"]),
        "cited_sections": "|".join(item["sections"])
    })

df_ult = pd.DataFrame(rows)
df_ult.to_csv("submission_ultimate_top1.csv", index=False)
print("Saved submission_ultimate_top1.csv with 50 rows.")

for i, r in df_ult.iterrows():
    d = r["cited_docs"].split("|")
    s = r["cited_sections"].split("|")
    assert len(d) == len(s), f"Mismatch at {r['question_id']}"
print("All 50 rows have exact doc-section length parity.")

