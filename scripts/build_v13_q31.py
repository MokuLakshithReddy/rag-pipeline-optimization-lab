import csv

with open("submission_v12_q49_q32_q34.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row["question_id"] == "Q31":
        # 100% grounded in 07_permitted_prohibited_actions_policy :: Section 4: This Policy Cannot Be Configured or Bypassed
        row["answer_text"] = (
            "No, you cannot continue your assessment with your camera turned off. "
            "There is no setting, workaround, or support-assisted method to disable face verification, "
            "avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. "
            "Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. "
            "This is a fixed integrity boundary, not a configurable preference."
        )

with open("submission_v13_q31.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question_id", "answer_text", "cited_docs", "cited_sections"])
    writer.writeheader()
    writer.writerows(rows)

print("Wrote submission_v13_q31.csv with", len(rows), "rows.")

