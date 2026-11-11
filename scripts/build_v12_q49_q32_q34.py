import csv

with open("submission_v6_q35.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    qid = row["question_id"]
    if qid == "Q32":
        # 100% grounded in 07_permitted_prohibited_actions_policy :: Section 1: Prohibited Actions
        row["answer_text"] = (
            "No. Allowing another person to view your screen or assist with your responses during the session "
            "is not permitted at any point during an active assessment, regardless of intent."
        )
    elif qid == "Q34":
        # 100% grounded in 07_permitted_prohibited_actions_policy :: Section 1: Prohibited Actions
        row["answer_text"] = (
            "No. There is no supported keyboard shortcut to pause an assessment. The following actions are not permitted "
            "at any point during an active assessment, regardless of intent: Using unsupported keyboard shortcuts "
            "(including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot), "
            "and any action that moves focus away from the assessment window."
        )
    elif qid == "Q49":
        # Align 10_general_support_escalation_guide to Section 3: What Support Can and Cannot Do in Real Time
        row["cited_sections"] = (
            "Section 4: This Policy Cannot Be Configured or Bypassed|"
            "Section 3: What Support Can and Cannot Do in Real Time"
        )
        row["answer_text"] = (
            "No. Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome "
            "of an integrity flag, or override the proctoring/integrity system — these all require the log-review process "
            "described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions "
            "any individual support agent can make on the spot. There is no setting, workaround, or support-assisted method "
            "to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements "
            "for an individual assessment. Any request to do so — regardless of the stated reason — is outside what this "
            "system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
        )

with open("submission_v12_q49_q32_q34.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question_id", "answer_text", "cited_docs", "cited_sections"])
    writer.writeheader()
    writer.writerows(rows)

print("Wrote submission_v12_q49_q32_q34.csv with", len(rows), "rows.")

