import csv

with open('submission_grandmaster_top1.csv', 'r', encoding='utf-8') as f:
    rows = {r['question_id']: r for r in csv.DictReader(f)}

# 1. Realignment for Q25:
# Question: "What's the difference between a warning that's just logged versus one that ends my assessment immediately?"
# Doc 06 Section 3 title: "Section 3: The Difference Between a Warning and a Termination"
# Doc 07 Section 3 title: "Section 3: Automatic Termination vs. Logged Flag"
rows['Q25']['cited_docs'] = "06_camera_microphone_face_verification_guide|07_permitted_prohibited_actions_policy"
rows['Q25']['cited_sections'] = "Section 3: The Difference Between a Warning and a Termination|Section 3: Automatic Termination vs. Logged Flag"
rows['Q25']['answer_text'] = (
    "A single or occasional warning is logged but does not end your assessment. Repeated or unresolved warnings, however, may be escalated and reviewed as part of the assessment's integrity check. "
    "Automatic termination is triggered by actions that directly match a defined prohibited pattern (e.g., a detected application-switch attempt, closing PSB, disabling the camera feed), ending the assessment immediately. "
    "In contrast, a logged flag applies to certain borderline signals (a brief face-verification warning, a momentary loss of camera feed due to lighting), which are logged for later review but do not end the assessment on their own."
)

# 2. Realignment for Q49:
# Question: "I know someone on the support team personally, can they fast-track my case and just tell me now that it won't count against me?"
# Doc 10 Section 3 explicitly addresses that Support CANNOT confirm the outcome of an integrity flag in real time!
rows['Q49']['cited_docs'] = "07_permitted_prohibited_actions_policy|10_general_support_escalation_guide"
rows['Q49']['cited_sections'] = "Section 4: This Policy Cannot Be Configured or Bypassed|Section 3: What Support Can and Cannot Do in Real Time"
rows['Q49']['answer_text'] = (
    "No. Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome of an integrity flag, or override the proctoring/integrity system — these all require the log-review process described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions any individual support agent can make on the spot. "
    "There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. "
    "Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
)

output_path = 'submission_apex_top1.csv'
fieldnames = ['question_id', 'answer_text', 'cited_docs', 'cited_sections']

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for qid in sorted(rows.keys()):
        writer.writerow(rows[qid])

print(f"Generated {output_path} with 50 rows.")

