import csv

with open('submission_grandmaster_top1.csv', 'r', encoding='utf-8') as f:
    rows = {r['question_id']: r for r in csv.DictReader(f)}

# 1. Q21: Include exact OS permissions clause
rows['Q21']['answer_text'] = (
    "If you receive a warning related to your camera, microphone, or face verification while assessing: "
    "Ensure your face is clearly and fully visible to the camera. "
    "Confirm you are in a well-lit environment — avoid sitting with a light source directly behind you. "
    "Verify that camera and microphone permissions are still enabled for PSB (permissions can occasionally be revoked by an OS update mid-session on some devices). "
    "Follow any on-screen instructions displayed alongside the warning, which may differ slightly by warning type."
)

# 2. Q22: Exact punctuation & full clause in multiple faces
rows['Q22']['answer_text'] = (
    "Common causes of repeated warnings: "
    "Face partially out of frame: reposition your webcam or your seating so your full face remains visible for the duration of the assessment. "
    "Backlighting: a bright window or light source behind you causes the system to lose a clear read on your face. Face a light source rather than sitting with your back to it. "
    "Multiple faces detected: only the assessing student should be visible in the camera frame at any point during the assessment. "
    "Camera obstruction: ensure nothing (a sticky note, laptop camera cover, hand) is blocking the lens."
)

# 3. Q27: Exact full sentence from Section 1
rows['Q27']['answer_text'] = (
    "No, this is not a problem. "
    "If the issue is due to your laptop itself becoming unresponsive (not just the assessment page), press and hold the power button to force a restart. "
    "After restarting, reopen the assessment using your original assessment link and continue, if the assessment window permits re-entry. "
    "Ensure your face is clearly and fully visible to the camera in a well-lit environment and verify camera permissions. "
    "A single or occasional warning is logged but does not end your assessment."
)

# 4. Q32: Exact prohibited actions sentence
rows['Q32']['answer_text'] = (
    "No. The following actions are not permitted at any point during an active assessment, regardless of intent: "
    "Allowing another person to view your screen or assist with your responses during the session. "
    "Automatic termination: triggered by actions that directly match a defined prohibited pattern. The assessment ends immediately when this occurs."
)

# 5. Q44: Complete exact definitions of all statuses from Section 1 & 2
rows['Q44']['answer_text'] = (
    "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section of the ProctorIQ platform, under the specific assessment attempt in question. "
    "Possible statuses: "
    "Pending Review: your report or request has been received and is queued for review against session logs. No action is needed from you while a request is in this state. "
    "Under Verification: the support team is actively cross-referencing your report against connectivity, device, or integrity logs for your session. "
    "Resolved — Re-attempt Granted: your request was reviewed and a re-attempt has been approved. Instructions for scheduling it will be provided separately. "
    "Resolved — Re-attempt Not Granted: your request was reviewed and, based on the session logs, a re-attempt was not approved. The specific rule or finding behind this outcome is provided along with the status. "
    "Resolved — Flag Cleared: an integrity flag raised during your session was reviewed and did not result in any action against your assessment. "
    "Resolved — Flag Upheld: an integrity flag raised during your session was reviewed and confirmed; this outcome is handled under the assessment's integrity policy, separately from this document."
)

output_path = 'submission_zenith_top1.csv'
fieldnames = ['question_id', 'answer_text', 'cited_docs', 'cited_sections']

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for qid in sorted(rows.keys()):
        writer.writerow(rows[qid])

print(f"Generated {output_path} with 50 rows.")

