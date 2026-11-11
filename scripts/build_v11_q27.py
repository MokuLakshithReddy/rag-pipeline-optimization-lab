import csv

with open('submission.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

# Hypothesis 7:
# Look closely at Q27 and Q28!
# First, Q27:
# Question: "My laptop froze, I had to restart it, and right after reopening I got a face verification warning — is that a problem?"
# Currently cites:
#   DOCS: 05_device_system_issue_playbook|06_camera_microphone_face_verification_guide
#   SECS: Section 1: Assessment Page Freezes or Gets Stuck|Section 1: Responding to a Warning
#
# Wait: Does Section 1 of Doc 06 answer "is that a problem?"
# No! Section 1 is "Responding to a Warning".
# Section 3 is "The Difference Between a Warning and a Termination", which explicitly states:
#   "A single or occasional warning is logged but does not end your assessment."
# Even in our Q27 answer right now, we already quote Section 3 ("A single or occasional warning is logged but does not end your assessment")!
# AND what about Doc 05 Section 1 vs Section 2?
# Doc 05 Section 1 says: "If the issue is due to your laptop itself becoming unresponsive (not just the assessment page), press and hold the power button to force a restart. After restarting, reopen the assessment using your original assessment link and continue, if the assessment window permits re-entry." -> That IS Section 1!
# Wait: what about Q27 changing Doc 06 from Section 1 to Section 3?
# Let's check what Q27 says in answer_text right now:
# "No, this is not a problem. Press and hold the power button to force a restart, then reopen using your original assessment link. When reopening, ensure your face is clearly and fully visible to the camera in a well-lit environment and verify camera permissions. A single or occasional warning is logged but does not end your assessment."

for row in reader:
    if row['question_id'] == 'Q27':
        row['cited_sections'] = 'Section 1: Assessment Page Freezes or Gets Stuck|Section 3: The Difference Between a Warning and a Termination'
        row['answer_text'] = (
            "No, a single warning right after reopening is not a problem on its own. "
            "If your laptop itself became unresponsive, forcing a restart and reopening the assessment using your original link is the expected recovery step if the assessment window permits re-entry. "
            "A single or occasional face verification warning is logged but does not end your assessment. "
            "Ensure your face is clearly and fully visible to the camera, confirm you are in a well-lit environment, and verify camera permissions remain enabled for PSB."
        )

with open('submission_v11_q27.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['question_id', 'answer_text', 'cited_docs', 'cited_sections'])
    writer.writeheader()
    writer.writerows(reader)

print('Generated submission_v11_q27.csv successfully')

