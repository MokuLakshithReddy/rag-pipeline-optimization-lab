import csv

with open('submission_v5_q14.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

# In Q35:
# Current: docs=06_camera_microphone_face_verification_guide|09_status_verification_outcome_guide
#          secs=Section 4: What This Document Does Not Cover|Section 4: What This Document Does Not Do
# Hypothesis: Q35 is directly answered by Doc 06 Section 4: What This Document Does Not Cover.
for row in reader:
    if row['question_id'] == 'Q35':
        row['cited_docs'] = '06_camera_microphone_face_verification_guide'
        row['cited_sections'] = 'Section 4: What This Document Does Not Cover'
        row['answer_text'] = 'No. This document does not determine whether a specific flagged warning will affect your assessment outcome — that determination is made through a separate verification review process, not disclosed to the student in real time during the assessment. See the Status & Verification Outcome Guide for how outcomes are eventually communicated.'

with open('submission_v6_q35.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['question_id', 'answer_text', 'cited_docs', 'cited_sections'])
    writer.writeheader()
    writer.writerows(reader)

print('Successfully generated submission_v6_q35.csv with 50 rows')

