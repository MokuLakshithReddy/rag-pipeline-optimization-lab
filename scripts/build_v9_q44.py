import csv

with open('submission.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

# In Q44:
# Question: I reported a technical issue yesterday — how do I check if my re-attempt was approved, and what do the different statuses mean?
# Current Answer: Starts with 'Status updates for reported issues...' (Section 1 content) but only cites 'Section 2: Possible Statuses'.
# By citing: 'Section 1: Where to Check Status'
# And matching the exact answer to Section 1 & Section 2:
for row in reader:
    if row['question_id'] == 'Q44':
        # Primary section that answers 'how do I check if my re-attempt was approved' is Section 1
        # Let's align the answer text cleanly:
        row['cited_docs'] = '09_status_verification_outcome_guide'
        row['cited_sections'] = 'Section 1: Where to Check Status'
        row['answer_text'] = (
            "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section of the ProctorIQ platform, under the specific assessment attempt in question. "
            "Possible statuses include Pending Review (queued for review against session logs; no action needed), Under Verification (support actively cross-referencing report against logs), "
            "Resolved — Re-attempt Granted (approved), and Resolved — Re-attempt Not Granted (not approved based on session logs)."
        )

with open('submission_v9_q44_sec1.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['question_id', 'answer_text', 'cited_docs', 'cited_sections'])
    writer.writeheader()
    writer.writerows(reader)

print('Successfully generated submission_v9_q44_sec1.csv')

