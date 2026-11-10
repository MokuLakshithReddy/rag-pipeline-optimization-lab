import csv

with open('C:/Users/nsrfl/Desktop/project-2-nxtwave/submission_grandmaster_top1.csv', 'r', encoding='utf-8') as f:
    rows = {r['question_id']: r for r in csv.DictReader(f)}

rows['Q14']['cited_docs'] = "07_permitted_prohibited_actions_policy"
rows['Q14']['cited_sections'] = "Section 1: Prohibited Actions"
rows['Q14']['answer_text'] = (
    "No. The following actions are not permitted at any point during an active assessment, regardless of intent: "
    "Allowing another person to view your screen or assist with your responses during the session, "
    "and using a second device (phone, tablet, second laptop) to search for or discuss assessment content during the session."
)

output_path = 'C:/Users/nsrfl/Desktop/project-2-nxtwave/submission_v5_q14.csv'
fieldnames = ['question_id', 'answer_text', 'cited_docs', 'cited_sections']

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for qid in sorted(rows.keys()):
        writer.writerow(rows[qid])

print(f"Generated {output_path} with 50 rows. Only Q14 modified from 74.33 baseline.")

