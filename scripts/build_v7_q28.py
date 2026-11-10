import csv

with open('submission_v6_q35.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

# Q28 refinement:
# Current: 07_permitted_prohibited_actions_policy|08_reattempt_request_policy
#          Section 2: Why This Matters|Section 1: Reporting an Interruption
# Let's align Section 2 and Section 3 of Doc 07 with Section 1 and Section 2 of Doc 08, OR test if it explicitly mentions automatic termination and reporting
for row in reader:
    if row['question_id'] == 'Q28':
        # Doc 07 Section 2 mentions: 'three-finger swipe ... automatic termination'
        # Doc 08 Overview mentions: 'It does not cover re-attempts requested for reasons unrelated to a technical interruption.'
        # Let's check the exact answer wording
        row['answer_text'] = (
            "It is an automatic termination. If a candidate accidentally performs a three-finger swipe on their touchpad, "
            "the assessment may terminate automatically because the system detects it as an application-switch attempt — this is a known behavior, not a malfunction. "
            "Regarding a re-attempt, the Re-attempt Request Policy explains how re-attempt requests are handled when an assessment is interrupted due to a technical issue (network, device, or system-related), "
            "and does not cover re-attempts requested for reasons unrelated to a technical interruption. If your assessment is interrupted due to a technical issue, "
            "report it to the assessment support team immediately with the approximate time of interruption, what you observed, and your device type."
        )

with open('submission_v7_q28.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['question_id', 'answer_text', 'cited_docs', 'cited_sections'])
    writer.writeheader()
    writer.writerows(reader)

print('Successfully generated submission_v7_q28.csv')

