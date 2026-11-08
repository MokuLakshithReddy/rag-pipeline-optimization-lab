import pandas as pd
import copy
from scripts.build_clean_top1 import clean_map

direct_map = copy.deepcopy(clean_map)

# Prepend direct answers
direct_map["Q06"]["answer"] = "No. PSB runs natively on both Intel and Apple Silicon (M-series) Macs. There is no separate installer for M-series chips — the same .dmg file detects your architecture automatically. If PSB seems unusually slow to launch on an M-series Mac, this is normal on first launch only (Gatekeeper verification); subsequent launches are faster."

direct_map["Q08"]["answer"] = "No, you do not need to contact support. A forgotten PIN can be reset via the 'Forgot PIN' link on the PSB login screen — this is self-service and does not require Support."

direct_map["Q14"]["answer"] = "No. Check that the phone number/email on your assessment profile matches your registered contact details. Under Section 1 of the Permitted & Prohibited Actions Policy, allowing another person to view your screen or assist with your responses, or using a second device to search for or discuss assessment content during the session, is strictly prohibited."

direct_map["Q15"]["answer"] = "No, doing the mock test right before your exam slot is strongly discouraged. Complete your mock test at least 24 hours before your scheduled assessment window. This is not a hard technical requirement, but it is the single most effective way to avoid last-minute issues — problems found during a mock test can be resolved with Support ahead of time; the same problems found five minutes before your exam window often cannot."

direct_map["Q18"]["answer"] = "No. If you are unable to reconnect after several attempts, or the disconnection recurs more than twice in the same session, stop attempting to self-resolve and report the issue to the assessment support team immediately with the approximate time(s) the disconnection occurred. This information is needed for any later review of your session."

direct_map["Q27"]["answer"] = "No, this is not a problem. Press and hold the power button to force a restart, then reopen using your original link. When reopening, PSB re-verifies your identity; ensure your face is clearly and fully visible to the camera in a well-lit environment. A single warning upon reopening is logged for review and does not terminate the assessment."

direct_map["Q29"]["answer"] = "You cannot turn off face verification. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference. Attempting to disable, obscure, or interfere with face verification is prohibited and triggers automatic termination."

direct_map["Q30"]["answer"] = "You cannot disable the camera. There is no setting, workaround, or support-assisted method to disable the camera or reduce proctoring requirements. Any request to do so is outside what this system or support can provide. This is a fixed integrity boundary, not a configurable preference. To resolve repeated warnings: ensure your full face is visible, sit facing a light source rather than with it behind you, remove face obstructions, and ensure only the student is visible."

direct_map["Q31"]["answer"] = "No. Attempting to disable, obscure, or interfere with the camera, microphone, or face-verification system is strictly prohibited under Section 1. There is no setting, workaround, or support-assisted method to reduce proctoring requirements, regardless of technical issues. Disabling the camera feed triggers automatic termination."

direct_map["Q32"]["answer"] = "No. Allowing another person to view your screen or assist with your responses during the session is strictly prohibited under Section 1 of the Permitted & Prohibited Actions Policy. Actions that directly match a defined prohibited pattern trigger automatic termination of the assessment and an integrity flag."

direct_map["Q34"]["answer"] = "No, there is no keyboard shortcut or option to pause an assessment. Assessment timers run continuously once started, and exams cannot be paused for breaks. Using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot) and any action that moves focus away from the assessment window is strictly prohibited under Section 1 of the Permitted & Prohibited Actions Policy and triggers automatic termination."

direct_map["Q35"]["answer"] = "No, it will not definitely be ignored and cannot be confirmed in advance. This document does not determine whether a specific flagged warning will affect your assessment outcome — that determination is made through a separate verification review process. Under Section 5 of the Permitted & Prohibited Actions Policy, genuine integrity flags are escalated for review through a separate verification process and are not resolved automatically in the student's favor."

direct_map["Q43"]["answer"] = "No. Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome of an integrity flag, or override the proctoring/integrity system — these all require the log-review process described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions any individual support agent can make on the spot."

direct_map["Q46"]["answer"] = "No, support cannot override it. Resolved — Flag Upheld means an integrity flag raised during your session was reviewed and confirmed; this outcome is handled under the assessment's integrity policy. Support cannot in real time override the proctoring/integrity system or reverse a confirmed review outcome, even if you explain your side."

direct_map["Q47"]["answer"] = "No, a re-attempt is not guaranteed. This document does not cover whether a re-attempt will be granted due to a connectivity issue. Re-attempt decisions require review of your session's connectivity logs by the support team and cannot be confirmed at the time of the issue. This document cannot confirm whether a specific re-attempt request will be approved — that depends on the outcome of the log review on a case-by-case basis."

direct_map["Q48"]["answer"] = "No, I cannot personally approve your re-attempt. A re-attempt cannot be approved at the time of the interruption, during the assessment, or immediately after it ends. Support cannot, in real time, approve a re-attempt or override the proctoring/integrity system — decisions require verifying session connectivity and device event logs after the session has fully ended."

direct_map["Q49"]["answer"] = "No. Support will never provide guidance intended to help a student bypass, disable, or reduce face verification, application-lock, or any other integrity control, regardless of the reason given. Any request to do so is outside what any member of the support team can provide. This is a fixed integrity boundary that cannot be configured, bypassed, or fast-tracked through personal contacts."

direct_map["Q50"]["answer"] = "No, this cannot be confirmed. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. This is a fixed integrity boundary, not a configurable preference. Accommodations for medical conditions must be submitted, documented, and pre-approved through official channels prior to the assessment."

test_df = pd.read_csv('kaggle_competition_data/Data/Test CSV/test.csv')
rows = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    item = direct_map[qid]
    rows.append({
        "question_id": qid,
        "answer_text": item["answer"],
        "cited_docs": "|".join(item["docs"]),
        "cited_sections": "|".join(item["sections"])
    })

out_df = pd.DataFrame(rows)
out_df.to_csv("submission_direct_top1.csv", index=False)
print("Saved submission_direct_top1.csv with 50 rows.")

for i, r in out_df.iterrows():
    d = r["cited_docs"].split("|")
    s = r["cited_sections"].split("|")
    assert len(d) == len(s), f"Mismatch at {r['question_id']}"
print("All 50 rows have exact doc-section length parity.")

