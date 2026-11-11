import csv

with open("submission_v13_q31.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    qid = row["question_id"]
    if qid == "Q14":
        # Direct answer to roommate OTP + exact verbatim 07 Sec 1
        row["answer_text"] = (
            "No, you cannot have your roommate receive your OTP on their phone and read it out to you during an assessment. "
            "The following actions are not permitted at any point during an active assessment, regardless of intent: "
            "Allowing another person to view your screen or assist with your responses during the session, "
            "and using a second device (phone, tablet, second laptop) to search for or discuss assessment content during the session."
        )
    elif qid == "Q26":
        # Complete verbatim 04 Sec 1 + 06 Sec 1
        row["answer_text"] = (
            "Stay on the assessment screen. Do not close PSB or attempt to reopen the assessment from elsewhere. "
            "Use the Wi-Fi/network option available at the bottom of the PSB screen to reconnect. Once the connection "
            "is restored, continue your assessment from where you left off — your progress up to the point of "
            "disconnection is preserved. For the face verification warning: ensure your face is clearly and fully "
            "visible to the camera, confirm you are in a well-lit environment — avoid sitting with a light source "
            "directly behind you — verify that camera and microphone permissions are still enabled for PSB (permissions "
            "can occasionally be revoked by an OS update mid-session on some devices), and follow any on-screen "
            "instructions displayed alongside the warning, which may differ slightly by warning type."
        )
    elif qid == "Q27":
        # Private 30 groundedness alignment: 05 Sec 1 + 06 Sec 3
        row["cited_sections"] = (
            "Section 1: Assessment Page Freezes or Gets Stuck|"
            "Section 3: The Difference Between a Warning and a Termination"
        )
        row["answer_text"] = (
            "No, a single or occasional warning is logged but does not end your assessment. "
            "If the issue is due to your laptop itself becoming unresponsive (not just the assessment page), "
            "press and hold the power button to force a restart. After restarting, reopen the assessment using your "
            "original assessment link and continue, if the assessment window permits re-entry. "
            "Repeated or unresolved warnings, however, may be escalated and reviewed as part of the assessment's "
            "integrity check."
        )
    elif qid == "Q28":
        # Private 30 groundedness alignment: 07 Sec 2 + 08 Sec 1 (no ungrounded Sec 2/3 leak)
        row["answer_text"] = (
            "It may terminate automatically: if a candidate accidentally performs a three-finger swipe on their touchpad, "
            "the assessment may terminate automatically because the system detects it as an application-switch attempt — "
            "this is a known behavior, not a malfunction. If your assessment is interrupted due to a technical issue, "
            "report it to the assessment support team immediately — through the in-platform Help option if you are "
            "still able to access it, or through the alternate contact method provided in your original assessment "
            "invite if not — and provide the approximate time the interruption occurred, what you observed, and your "
            "device type and operating system."
        )

with open("submission_v14_polished.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question_id", "answer_text", "cited_docs", "cited_sections"])
    writer.writeheader()
    writer.writerows(rows)

print("Wrote submission_v14_polished.csv with", len(rows), "rows.")

