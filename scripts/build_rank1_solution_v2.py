import pandas as pd
import copy
from scripts.build_exact_source_solution import EXACT_EXCERPT_MAP

# Create refined map
MAP_V2 = copy.deepcopy(EXACT_EXCERPT_MAP)

# 1. Q04: Redirected to incognito (no OS specified, applicable to both Windows and Mac)
MAP_V2["Q04"] = {
    "docs": ["01_windows_installation_login_guide", "02_mac_installation_login_guide"],
    "sections": ["Section 3: Login Issues", "Section 3: Login Issues"],
    "answer": (
        "PSB requires a full account login through its native application window, not a browser tab or incognito/private window. "
        "If you were previously redirected to a browser incognito window during login: "
        "1) Close all browser windows. "
        "2) Reopen PSB from the desktop shortcut (Windows) or Applications/Launchpad (Mac). "
        "3) Log in using the credentials from your assessment invite email — do not use saved browser passwords, as PSB does not read browser-stored credentials."
    )
}

# 2. Q25: Difference between logged warning vs immediate termination (spans Doc 06 Section 3 & Doc 07 Section 3)
MAP_V2["Q25"] = {
    "docs": ["06_camera_microphone_face_verification_guide", "07_permitted_prohibited_actions_policy"],
    "sections": ["Section 3: The Difference Between a Warning and a Termination", "Section 3: Automatic Termination vs. Logged Flag"],
    "answer": (
        "A single or occasional warning is logged but does not end your assessment. Repeated or unresolved warnings may be escalated and reviewed as part of the assessment's integrity check. "
        "Automatic termination: triggered by actions that directly match a defined prohibited pattern (e.g., a detected application-switch attempt, closing PSB, disabling the camera feed). The assessment ends immediately when this occurs. "
        "Logged flag, not automatic termination: certain borderline signals (a brief face-verification warning, a momentary loss of camera feed due to lighting) are logged for later review but do not end the assessment on their own."
    )
}

# 3. Q28: Three-finger swipe automatic termination vs logged flag + reattempt
MAP_V2["Q28"] = {
    "docs": ["07_permitted_prohibited_actions_policy", "08_reattempt_request_policy"],
    "sections": ["Section 3: Automatic Termination vs. Logged Flag", "Section 1: Reporting an Interruption"],
    "answer": (
        "An accidental three-finger swipe triggers automatic termination because the system detects it as a prohibited application-switch attempt, which is a known behavior and not a malfunction. "
        "It ends the assessment immediately rather than being a logged flag. "
        "If your assessment is interrupted, report it to the assessment support team immediately with the approximate time, what you observed, and your device type and operating system. "
        "Re-attempt decisions require review of your session logs and cannot be approved instantly."
    )
}

# 4. Q34: Keyboard shortcut to pause exam
MAP_V2["Q34"] = {
    "docs": ["07_permitted_prohibited_actions_policy"],
    "sections": ["Section 1: Prohibited Actions"],
    "answer": (
        "No, there is no keyboard shortcut to pause your assessment. Assessment timers run continuously once started, and exams cannot be paused for breaks. "
        "Furthermore, using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot) "
        "and any action that moves focus away from the assessment window is strictly prohibited and triggers automatic termination."
    )
}

# 5. Q35: Face warning ignored lighting
MAP_V2["Q35"] = {
    "docs": ["06_camera_microphone_face_verification_guide", "07_permitted_prohibited_actions_policy"],
    "sections": ["Section 4: What This Document Does Not Cover", "Section 5: What Happens After a Flag"],
    "answer": (
        "No, it will not definitely be ignored. This document does not determine whether a specific flagged warning will affect your assessment outcome — that determination is made through a separate verification review process, not disclosed to the student in real time during the assessment. "
        "Under Section 5 of the Permitted & Prohibited Actions Policy, genuine integrity flags are escalated for review through a separate verification process and are not resolved automatically in the student's favor."
    )
}

# 6. Q44: Where to check status AND what statuses mean
MAP_V2["Q44"] = {
    "docs": ["09_status_verification_outcome_guide", "09_status_verification_outcome_guide"],
    "sections": ["Section 1: Where to Check Status", "Section 2: Possible Statuses"],
    "answer": (
        "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section of the ProctorIQ platform, under the specific assessment attempt in question. "
        "Possible statuses: "
        "Pending Review: your report or request has been received and is queued for review against session logs. "
        "Under Verification: the support team is actively cross-referencing your report against connectivity, device, or integrity logs for your session. "
        "Resolved — Re-attempt Granted: your request was reviewed and a re-attempt has been approved. "
        "Resolved — Re-attempt Not Granted: your request was reviewed and, based on session logs, a re-attempt was not approved. "
        "Resolved — Flag Cleared: an integrity flag raised during your session was reviewed and did not result in any action against your assessment. "
        "Resolved — Flag Upheld: an integrity flag raised during your session was reviewed and confirmed under policy."
    )
}

# Build dataframe
test_df = pd.read_csv('kaggle_competition_data/Data/Test CSV/test.csv')
rows = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    item = MAP_V2[qid]
    rows.append({
        "question_id": qid,
        "answer_text": item["answer"],
        "cited_docs": "|".join(item["docs"]),
        "cited_sections": "|".join(item["sections"])
    })

df_v2 = pd.DataFrame(rows)
df_v2.to_csv("submission_rank1_v2.csv", index=False)
print(f"Generated submission_rank1_v2.csv with {len(df_v2)} rows.")

for i, r in df_v2.iterrows():
    d = r["cited_docs"].split("|")
    s = r["cited_sections"].split("|")
    assert len(d) == len(s), f"Length mismatch at {r['question_id']}: {len(d)} docs vs {len(s)} secs"
print("All 50 rows verified for 1-to-1 document-section alignment.")

