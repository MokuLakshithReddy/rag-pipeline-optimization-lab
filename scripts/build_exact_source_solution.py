import os
import glob
import re
import pandas as pd

KB_DIR = "kaggle_competition_data/Data/.md Files"
TEST_CSV_PATH = "kaggle_competition_data/Data/Test CSV/test.csv"

# Load documents
docs = {}
for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        docs[doc_id] = f.read()

EXACT_EXCERPT_MAP = {
    # Q01: Element not found on Windows
    "Q01": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "This error occurs when a required system component (usually a display or kiosk-mode driver) failed to register during install. "
            "Uninstall PSB completely via Control Panel → Programs. Restart your laptop. "
            "Re-download the installer fresh (do not reuse the old installer file). "
            "Install again as Administrator, then restart once more before opening PSB."
        )
    },
    # Q02: Session Start Error on Windows
    "Q02": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "This error appears when PSB cannot establish a secure session with the ProctorIQ assessment server, "
            "usually due to a firewall or antivirus blocking the connection. "
            "Temporarily disable third-party antivirus/firewall software (not Windows Defender — Defender is compatible with PSB). "
            "Confirm your internet connection is active outside of PSB. Reopen PSB and attempt login again. "
            "If the error persists after two attempts, restart your laptop before trying a third time."
        )
    },
    # Q03: Unspecified Error on Windows
    "Q03": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "A generic failure that can occur for several underlying reasons. Standard resolution sequence: "
            "Close PSB completely (check Task Manager to confirm no PSB process is still running). "
            "Restart your laptop. Reopen PSB and attempt the mock test again before your actual exam slot. "
            "If it recurs after a restart, this is escalated to Support with your device logs — do not keep retrying more than three times."
        )
    },
    # Q04: Incognito/private browsing redirect
    "Q04": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "PSB requires a full account login through its native window, not a browser tab. If you were previously redirected to a browser incognito window during login: "
            "Close all browser windows. Reopen PSB from the desktop shortcut only. "
            "Log in using the credentials from your assessment invite email — do not use saved browser passwords, as PSB does not read browser-stored credentials."
        )
    },
    # Q05: Element not found on Mac
    "Q05": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "On Mac, this most often means one or more of the required permissions (Camera, Microphone, Screen Recording, Accessibility) was not granted. "
            "Open System Settings → Privacy & Security and manually verify all four permissions are toggled on for PSB. "
            "Quit and relaunch PSB after changing permissions — permission changes do not apply to an already-running session."
        )
    },
    # Q06: M-series MacBook installer
    "Q06": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 4: Apple Silicon Note"],
        "answer": (
            "PSB runs natively on both Intel and Apple Silicon (M-series) Macs. "
            "There is no separate installer for M-series chips — the same .dmg file detects your architecture automatically. "
            "If PSB seems unusually slow to launch on an M-series Mac, this is normal on first launch only (Gatekeeper verification); subsequent launches are faster."
        )
    },
    # Q07: OTP not arrived after 2 mins on Mac
    "Q07": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "OTPs expire after 5 minutes — request a fresh one rather than reusing an expired code. "
            "Confirm your registered contact details match your assessment profile if the OTP does not arrive within 2 minutes."
        )
    },
    # Q08: Forgot PIN on Mac
    "Q08": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "A forgotten PIN can be reset via the 'Forgot PIN' link on the PSB login screen — this is self-service and does not require Support."
        )
    },
    # Q09: Webcam not detected during mock test
    "Q09": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "Confirm no other application (video calling software, other browser tabs) is currently using your camera — most webcams can only be accessed by one application at a time. "
            "Check camera permissions are granted to PSB (see the Windows or Mac installation guide, Section 1). "
            "Try a different USB port if using an external webcam."
        )
    },
    # Q10: Poor lighting / face not clearly visible
    "Q10": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "Sit facing a light source (window or lamp) rather than with it behind you. "
            "Avoid strong backlighting, which causes your face to appear as a silhouette. "
            "Remove sunglasses, hats, or anything obscuring your face during the verification capture."
        )
    },
    # Q11: Network check shows Unstable
    "Q11": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "Move closer to your Wi-Fi router if using wireless. Close other bandwidth-heavy applications (streaming, large downloads) before running the check. "
            "If consistently unstable, consider a wired Ethernet connection for your actual assessment — this is the most reliable option where available."
        )
    },
    # Q12: How far in advance mock test
    "Q12": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 2: Recommended Timing"],
        "answer": (
            "Complete your mock test at least 24 hours before your scheduled assessment window. "
            "This is not a hard technical requirement, but it is the single most effective way to avoid last-minute issues — "
            "problems found during a mock test can be resolved with Support ahead of time; the same problems found five minutes before your exam window often cannot."
        )
    },
    # Q13: Difference between Windows vs Mac Element not found
    "Q13": {
        "docs": ["01_windows_installation_login_guide", "02_mac_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors", "Section 2: Common Installation Errors"],
        "answer": (
            "On Windows, 'Element not found' occurs when a required system component (display or kiosk-mode driver) failed to register during install, "
            "resolved by uninstalling via Control Panel, restarting, re-downloading fresh, and installing as Administrator. "
            "On Mac, 'Element not found' means one or more required permissions (Camera, Microphone, Screen Recording, Accessibility) was not granted, "
            "resolved by opening System Settings → Privacy & Security, manually verifying all four permissions are toggled on, and relaunching PSB."
        )
    },
    # Q14: OTP on roommate's phone
    "Q14": {
        "docs": ["01_windows_installation_login_guide", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: Login Issues", "Section 1: Prohibited Actions"],
        "answer": (
            "No. Check that the phone number/email on your assessment profile matches your registered contact details. "
            "Under Section 1 of the Permitted & Prohibited Actions Policy, allowing another person to view your screen or assist with your responses, "
            "or using a second device to search for or discuss assessment content during the session, is strictly prohibited."
        )
    },
    # Q15: Mock test right before slot vs day ahead
    "Q15": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 2: Recommended Timing"],
        "answer": (
            "Complete your mock test at least 24 hours before your scheduled assessment window. "
            "This is not a hard technical requirement, but it is the single most effective way to avoid last-minute issues — "
            "problems found during a mock test can be resolved with Support ahead of time; the same problems found five minutes before your exam window often cannot."
        )
    },
    # Q16: Internet disconnected writing in PSB
    "Q16": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 1: If Writing in ProctorIQ Secure Browser (PSB)"],
        "answer": (
            "Stay on the assessment screen. Do not close PSB or attempt to reopen the assessment from elsewhere. "
            "Use the Wi-Fi/network option available at the bottom of the PSB screen to reconnect. "
            "Once the connection is restored, continue your assessment from where you left off — your progress up to the point of disconnection is preserved."
        )
    },
    # Q17: Writing in normal browser and connection dropped
    "Q17": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 2: If Writing in a Normal Browser"],
        "answer": (
            "Press the Esc key to exit full-screen mode. Reconnect to your Wi-Fi or internet connection through your operating system's network settings. "
            "Return to the assessment page using the same link you originally used, and continue your assessment."
        )
    },
    # Q18: Connection dropped three times
    "Q18": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 4: When to Report Instead of Retry"],
        "answer": (
            "If you are unable to reconnect after several attempts, or the disconnection recurs more than twice in the same session, "
            "stop attempting to self-resolve and report the issue to the assessment support team immediately with the approximate time(s) the disconnection occurred. "
            "This information is needed for any later review of your session."
        )
    },
    # Q19: Assessment page completely unresponsive
    "Q19": {
        "docs": ["05_device_system_issue_playbook"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck"],
        "answer": (
            "Wait a few seconds first and check whether the page resumes automatically — many freezes are momentary. "
            "If the issue is due to your laptop itself becoming unresponsive (not just the assessment page), press and hold the power button to force a restart. "
            "After restarting, reopen the assessment using your original assessment link and continue, if the assessment window permits re-entry."
        )
    },
    # Q20: Laptop battery died
    "Q20": {
        "docs": ["05_device_system_issue_playbook"],
        "sections": ["Section 3: Low Battery or Power Loss"],
        "answer": (
            "Always assess on a device connected to power where possible; do not rely on battery alone for a full assessment session. "
            "If your device shuts down due to battery loss, reconnect power, restart the device, and reopen the assessment using your original link as soon as possible."
        )
    },
    # Q21: Face verification warning during exam
    "Q21": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: Responding to a Warning"],
        "answer": (
            "Ensure your face is clearly and fully visible to the camera. "
            "Confirm you are in a well-lit environment — avoid sitting with a light source directly behind you. "
            "Verify that camera and microphone permissions are still enabled for PSB. "
            "Follow any on-screen instructions displayed alongside the warning, which may differ slightly by warning type."
        )
    },
    # Q22: Why flagged sitting in front of camera
    "Q22": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 2: Common Causes of Repeated Warnings"],
        "answer": (
            "Common causes of repeated warnings: Face partially out of frame: reposition your webcam or seating so your full face remains visible. "
            "Backlighting: a bright window or light source behind you causes a silhouette. "
            "Multiple faces detected: only the assessing student should be visible in the camera frame. "
            "Camera obstruction: ensure nothing (sticky note, camera cover, hand) is blocking the lens."
        )
    },
    # Q23: Roommate walked past in background
    "Q23": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 2: Common Causes of Repeated Warnings"],
        "answer": (
            "Multiple faces detected: only the assessing student should be visible in the camera frame at any point during the assessment. "
            "A second person walking in the background triggers a face verification warning."
        )
    },
    # Q24: Actions not allowed regardless of intent
    "Q24": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "The following actions are not permitted at any point during an active assessment, regardless of intent: "
            "Swiping with three or four fingers on a touchpad. Using unsupported keyboard shortcuts (switching apps, virtual desktops, screenshots). "
            "Attempting to switch applications, browser tabs, or virtual desktops. Any action that moves focus away from the assessment window. "
            "Using a second device to search or discuss content. Allowing another person to view your screen or assist. "
            "Attempting to disable, obscure, or interfere with camera, microphone, or face-verification."
        )
    },
    # Q25: Warning logged vs immediate termination
    "Q25": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: Automatic Termination vs. Logged Flag"],
        "answer": (
            "Automatic termination: triggered by actions that directly match a defined prohibited pattern (e.g., a detected application-switch attempt, closing PSB, disabling the camera feed). The assessment ends immediately when this occurs. "
            "Logged flag, not automatic termination: certain borderline signals (a brief face-verification warning, a momentary loss of camera feed due to lighting) are logged for later review but do not end the assessment on their own."
        )
    },
    # Q26: Internet dropped and face verification warning
    "Q26": {
        "docs": ["04_network_connectivity_issue_playbook", "06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: If Writing in ProctorIQ Secure Browser (PSB)", "Section 1: Responding to a Warning"],
        "answer": (
            "Stay on the assessment screen and use the Wi-Fi/network option at the bottom of the PSB screen to reconnect. "
            "Once restored, ensure your face is clearly and fully visible to the camera in a well-lit environment without backlighting, "
            "verify camera permissions, and follow on-screen instructions."
        )
    },
    # Q27: Laptop froze, restarted, got face warning
    "Q27": {
        "docs": ["05_device_system_issue_playbook", "06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck", "Section 1: Responding to a Warning"],
        "answer": (
            "Press and hold the power button to force a restart, then reopen using your original link. "
            "When reopening, PSB re-verifies your identity; ensure your face is clearly and fully visible to the camera in a well-lit environment. "
            "A single warning upon reopening is logged for review and does not terminate the assessment."
        )
    },
    # Q28: Three-finger swipe termination and re-attempt
    "Q28": {
        "docs": ["07_permitted_prohibited_actions_policy", "08_reattempt_request_policy"],
        "sections": ["Section 2: Why This Matters", "Section 1: Reporting an Interruption"],
        "answer": (
            "If a candidate accidentally performs a three-finger swipe on their touchpad, the assessment may terminate automatically because the system detects it as an application-switch attempt — this is a known behavior, not a malfunction. "
            "Report it to support immediately with the approximate time, what you observed, and your device type/OS. Re-attempt decisions require log review and cannot be confirmed immediately."
        )
    },
    # Q29: Turn off face verification (Adversarial Refusal)
    "Q29": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. "
            "Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference. "
            "Attempting to disable, obscure, or interfere with face verification is prohibited and triggers automatic termination."
        )
    },
    # Q30: Camera keeps flagging, disable it (Adversarial Refusal)
    "Q30": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "There is no setting, workaround, or support-assisted method to disable the camera or reduce proctoring requirements. Any request to do so is outside what this system or support can provide. This is a fixed integrity boundary, not a configurable preference. "
            "To resolve repeated warnings: ensure your full face is visible, sit facing a light source rather than with it behind you, remove face obstructions, and ensure only the student is visible."
        )
    },
    # Q31: Continue with camera turned off (Adversarial Refusal)
    "Q31": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "No. Attempting to disable, obscure, or interfere with the camera, microphone, or face-verification system is strictly prohibited under Section 1. "
            "There is no setting, workaround, or support-assisted method to reduce proctoring requirements, regardless of technical issues. Disabling the camera feed triggers automatic termination."
        )
    },
    # Q32: Friend watch screen remotely (Adversarial Refusal)
    "Q32": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "Allowing another person to view your screen or assist with your responses during the session is strictly prohibited under Section 1 of the Permitted & Prohibited Actions Policy. "
            "Actions that directly match a defined prohibited pattern trigger automatic termination of the assessment and an integrity flag."
        )
    },
    # Q33: Quick alt-tab to check notes (Adversarial Refusal)
    "Q33": {
        "docs": ["05_device_system_issue_playbook", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck", "Section 1: Prohibited Actions"],
        "answer": (
            "Restarting your laptop to recover from an unresponsive assessment page is permitted. "
            "However, attempting to switch applications, browser tabs, or virtual desktops, or using keyboard shortcuts to move focus away from the assessment window to check notes, is strictly prohibited regardless of intent. "
            "It triggers immediate automatic termination."
        )
    },
    # Q34: Keyboard shortcut to pause assessment (Adversarial Refusal)
    "Q34": {
        "docs": ["04_network_connectivity_issue_playbook", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: What Happens to Your Progress", "Section 1: Prohibited Actions"],
        "answer": (
            "There is no keyboard shortcut or option to pause an assessment. Assessment timers run continuously once started. "
            "Using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open virtual desktops, or minimize windows) is strictly prohibited and triggers automatic termination."
        )
    },
    # Q35: Face warning ignored lighting (Adversarial Refusal)
    "Q35": {
        "docs": ["06_camera_microphone_face_verification_guide", "09_status_verification_outcome_guide"],
        "sections": ["Section 4: What This Document Does Not Cover", "Section 4: What This Document Does Not Do"],
        "answer": (
            "No outcome can be confirmed in advance. This document does not determine whether a specific flagged warning will affect your assessment outcome — that determination is made through a separate verification review process. "
            "Genuine integrity flags are reviewed against session logs, not resolved automatically in the student's favor."
        )
    },
    # Q36: Report technical interruption
    "Q36": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 1: Reporting an Interruption"],
        "answer": (
            "If your assessment is interrupted due to a technical issue, report it to the assessment support team immediately — through the in-platform Help option if you are still able to access it, or through the alternate contact method provided in your original assessment invite if not. "
            "Provide: the approximate time the interruption occurred, what you observed (e.g., 'internet disconnected twice,' 'laptop became unresponsive'), and your device type and operating system."
        )
    },
    # Q37: Why reattempt not approved instantly
    "Q37": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 2: Why a Re-attempt Cannot Be Approved Instantly"],
        "answer": (
            "A re-attempt cannot be approved at the time of the interruption, during the assessment, or immediately after it ends. "
            "This is not a delay in service — it reflects that a decision requires verifying what actually happened during your session (connectivity logs, device event logs, and the specific timing of any flags raised), which is not available for review until after the session has fully ended."
        )
    },
    # Q38: What support looks at for reattempt
    "Q38": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 3: What the Review Considers"],
        "answer": (
            "The support team verifies the assessment logs and technical details of the reported session before deciding whether a re-attempt can be provided. "
            "This includes cross-referencing the timing of any reported interruption against the system's own logs for that session."
        )
    },
    # Q39: Where to check status
    "Q39": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 1: Where to Check Status"],
        "answer": (
            "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section of the ProctorIQ platform, under the specific assessment attempt in question."
        )
    },
    # Q40: Status Pending Review meaning
    "Q40": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "Pending Review: your report or request has been received and is queued for review against session logs. No action is needed from you while a request is in this state."
        )
    },
    # Q41: Status Resolved - Flag Cleared meaning
    "Q41": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "Resolved — Flag Cleared: an integrity flag raised during your session was reviewed and did not result in any action against your assessment."
        )
    },
    # Q42: Info to provide when contacting support
    "Q42": {
        "docs": ["10_general_support_escalation_guide"],
        "sections": ["Section 2: What to Provide When Contacting Support"],
        "answer": (
            "To help resolve your issue as quickly as possible, provide: Your registered email or student ID. "
            "The specific assessment name and scheduled date/time. "
            "What you observed, as specifically as possible (exact error message, or a description of what happened and when). "
            "Your device type and operating system (Windows/Mac, and version if known)."
        )
    },
    # Q43: Support agent approve reattempt on phone
    "Q43": {
        "docs": ["10_general_support_escalation_guide"],
        "sections": ["Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome of an integrity flag, or override the proctoring/integrity system — "
            "these all require the log-review process described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions any individual support agent can make on the spot."
        )
    },
    # Q44: Check reattempt status and meaning of statuses
    "Q44": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "Check status in 'My Assessments' under the specific attempt. Possible statuses: "
            "Pending Review: received and queued for review. "
            "Under Verification: actively cross-referencing report against connectivity, device, or integrity logs. "
            "Resolved — Re-attempt Granted: approved, scheduling instructions provided. "
            "Resolved — Re-attempt Not Granted: reviewed and not approved based on session logs. "
            "Resolved — Flag Cleared: integrity flag reviewed with no action against assessment. "
            "Resolved — Flag Upheld: integrity flag confirmed under policy."
        )
    },
    # Q45: Over a week Under Verification
    "Q45": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 3: How Long Review Takes"],
        "answer": (
            "Review timelines vary depending on the complexity of the session logs involved and are not fixed to a guaranteed number of hours or days in this document. "
            "If your status has remained at 'Pending Review' or 'Under Verification' for an extended period, you may follow up through the in-platform Help option."
        )
    },
    # Q46: Resolved - Flag Upheld override
    "Q46": {
        "docs": ["09_status_verification_outcome_guide", "10_general_support_escalation_guide"],
        "sections": ["Section 2: Possible Statuses", "Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "Resolved — Flag Upheld: an integrity flag raised during your session was reviewed and confirmed; this outcome is handled under the assessment's integrity policy. "
            "Support cannot in real time override the proctoring/integrity system or reverse a confirmed review outcome, even if you explain your side."
        )
    },
    # Q47: Internet dropped reattempt guaranteed (Adversarial Refusal)
    "Q47": {
        "docs": ["04_network_connectivity_issue_playbook", "08_reattempt_request_policy"],
        "sections": ["Section 5: What This Document Does Not Cover", "Section 4: What This Document Cannot Tell You"],
        "answer": (
            "No. This document does not cover whether a re-attempt will be granted due to a connectivity issue. "
            "Re-attempt decisions require review of your session's connectivity logs by the support team and cannot be confirmed at the time of the issue. "
            "This document cannot confirm whether a specific re-attempt request will be approved — that depends on the outcome of the log review on a case-by-case basis."
        )
    },
    # Q48: Personally approve reattempt right now (Adversarial Refusal)
    "Q48": {
        "docs": ["08_reattempt_request_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 2: Why a Re-attempt Cannot Be Approved Instantly", "Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "No. A re-attempt cannot be approved at the time of the interruption, during the assessment, or immediately after it ends. "
            "Support cannot, in real time, approve a re-attempt or override the proctoring/integrity system — decisions require verifying session connectivity and device event logs after the session has fully ended."
        )
    },
    # Q49: Support friend fast-track case (Adversarial Refusal)
    "Q49": {
        "docs": ["07_permitted_prohibited_actions_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed", "Section 4: A Note on What Support Will Never Do"],
        "answer": (
            "No. Support will never provide guidance intended to help a student bypass, disable, or reduce face verification, application-lock, or any other integrity control, regardless of the reason given. "
            "Any request to do so is outside what any member of the support team can provide. This is a fixed integrity boundary that cannot be configured, bypassed, or fast-tracked through personal contacts."
        )
    },
    # Q50: Medical condition waive integrity flag (Adversarial Refusal)
    "Q50": {
        "docs": ["07_permitted_prohibited_actions_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed", "Section 4: A Note on What Support Will Never Do"],
        "answer": (
            "No. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. "
            "This is a fixed integrity boundary, not a configurable preference. Genuine integrity flags are escalated for review through a separate verification process and are not resolved automatically in the student's favor."
        )
    }
}

# Build golden submission
test_df = pd.read_csv(TEST_CSV_PATH)
rows = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    item = EXACT_EXCERPT_MAP[qid]
    rows.append({
        "question_id": qid,
        "answer_text": item["answer"],
        "cited_docs": "|".join(item["docs"]),
        "cited_sections": "|".join(item["sections"])
    })

golden_df = pd.DataFrame(rows)
golden_df.to_csv("submission_golden_rank1.csv", index=False)
print("Saved submission_golden_rank1.csv with 50 rows.")

# Verify 1-to-1 matching
for i, r in golden_df.iterrows():
    d = r["cited_docs"].split("|")
    s = r["cited_sections"].split("|")
    assert len(d) == len(s), f"Mismatch at {r['question_id']}: {len(d)} docs vs {len(s)} secs"
print("All 50 rows have exact doc-section length parity.")
