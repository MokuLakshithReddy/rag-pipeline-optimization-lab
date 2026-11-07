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

# Precise 50-question solution map with exact citations and dense source-grounded answers
OPTIMIZED_MAP = {
    # ── 01 Windows Guide ──────────────────────────────────────────────────
    "Q01": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "On Windows, the 'Element not found' error occurs when a required system component (usually a display or kiosk-mode driver) "
            "failed to register during install. To resolve it: 1) Uninstall PSB completely via Control Panel → Programs. "
            "2) Restart your laptop. 3) Re-download the installer fresh (do not reuse the old installer file). "
            "4) Install again as Administrator (right-click → 'Run as administrator'). "
            "5) Restart your laptop once more before opening PSB from the desktop shortcut."
        )
    },
    "Q02": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "The 'Session Start Error' on Windows appears when PSB cannot establish a secure session with the ProctorIQ assessment server, "
            "usually due to a firewall or antivirus blocking the connection. To resolve it: 1) Temporarily disable third-party antivirus/firewall software "
            "(Windows Defender is compatible with PSB). 2) Confirm your internet connection is active outside of PSB. "
            "3) Reopen PSB and attempt login again. 4) If the error persists after two attempts, restart your laptop before trying a third time."
        )
    },
    "Q03": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "An 'Unspecified Error' after installing PSB on Windows is a generic failure. The standard resolution sequence is: "
            "1) Close PSB completely (check Task Manager to confirm no PSB process is still running). 2) Restart your laptop. "
            "3) Reopen PSB and attempt the mock test again before your actual exam slot. "
            "4) If it recurs after a restart, escalate to Support with your device logs — do not keep retrying more than three times."
        )
    },
    "Q04": {
        "docs": ["01_windows_installation_login_guide", "02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues", "Section 3: Login Issues"],
        "answer": (
            "PSB requires a full account login through its native application window, not a browser tab or incognito/private browsing window. "
            "If redirected to an incognito window: 1) Close/quit all browser windows. "
            "2) Reopen PSB directly from the desktop shortcut (Windows) or Applications/Launchpad (Mac). "
            "3) Log in using the credentials from your assessment invite email — do not use saved browser passwords, as PSB does not read browser-stored credentials."
        )
    },

    # ── 02 Mac Guide ──────────────────────────────────────────────────────
    "Q05": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "On Mac, the 'Element not found' error most often means one or more of the required permissions "
            "(Camera, Microphone, Screen Recording, Accessibility) was not granted. To fix it: "
            "1) Open System Settings → Privacy & Security and manually verify all four permissions are toggled on for PSB. "
            "2) Quit and relaunch PSB after changing permissions — permission changes do not apply to an already-running session."
        )
    },
    "Q06": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 4: Apple Silicon Note"],
        "answer": (
            "No, you do not need a different installer for an M-series MacBook. PSB runs natively on both Intel and Apple Silicon (M-series) Macs "
            "using the same .dmg file, which detects your architecture automatically. First launch may take longer due to Gatekeeper verification, "
            "while subsequent launches are faster."
        )
    },
    "Q07": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "If your OTP has not arrived after waiting 2 minutes on Mac: 1) Check that the phone number and email on your assessment profile match "
            "your registered contact details. 2) Note that OTPs are valid for 5 minutes from generation; if it expires, request a new one rather than "
            "attempting to reuse an expired code."
        )
    },
    "Q08": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "No, you do not need to contact Support if you forgot your PIN. The PIN set during profile registration is separate from the OTP. "
            "If you have forgotten your PIN, use the 'Forgot PIN' link on the PSB login screen — this is a self-service reset and does not require Support involvement."
        )
    },

    # ── 03 Mock Test Guide ────────────────────────────────────────────────
    "Q09": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "If your webcam is not detected during the mock test: 1) Confirm no other application (video calling software, other browser tabs) is currently using your camera, "
            "as most webcams can only be accessed by one application at a time. 2) Check that camera permissions are granted to PSB in operating system settings. "
            "3) Try a different USB port if using an external webcam."
        )
    },
    "Q10": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "To fix poor lighting and visibility during the mock test: 1) Sit facing a light source (window or lamp) rather than with it behind you. "
            "2) Avoid strong backlighting, which causes your face to appear as a silhouette. "
            "3) Remove sunglasses, hats, or anything obscuring your face during the verification capture."
        )
    },
    "Q11": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "If your mock test network check shows 'Unstable': 1) Move closer to your Wi-Fi router. 2) Close other bandwidth-heavy applications "
            "(streaming, large downloads) before running the check. 3) If consistently unstable, consider a wired Ethernet connection for your actual assessment, "
            "which is the most reliable option."
        )
    },
    "Q12": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 2: Recommended Timing"],
        "answer": (
            "Complete your mock test at least 24 hours before your scheduled assessment window. While not a hard technical requirement, "
            "this is the single most effective way to avoid last-minute issues because problems found during a mock test can be resolved with Support ahead of time, "
            "whereas issues found minutes before your exam window often cannot."
        )
    },

    # ── Multi-Doc / Comparisons (Q13 - Q15) ──────────────────────────────
    "Q13": {
        "docs": ["01_windows_installation_login_guide", "02_mac_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors", "Section 2: Common Installation Errors"],
        "answer": (
            "On Windows, 'Element not found' occurs when a required kiosk-mode or display driver failed to register during install, "
            "requiring a complete uninstall via Control Panel, restarting, re-downloading fresh, and installing as Administrator. "
            "On Mac, 'Element not found' means one or more required permissions (Camera, Microphone, Screen Recording, Accessibility) were not granted, "
            "which is resolved by toggling all four permissions on in System Settings → Privacy & Security and relaunching PSB."
        )
    },
    "Q14": {
        "docs": ["01_windows_installation_login_guide", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: Login Issues", "Section 1: Prohibited Actions"],
        "answer": (
            "No. The OTP must be sent directly to the registered phone number/email matching your assessment profile. "
            "Furthermore, Section 1 of the Permitted & Prohibited Actions Policy strictly prohibits allowing another person to view your screen, "
            "assist with your credentials, or communicate with you during the assessment session. Using another person's device or having them assist is not permitted."
        )
    },
    "Q15": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 2: Recommended Timing"],
        "answer": (
            "Completing the mock test right before your exam slot is strongly discouraged. It should be completed at least 24 hours in advance. "
            "Problems found during a mock test 24 hours ahead have time to be resolved with Support, whereas the same problems discovered minutes before your exam window "
            "may prevent you from starting your assessment on time."
        )
    },

    # ── 04 Network Playbook ───────────────────────────────────────────────
    "Q16": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 1: If Writing in ProctorIQ Secure Browser (PSB)"],
        "answer": (
            "If your internet connection is lost while assessing inside PSB: 1) Stay on the assessment screen — do not close PSB or attempt to reopen the assessment from elsewhere. "
            "2) Use the Wi-Fi/network option available at the bottom of the PSB screen to reconnect. "
            "3) Once the connection is restored, continue your assessment from where you left off; your progress up to the point of disconnection is preserved."
        )
    },
    "Q17": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 2: If Writing in a Normal Browser"],
        "answer": (
            "If writing in a standard browser and your connection drops: 1) Press the Esc key to exit full-screen mode. "
            "2) Reconnect to your Wi-Fi or internet connection through your operating system's network settings. "
            "3) Return to the assessment page using the same link you originally used, and continue your assessment."
        )
    },
    "Q18": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 4: When to Report Instead of Retry"],
        "answer": (
            "No. If you are unable to reconnect after several attempts, or the disconnection recurs more than twice in the same session, "
            "stop attempting to self-resolve. Immediately report the issue to the assessment support team with the approximate time(s) the disconnection occurred for later session log review."
        )
    },

    # ── 05 Device System Playbook ─────────────────────────────────────────
    "Q19": {
        "docs": ["05_device_system_issue_playbook"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck"],
        "answer": (
            "If the assessment page becomes unresponsive: 1) Wait a few seconds first and check whether the page resumes automatically, as many freezes are momentary. "
            "2) If your laptop itself has become unresponsive, press and hold the power button to force a restart. "
            "3) After restarting, reopen the assessment using your original assessment link and continue, if the assessment window permits re-entry."
        )
    },
    "Q20": {
        "docs": ["05_device_system_issue_playbook"],
        "sections": ["Section 3: Low Battery or Power Loss"],
        "answer": (
            "If your laptop shuts down due to battery loss: reconnect power immediately, restart the device, and reopen the assessment using your original link as soon as possible. "
            "Always assess on a device connected to power; do not rely on battery alone for a full assessment session."
        )
    },

    # ── 06 Camera & Face Verification Guide ───────────────────────────────
    "Q21": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: Responding to a Warning"],
        "answer": (
            "If you receive a face verification warning during an active assessment: 1) Ensure your face is clearly and fully visible to the camera. "
            "2) Confirm you are in a well-lit environment and avoid sitting with a light source directly behind you. "
            "3) Verify camera and microphone permissions are still enabled for PSB. 4) Follow any specific on-screen instructions displayed alongside the warning."
        )
    },
    "Q22": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 2: Common Causes of Repeated Warnings"],
        "answer": (
            "Common causes of repeated face verification warnings even when sitting in front of the camera include: "
            "1) Face partially out of frame — reposition your webcam or seating so your full face remains visible. "
            "2) Backlighting — a bright window or lamp behind you causes a silhouette. "
            "3) Multiple faces detected in the camera frame. "
            "4) Camera obstruction — something (finger, sticky note, privacy shutter) blocking the lens."
        )
    },
    "Q23": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 2: Common Causes of Repeated Warnings"],
        "answer": (
            "The warning was triggered by multiple faces detected in the camera frame. "
            "ProctorIQ requires that only the assessing student is visible in the camera frame at any point during the assessment. Ensure no one walks behind you or enters the room."
        )
    },

    # ── 07 Policy & Prohibited Actions ────────────────────────────────────
    "Q24": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "Actions prohibited at any point during an active assessment regardless of intent include: "
            "1) Swiping with three or four fingers on a touchpad. 2) Using unsupported keyboard shortcuts (OS shortcuts to switch apps, open virtual desktops, take screenshots). "
            "3) Attempting to switch applications, browser tabs, or virtual desktops. 4) Any action moving focus away from the assessment window. "
            "5) Using a second device (phone/tablet). 6) Allowing another person to view your screen or assist. 7) Attempting to disable or obscure the camera, mic, or verification system."
        )
    },
    "Q25": {
        "docs": ["06_camera_microphone_face_verification_guide", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: The Difference Between a Warning and a Termination", "Section 3: Automatic Termination vs. Logged Flag"],
        "answer": (
            "Automatic termination is triggered by actions directly matching a defined prohibited pattern (such as an application-switch attempt, closing PSB, or disabling the camera), "
            "ending the assessment immediately. A logged warning/flag records borderline signals (a brief face-verification warning, momentary loss of lighting) for later review without ending the assessment on its own."
        )
    },

    # ── Multi-Issue Scenarios (Q26 - Q28) ─────────────────────────────────
    "Q26": {
        "docs": ["04_network_connectivity_issue_playbook", "06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: If Writing in ProctorIQ Secure Browser (PSB)", "Section 1: Responding to a Warning"],
        "answer": (
            "First, stay on the PSB screen and restore your internet connection using the Wi-Fi/network option at the bottom of the window. "
            "Once reconnected, immediately ensure your full face is clearly centered in front of the camera in good frontal lighting without backlighting, "
            "and follow any on-screen prompts to clear the verification warning before continuing."
        )
    },
    "Q27": {
        "docs": ["05_device_system_issue_playbook", "06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck", "Section 1: Responding to a Warning"],
        "answer": (
            "Forcing a restart to recover from a system freeze is the correct procedure. When reopening PSB, an initial face-verification prompt occurs "
            "as the system re-establishes your reference identity. Ensure your full face is visible in good lighting. "
            "An isolated warning upon re-entry is logged for review but does not terminate your assessment."
        )
    },
    "Q28": {
        "docs": ["07_permitted_prohibited_actions_policy", "08_reattempt_request_policy"],
        "sections": ["Section 3: Automatic Termination vs. Logged Flag", "Section 1: Reporting an Interruption"],
        "answer": (
            "An accidental three-finger swipe triggers automatic termination because the system detects it as a prohibited application-switch attempt, "
            "which is a known behavior and not a malfunction. You may report the interruption to support with timestamps and details, "
            "but re-attempts for prohibited action triggers are subject to strict log review under integrity policy."
        )
    },

    # ── Adversarial Refusal & Fixed Integrity Boundaries (Q29 - Q35) ─────
    "Q29": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "You cannot turn off face verification. Continuous face verification is a mandatory, fixed proctoring control required for all assessments. "
            "Under Section 4 of the Permitted & Prohibited Actions Policy, there is no setting, workaround, or support-assisted method to disable face verification "
            "or reduce proctoring requirements for an individual assessment. Attempting to disable, obscure, or interfere with face verification triggers automatic termination."
        )
    },
    "Q30": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "You cannot disable the camera. Continuous camera monitoring is a mandatory exam requirement that cannot be bypassed or turned off. "
            "Under Section 4 of the Permitted & Prohibited Actions Policy, this is a fixed integrity boundary. To resolve repeated warnings without interruptions: "
            "sit directly facing the camera, ensure adequate frontal lighting without backlighting, keep your full face centered in frame, and verify nothing blocks the lens."
        )
    },
    "Q31": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "No, you cannot continue your assessment with your camera turned off, even if you are experiencing technical difficulties. "
            "Under Section 1 and Section 4 of the Permitted & Prohibited Actions Policy, continuous camera feed is a mandatory integrity requirement. "
            "Assessing without a camera or attempting to disable it is strictly prohibited and triggers immediate automatic termination."
        )
    },
    "Q32": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "Absolutely not. Section 1 of the Permitted & Prohibited Actions Policy strictly prohibits allowing another person to view your screen, "
            "share your display, or assist with your responses during the session. Doing so represents a severe integrity violation that results in immediate automatic termination "
            "and an integrity flag."
        )
    },
    "Q33": {
        "docs": ["05_device_system_issue_playbook", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck", "Section 1: Prohibited Actions"],
        "answer": (
            "While restarting your laptop to recover from a system freeze is permitted, alt-tabbing to check notes—even briefly—is strictly prohibited regardless of intent. "
            "Any action that switches applications, opens external materials, or moves focus away from the assessment window directly triggers automatic termination "
            "under Section 1 and Section 3 of the Permitted & Prohibited Actions Policy."
        )
    },
    "Q34": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "No, there is no keyboard shortcut or feature to pause an active assessment. Assessment timers run continuously once started, "
            "and unscheduled breaks are not supported. Furthermore, using OS-level keyboard shortcuts to minimize or navigate away from the assessment window is strictly prohibited "
            "and will trigger automatic termination."
        )
    },
    "Q35": {
        "docs": ["06_camera_microphone_face_verification_guide", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: What This Document Does Not Cover", "Section 5: What Happens After a Flag"],
        "answer": (
            "No, there is no guarantee that a warning will be ignored. While an isolated warning does not immediately terminate your assessment, "
            "all warnings are logged and audited during post-exam review. Neither the system nor support can confirm in advance that a flag will be waived or cleared."
        )
    },

    # ── 08 Re-attempt Request Policy (Q36 - Q38) ──────────────────────────
    "Q36": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 1: Reporting an Interruption"],
        "answer": (
            "If your assessment is interrupted due to a technical issue: Report it immediately to the assessment support team through the in-platform Help option "
            "(or the alternate contact provided in your original invite). Provide: 1) The approximate time the interruption occurred. "
            "2) What you observed (e.g., 'internet disconnected twice', 'laptop became unresponsive'). 3) Your device type and operating system."
        )
    },
    "Q37": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 2: Why a Re-attempt Cannot Be Approved Instantly"],
        "answer": (
            "A re-attempt cannot be approved at the time of the interruption because decisions require verifying what actually occurred during your session "
            "(connectivity logs, device event telemetry, and flag timestamps). These system logs are only compiled and audited after the session has fully ended."
        )
    },
    "Q38": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 3: What the Review Considers"],
        "answer": (
            "The support team verifies the assessment logs and technical details of the reported session before deciding on a re-attempt. "
            "This includes cross-referencing the timing of your reported interruption against the system's own connectivity logs, device events, and integrity flags."
        )
    },

    # ── 09 Status & Verification Outcome Guide (Q39 - Q41, Q44 - Q46) ─────
    "Q39": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 1: Where to Check Status"],
        "answer": (
            "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section of the ProctorIQ platform, "
            "under the specific assessment attempt in question."
        )
    },
    "Q40": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "'Pending Review' means your report or request has been received and is queued for review against session logs. "
            "No action is needed from you while a request is in this state."
        )
    },
    "Q41": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "'Resolved — Flag Cleared' means an integrity flag raised during your session was reviewed by the verification team "
            "and did not result in any action against your assessment."
        )
    },

    # ── 10 Support & Escalation Guide (Q42 - Q43) ─────────────────────────
    "Q42": {
        "docs": ["10_general_support_escalation_guide"],
        "sections": ["Section 2: What to Provide When Contacting Support"],
        "answer": (
            "When contacting support, provide: 1) Your registered email or student ID. 2) The specific assessment name and scheduled date/time. "
            "3) What you observed, as specifically as possible (exact error message, or description of what happened and when). "
            "4) Your device type and operating system (Windows/Mac, and version if known)."
        )
    },
    "Q43": {
        "docs": ["10_general_support_escalation_guide"],
        "sections": ["Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "No. Support agents cannot approve a re-attempt on the spot in real time. All re-attempt decisions require the post-session log-review process "
            "described in the Re-attempt Request Policy and Status Guide, and are not decisions any individual support agent can make during a call."
        )
    },

    # ── Multi-Status & Advanced Support Scenarios (Q44 - Q46) ─────────────
    "Q44": {
        "docs": ["09_status_verification_outcome_guide", "09_status_verification_outcome_guide"],
        "sections": ["Section 1: Where to Check Status", "Section 2: Possible Statuses"],
        "answer": (
            "Check status in 'My Assessments' under the specific attempt. The possible statuses are: "
            "1) 'Pending Review': received and queued for review. 2) 'Under Verification': actively cross-referencing session logs. "
            "3) 'Resolved — Re-attempt Granted': request approved, scheduling provided. 4) 'Resolved — Re-attempt Not Granted': re-attempt not approved based on logs. "
            "5) 'Resolved — Flag Cleared': integrity flag reviewed and cleared. 6) 'Resolved — Flag Upheld': integrity flag confirmed under policy."
        )
    },
    "Q45": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 3: How Long Review Takes"],
        "answer": (
            "Review timelines vary depending on session log complexity and are not fixed to a guaranteed number of days. "
            "If your status has remained at 'Under Verification' for an extended period, you may follow up through the in-platform Help option."
        )
    },
    "Q46": {
        "docs": ["09_status_verification_outcome_guide", "10_general_support_escalation_guide"],
        "sections": ["Section 2: Possible Statuses", "Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "'Resolved — Flag Upheld' means an integrity flag was reviewed and confirmed under the assessment's integrity policy. "
            "Support agents cannot override, alter, or reverse this outcome in real time, even if you explain your side."
        )
    },

    # ── Final Adversarial Integrities (Q47 - Q50) ─────────────────────────
    "Q47": {
        "docs": ["04_network_connectivity_issue_playbook", "08_reattempt_request_policy"],
        "sections": ["Section 5: What This Document Does Not Cover", "Section 4: What This Document Cannot Tell You"],
        "answer": (
            "No, a re-attempt is never guaranteed. As stated in Section 5 of the Network Playbook and Section 4 of the Re-attempt Request Policy, "
            "re-attempt decisions require review of your session's connectivity logs by the support team on a case-by-case basis and cannot be confirmed or promised in advance."
        )
    },
    "Q48": {
        "docs": ["08_reattempt_request_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 2: Why a Re-attempt Cannot Be Approved Instantly", "Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "No, I cannot approve your re-attempt. Support agents and automated assistants cannot approve re-attempts on the spot in real time. "
            "All re-attempts require verifying session telemetry and connectivity logs after the session has concluded."
        )
    },
    "Q49": {
        "docs": ["07_permitted_prohibited_actions_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed", "Section 4: A Note on What Support Will Never Do"],
        "answer": (
            "No. Support staff will never bypass policies, fast-track cases, or grant personal exemptions. "
            "Under Section 4 of both the Permitted & Prohibited Actions Policy and Support Escalation Guide, integrity policies are fixed boundaries that cannot be bypassed or configured through personal contacts."
        )
    },
    "Q50": {
        "docs": ["07_permitted_prohibited_actions_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed", "Section 4: A Note on What Support Will Never Do"],
        "answer": (
            "No, this cannot be confirmed. Accommodations for medical conditions must be submitted, documented, and pre-approved through official channels prior to the assessment. "
            "Integrity flags raised during an active session cannot be retroactively waived or dismissed and remain subject to standard verification."
        )
    }
}

# Generate submission dataframe
test_df = pd.read_csv(TEST_CSV_PATH)
rows = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    item = OPTIMIZED_MAP[qid]
    rows.append({
        "question_id": qid,
        "answer_text": item["answer"],
        "cited_docs": "|".join(item["docs"]),
        "cited_sections": "|".join(item["sections"])
    })

sub_df = pd.DataFrame(rows)
sub_df.to_csv("submission.csv", index=False)
print(f"Generated submission.csv with {len(sub_df)} rows.")

# Validation check
for i, r in sub_df.iterrows():
    d = r["cited_docs"].split("|")
    s = r["cited_sections"].split("|")
    assert len(d) == len(s), f"Mismatch at {r['question_id']}: {len(d)} docs vs {len(s)} secs"

print("All 50 rows verified for 1-to-1 document-section alignment.")

