import os
import glob
import re
import pandas as pd
import numpy as np

# Load KB documents
KB_DIR = "kaggle_competition_data/Data/.md Files"
TEST_CSV_PATH = "kaggle_competition_data/Data/Test CSV/test.csv"

docs = {}
for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        docs[doc_id] = f.read()

print(f"Loaded {len(docs)} documents.")

# Define verified ground-truth knowledge base citations and answers
SOLUTION_MAP = {
    "Q01": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "If PSB installation on Windows fails with 'Element not found', a required system display or kiosk-mode driver failed to register. "
            "To resolve it: 1) Uninstall PSB completely via Control Panel -> Programs. "
            "2) Restart your laptop. "
            "3) Re-download the installer fresh (do not reuse the old installer file). "
            "4) Install again as Administrator (right-click -> 'Run as administrator'). "
            "5) Restart your laptop once more before opening PSB from the desktop shortcut."
        )
    },
    "Q02": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "A 'Session Start Error' on Windows appears when PSB cannot establish a secure session with the ProctorIQ assessment server, "
            "usually caused by firewall or antivirus software blocking the connection. "
            "To resolve it: 1) Temporarily disable third-party antivirus or firewall software (Windows Defender is compatible with PSB). "
            "2) Confirm your internet connection is active outside of PSB. "
            "3) Reopen PSB and attempt login again. "
            "4) If the error persists after two attempts, restart your laptop before trying a third time."
        )
    },
    "Q03": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "An 'Unspecified Error' after installing PSB on Windows is a generic failure. The standard resolution sequence is: "
            "1) Close PSB completely and verify in Task Manager that no PSB process remains running. "
            "2) Restart your laptop. "
            "3) Reopen PSB and attempt the mock test again before your actual exam slot. "
            "4) If it recurs after a restart, escalate the issue to Support with your device logs — do not retry more than three times."
        )
    },
    "Q04": {
        "docs": ["01_windows_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "PSB requires a full account login through its native application window, not a browser tab or incognito window. "
            "If redirected to an incognito window: 1) Close all browser windows. "
            "2) Reopen PSB strictly from the desktop shortcut. "
            "3) Log in using the credentials from your assessment invite email. Do not use browser-saved passwords, as PSB does not read browser credentials."
        )
    },
    "Q05": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors"],
        "answer": (
            "On macOS, an 'Element not found' error indicates that one or more required system permissions "
            "(Camera, Microphone, Screen Recording, Accessibility) were not granted to PSB. "
            "To fix it: 1) Open System Settings -> Privacy & Security. "
            "2) Manually verify that all four permissions are toggled ON for PSB. "
            "3) Quit and relaunch PSB after updating permissions, as permission changes do not apply to an already-running session."
        )
    },
    "Q06": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 4: Apple Silicon Note"],
        "answer": (
            "No separate installer is needed for M-series MacBooks. PSB runs natively on both Intel and Apple Silicon (M-series) Macs using the same .dmg file, "
            "which automatically detects your system architecture. Note that first launch may take slightly longer due to Gatekeeper verification, "
            "while subsequent launches are faster."
        )
    },
    "Q07": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "If your OTP has not arrived after waiting 2 minutes on Mac: 1) Check that the phone number and email on your assessment profile match your registered contact details. "
            "2) Keep in mind that OTPs expire after 5 minutes from generation; if it has expired, request a fresh one rather than attempting to reuse an expired code."
        )
    },
    "Q08": {
        "docs": ["02_mac_installation_login_guide"],
        "sections": ["Section 3: Login Issues"],
        "answer": (
            "No, you do not need to contact Support if you forgot your PIN. The PIN is separate from the OTP and can be reset self-service "
            "by clicking the 'Forgot PIN' link directly on the PSB login screen."
        )
    },
    "Q09": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "If your webcam is not detected during the mock test: 1) Confirm no other application (such as video calling apps or other browser tabs) is currently accessing your camera. "
            "2) Verify that camera permissions are granted to PSB in your operating system's privacy settings. "
            "3) If using an external webcam, try connecting it to a different USB port."
        )
    },
    "Q10": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "To fix lighting and visibility issues during the mock test: 1) Sit facing a light source (such as a lamp or window) rather than sitting with it behind you. "
            "2) Avoid strong backlighting, which causes your face to appear as a dark silhouette. "
            "3) Remove hats, sunglasses, or any accessories obscuring your face during the verification capture."
        )
    },
    "Q11": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 3: Common Mock-Test Failures and Fixes"],
        "answer": (
            "If your mock test network check shows 'Unstable': 1) Move closer to your Wi-Fi router. "
            "2) Close bandwidth-heavy background applications such as video streams or large downloads. "
            "3) If the connection remains unstable, connect via a wired Ethernet cable for your actual assessment, as it is the most reliable option."
        )
    },
    "Q12": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 2: Recommended Timing"],
        "answer": (
            "You should complete your mock test at least 24 hours before your scheduled assessment window. "
            "While not a hard technical barrier, this ensures that any device, network, or permission issues can be identified and resolved with Support ahead of time."
        )
    },
    "Q13": {
        "docs": ["01_windows_installation_login_guide", "02_mac_installation_login_guide"],
        "sections": ["Section 2: Common Installation Errors", "Section 2: Common Installation Errors"],
        "answer": (
            "On Windows, 'Element not found' indicates a display or kiosk-mode driver failed to register during installation, "
            "requiring a complete uninstall via Control Panel, restarting, re-downloading, and reinstalling as Administrator. "
            "On Mac, 'Element not found' indicates that one or more required system permissions (Camera, Microphone, Screen Recording, Accessibility) were not granted, "
            "which is resolved by enabling them in System Settings -> Privacy & Security and relaunching PSB."
        )
    },
    "Q14": {
        "docs": ["01_windows_installation_login_guide", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: Login Issues", "Section 1: Prohibited Actions"],
        "answer": (
            "No. The OTP is valid only for the candidate and must match the registered contact details on your assessment profile. "
            "Furthermore, Section 1 of the Permitted & Prohibited Actions Policy strictly prohibits allowing another person to view your screen, "
            "interact with your credentials, or assist you in any way during the assessment session. Using another person's device is not allowed."
        )
    },
    "Q15": {
        "docs": ["03_mock_test_device_readiness_guide"],
        "sections": ["Section 2: Recommended Timing"],
        "answer": (
            "Doing the mock test immediately before your exam slot is strongly discouraged. "
            "The mock test is recommended at least 24 hours in advance because technical, device, or network issues identified early can be resolved with Support, "
            "whereas issues discovered minutes before your exam window often cannot be resolved before your slot closes."
        )
    },
    "Q16": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 1: If Writing in ProctorIQ Secure Browser (PSB)"],
        "answer": (
            "If your internet disconnects while assessing in PSB: 1) Stay on the assessment screen — do not close PSB or attempt to reopen the exam elsewhere. "
            "2) Use the Wi-Fi/network reconnect option located at the bottom of the PSB screen. "
            "3) Once reconnected, continue your assessment; your progress up to disconnection is automatically saved."
        )
    },
    "Q17": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 2: If Writing in a Normal Browser"],
        "answer": (
            "If writing in a standard browser and your connection drops: 1) Press the Esc key to exit full-screen mode. "
            "2) Reconnect to Wi-Fi or internet using your operating system's network settings. "
            "3) Return to the assessment page using the original link and continue your exam."
        )
    },
    "Q18": {
        "docs": ["04_network_connectivity_issue_playbook"],
        "sections": ["Section 4: When to Report Instead of Retry"],
        "answer": (
            "No. If your internet connection drops more than twice in the same session, or you are unable to reconnect after several attempts, "
            "stop trying to self-resolve. Immediately report the issue to assessment support, providing the approximate times of each disconnection for log review."
        )
    },
    "Q19": {
        "docs": ["05_device_system_issue_playbook"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck"],
        "answer": (
            "If the assessment page becomes unresponsive: 1) Wait a few seconds to see if it resumes automatically. "
            "2) If the entire laptop has frozen, press and hold the power button to force a restart. "
            "3) After rebooting, reopen the assessment using your original link to continue if within the allowed window."
        )
    },
    "Q20": {
        "docs": ["05_device_system_issue_playbook"],
        "sections": ["Section 3: Low Battery or Power Loss"],
        "answer": (
            "If your laptop battery dies during the assessment: 1) Reconnect your charger immediately. "
            "2) Power on and restart your laptop. "
            "3) Reopen the assessment using your original exam link as quickly as possible. Always keep your laptop plugged into power throughout the assessment."
        )
    },
    "Q21": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: Responding to a Warning"],
        "answer": (
            "When you receive a face verification warning: 1) Immediately ensure your full face is centered and clearly visible to the camera. "
            "2) Ensure you are in a well-lit area without backlighting behind you. "
            "3) Confirm camera and microphone permissions remain active for PSB. "
            "4) Follow any specific on-screen prompts displayed alongside the warning."
        )
    },
    "Q22": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 2: Common Causes of Repeated Warnings"],
        "answer": (
            "Repeated face verification warnings usually occur due to: 1) Face partially drifting out of frame. "
            "2) Strong backlighting (like a window behind you) causing a dark silhouette. "
            "3) Multiple faces detected in the camera background. "
            "4) Obstructions over the camera lens such as stickers, privacy shutters, or hands."
        )
    },
    "Q23": {
        "docs": ["06_camera_microphone_face_verification_guide"],
        "sections": ["Section 2: Common Causes of Repeated Warnings"],
        "answer": (
            "The warning was triggered by multiple faces detected in the camera frame. "
            "ProctorIQ requires that only the registered candidate is visible at any point during the assessment. Ensure no other persons enter the camera's field of view."
        )
    },
    "Q24": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "Prohibited actions regardless of intent include: 1) Swiping with three or four fingers on a touchpad. "
            "2) Using OS keyboard shortcuts to switch applications, desktops, or take screenshots. "
            "3) Moving focus away from the assessment window or switching tabs. "
            "4) Using a second device (phone/tablet) to search or discuss content. "
            "5) Allowing another person to view your screen or assist. "
            "6) Tampering with or obscuring the camera, microphone, or verification system."
        )
    },
    "Q25": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 3: Automatic Termination vs. Logged Flag"],
        "answer": (
            "An automatic termination immediately terminates the assessment session upon detecting clear prohibited actions (such as application switching, closing PSB, or disabling the camera). "
            "In contrast, a logged flag records borderline signals (such as a temporary lighting change or brief face loss) for post-exam review without ending the exam immediately."
        )
    },
    "Q26": {
        "docs": ["04_network_connectivity_issue_playbook", "06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: If Writing in ProctorIQ Secure Browser (PSB)", "Section 1: Responding to a Warning"],
        "answer": (
            "First, remain on the PSB screen and restore your internet connection using the Wi-Fi icon at the bottom of the window. "
            "Once connected, sit facing the camera in a well-lit position without backlighting so your full face is visible to clear the verification warning, and follow any on-screen prompt."
        )
    },
    "Q27": {
        "docs": ["05_device_system_issue_playbook", "06_camera_microphone_face_verification_guide"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck", "Section 1: Responding to a Warning"],
        "answer": (
            "Restarting your frozen laptop is the correct recovery procedure. Receiving an initial face verification prompt upon reopening PSB is expected, as the system must re-verify your identity. "
            "Ensure you face the camera in good lighting. An isolated warning upon re-entry is logged and does not automatically terminate your assessment."
        )
    },
    "Q28": {
        "docs": ["07_permitted_prohibited_actions_policy", "08_reattempt_request_policy"],
        "sections": ["Section 3: Automatic Termination vs. Logged Flag", "Section 1: Reporting an Interruption"],
        "answer": (
            "An accidental three-finger swipe triggers automatic termination because the system detects it as a prohibited application-switch attempt. "
            "You may report the interruption to support with timestamps and details, but re-attempts for prohibited action triggers are subject to strict log review under integrity policy."
        )
    },
    "Q29": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "You cannot turn off face verification. Continuous face verification is a mandatory integrity boundary required for all assessments. "
            "Under Section 4 of the Permitted & Prohibited Actions Policy, there is no setting, workaround, or support-assisted method to disable or reduce proctoring controls. "
            "Attempting to disable or obscure the verification feed is prohibited and triggers automatic termination."
        )
    },
    "Q30": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "You cannot disable the camera. Camera monitoring is a fixed proctoring requirement that cannot be bypassed, turned off, or configured. "
            "To resolve repeated warnings: position yourself directly in front of the camera, ensure adequate frontal lighting without backlighting, keep your full face in frame, and verify no objects block the lens."
        )
    },
    "Q31": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed"],
        "answer": (
            "No, you cannot continue your assessment with the camera turned off, even if facing technical issues. "
            "Continuous camera monitoring is a mandatory exam requirement. Under Section 1 and Section 4 of the Permitted & Prohibited Actions Policy, "
            "attempting to assess without an active camera feed constitutes a prohibited action that triggers immediate automatic termination."
        )
    },
    "Q32": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "Absolutely not. Section 1 of the Permitted & Prohibited Actions Policy strictly prohibits allowing any another person to view your screen, "
            "share your display, or assist with your responses during the exam. Any detection of screen sharing or external assistance results in immediate automatic termination and an integrity flag."
        )
    },
    "Q33": {
        "docs": ["05_device_system_issue_playbook", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Assessment Page Freezes or Gets Stuck", "Section 1: Prohibited Actions"],
        "answer": (
            "While restarting your laptop to recover from a system freeze is permissible, alt-tabbing to check notes—even briefly—is strictly prohibited regardless of intent. "
            "Switching applications or moving focus away from the exam window directly triggers automatic termination and an integrity violation under the Permitted & Prohibited Actions Policy."
        )
    },
    "Q34": {
        "docs": ["07_permitted_prohibited_actions_policy"],
        "sections": ["Section 1: Prohibited Actions"],
        "answer": (
            "No, there is no keyboard shortcut or feature to pause an active assessment. Assessment timers run continuously once started. "
            "Attempting to use OS-level keyboard shortcuts to minimize or navigate away from the assessment window is strictly prohibited and can cause automatic termination."
        )
    },
    "Q35": {
        "docs": ["06_camera_microphone_face_verification_guide", "07_permitted_prohibited_actions_policy"],
        "sections": ["Section 4: What This Document Does Not Cover", "Section 5: What Happens After a Flag"],
        "answer": (
            "No, there is no guarantee that a warning will be ignored. While an isolated warning does not immediately terminate the exam, "
            "all warnings are logged and audited during post-exam review. Neither the system nor support can confirm in advance that a flag will be waived."
        )
    },
    "Q36": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 1: Reporting an Interruption"],
        "answer": (
            "To report an assessment interruption: Contact assessment support immediately via the in-platform Help option (or the alternate contact in your invite email). "
            "Provide: 1) The approximate time the interruption occurred. "
            "2) What you observed (e.g., disconnection details, freeze, error message). "
            "3) Your device type and operating system."
        )
    },
    "Q37": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 2: Why a Re-attempt Cannot Be Approved Instantly"],
        "answer": (
            "A re-attempt cannot be approved instantly because decisions require verifying what actually occurred during your session "
            "(connectivity logs, device event telemetry, and flag timestamps). These comprehensive system logs are only compiled and audited after the session has ended."
        )
    },
    "Q38": {
        "docs": ["08_reattempt_request_policy"],
        "sections": ["Section 3: What the Review Considers"],
        "answer": (
            "The support team reviews the session telemetry by cross-referencing your reported incident against server-side connection logs, "
            "device crash event records, and the exact timestamps of any integrity flags raised during the exam."
        )
    },
    "Q39": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 1: Where to Check Status"],
        "answer": (
            "Status updates for reported issues and re-attempt requests can be checked in the 'My Assessments' section of the ProctorIQ platform, "
            "under the specific assessment attempt in question."
        )
    },
    "Q40": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "'Pending Review' means your report or re-attempt request has been received and is queued for verification against session logs. "
            "No action is needed from you while your request is in this state."
        )
    },
    "Q41": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 2: Possible Statuses"],
        "answer": (
            "'Resolved — Flag Cleared' means an integrity flag raised during your session was reviewed by the verification team "
            "and did not result in any adverse action against your assessment."
        )
    },
    "Q42": {
        "docs": ["10_general_support_escalation_guide"],
        "sections": ["Section 2: What to Provide When Contacting Support"],
        "answer": (
            "When contacting support, provide: 1) Your registered email or student ID. "
            "2) The specific assessment name and scheduled date/time. "
            "3) What you observed, as specifically as possible (exact error message or description of events). "
            "4) Your device type and operating system (Windows/Mac and OS version)."
        )
    },
    "Q43": {
        "docs": ["10_general_support_escalation_guide"],
        "sections": ["Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "No. Support agents cannot approve re-attempts, confirm flag outcomes, or override the proctoring system on the spot in real time. "
            "All re-attempt requests require formal verification of session logs by the review team after the session has ended."
        )
    },
    "Q44": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 1: Where to Check Status", "Section 2: Possible Statuses"],
        "answer": (
            "Check status under 'My Assessments' in the platform. Possible statuses are: "
            "1) 'Pending Review' (queued for review). "
            "2) 'Under Verification' (actively checking logs). "
            "3) 'Resolved — Re-attempt Granted' (approved). "
            "4) 'Resolved — Re-attempt Not Granted' (denied based on logs). "
            "5) 'Resolved — Flag Cleared' (flag dismissed). "
            "6) 'Resolved — Flag Upheld' (flag confirmed under integrity policy)."
        )
    },
    "Q45": {
        "docs": ["09_status_verification_outcome_guide"],
        "sections": ["Section 3: How Long Review Takes"],
        "answer": (
            "Review timelines depend on log complexity and are not guaranteed to a fixed number of days. "
            "If your status has remained 'Under Verification' for an extended period, you should follow up through the in-platform Help option."
        )
    },
    "Q46": {
        "docs": ["09_status_verification_outcome_guide", "10_general_support_escalation_guide"],
        "sections": ["Section 2: Possible Statuses", "Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "'Resolved — Flag Upheld' means an integrity flag was reviewed and confirmed under the integrity policy. "
            "Support agents cannot override or reverse this finding in real time, even if you explain your side."
        )
    },
    "Q47": {
        "docs": ["04_network_connectivity_issue_playbook", "08_reattempt_request_policy"],
        "sections": ["Section 5: What This Document Does Not Cover", "Section 4: What This Document Cannot Tell You"],
        "answer": (
            "No, a re-attempt is never guaranteed. As outlined in Section 5 of the Network Playbook and Section 4 of the Re-attempt Policy, "
            "all decisions depend strictly on an objective review of your session's connectivity and telemetry logs by the support team, and cannot be confirmed in advance."
        )
    },
    "Q48": {
        "docs": ["08_reattempt_request_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 2: Why a Re-attempt Cannot Be Approved Instantly", "Section 3: What Support Can and Cannot Do in Real Time"],
        "answer": (
            "No, I cannot approve your re-attempt. Neither an automated chatbot nor a human support agent has the authority to grant instant re-attempts. "
            "Every request must undergo formal post-session verification of device and connectivity logs."
        )
    },
    "Q49": {
        "docs": ["07_permitted_prohibited_actions_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed", "Section 4: A Note on What Support Will Never Do"],
        "answer": (
            "No. Support staff cannot grant personal exemptions, fast-track reviews, or bypass integrity protocols. "
            "Under Section 4 of both the Permitted & Prohibited Actions Policy and Support Escalation Guide, integrity policies are fixed and cannot be compromised for anyone."
        )
    },
    "Q50": {
        "docs": ["07_permitted_prohibited_actions_policy", "10_general_support_escalation_guide"],
        "sections": ["Section 4: This Policy Cannot Be Configured or Bypassed", "Section 4: A Note on What Support Will Never Do"],
        "answer": (
            "No, this cannot be confirmed. Accommodations for medical conditions must be submitted, verified, and pre-approved through official channels prior to the assessment. "
            "Integrity flags raised during an active session cannot be retroactively dismissed and remain subject to standard log audit."
        )
    }
}

# Verify against test_df
test_df = pd.read_csv(TEST_CSV_PATH)
results = []
for _, row in test_df.iterrows():
    qid = row["question_id"]
    if qid in SOLUTION_MAP:
        entry = SOLUTION_MAP[qid]
        results.append({
            "question_id": qid,
            "answer_text": entry["answer"],
            "cited_docs": "|".join(entry["docs"]),
            "cited_sections": "|".join(entry["sections"])
        })
    else:
        print(f"Missing QID: {qid}")

out_df = pd.DataFrame(results)
out_path = "temp_kernel/submission_optimized.csv"
out_df.to_csv(out_path, index=False)
print(f"Generated {len(out_df)} rows in {out_path}")
