import csv
import os

solutions = {
    "Q01": {
        "docs": "01_windows_installation_login_guide",
        "secs": "Section 2: Common Installation Errors",
        "ans": "This error occurs when a required system component (usually a display or kiosk-mode driver) failed to register during install. Uninstall PSB completely via Control Panel → Programs. Restart your laptop. Re-download the installer fresh (do not reuse the old installer file). Install again as Administrator, then restart once more before opening PSB."
    },
    "Q02": {
        "docs": "01_windows_installation_login_guide",
        "secs": "Section 2: Common Installation Errors",
        "ans": "This error appears when PSB cannot establish a secure session with the ProctorIQ assessment server, usually due to a firewall or antivirus blocking the connection. Temporarily disable third-party antivirus/firewall software (not Windows Defender — Defender is compatible with PSB). Confirm your internet connection is active outside of PSB. Reopen PSB and attempt login again. If the error persists after two attempts, restart your laptop before trying a third time."
    },
    "Q03": {
        "docs": "01_windows_installation_login_guide",
        "secs": "Section 2: Common Installation Errors",
        "ans": "A generic failure that can occur for several underlying reasons. Standard resolution sequence: Close PSB completely (check Task Manager to confirm no PSB process is still running). Restart your laptop. Reopen PSB and attempt the mock test again before your actual exam slot. If it recurs after a restart, this is escalated to Support with your device logs — do not keep retrying more than three times."
    },
    "Q04": {
        "docs": "01_windows_installation_login_guide",
        "secs": "Section 3: Login Issues",
        "ans": "PSB requires a full account login through its native window, not a browser tab. If you were previously redirected to a browser incognito window during login: Close all browser windows. Reopen PSB from the desktop shortcut only. Log in using the credentials from your assessment invite email — do not use saved browser passwords, as PSB does not read browser-stored credentials."
    },
    "Q05": {
        "docs": "02_mac_installation_login_guide",
        "secs": "Section 2: Common Installation Errors",
        "ans": "On Mac, this most often means one or more of the required permissions (Camera, Microphone, Screen Recording, Accessibility) was not granted. Open System Settings → Privacy & Security and manually verify all four permissions are toggled on for PSB. Quit and relaunch PSB after changing permissions — permission changes do not apply to an already-running session."
    },
    "Q06": {
        "docs": "02_mac_installation_login_guide",
        "secs": "Section 4: Apple Silicon Note",
        "ans": "No. PSB runs natively on both Intel and Apple Silicon (M-series) Macs. There is no separate installer for M-series chips — the same .dmg file detects your architecture automatically. If PSB seems unusually slow to launch on an M-series Mac, this is normal on first launch only (Gatekeeper verification); subsequent launches are faster."
    },
    "Q07": {
        "docs": "02_mac_installation_login_guide",
        "secs": "Section 3: Login Issues",
        "ans": "Confirm your registered contact details match your assessment profile if the OTP does not arrive within 2 minutes. OTPs expire after 5 minutes — request a fresh one rather than reusing an expired code."
    },
    "Q08": {
        "docs": "02_mac_installation_login_guide",
        "secs": "Section 3: Login Issues",
        "ans": "No, you do not need to contact support. A forgotten PIN can be reset via the 'Forgot PIN' link on the PSB login screen — this is self-service and does not require Support."
    },
    "Q09": {
        "docs": "03_mock_test_device_readiness_guide",
        "secs": "Section 3: Common Mock-Test Failures and Fixes",
        "ans": "Confirm no other application (video calling software, other browser tabs) is currently using your camera — most webcams can only be accessed by one application at a time. Check camera permissions are granted to PSB (see the Windows or Mac installation guide, Section 1). Try a different USB port if using an external webcam."
    },
    "Q10": {
        "docs": "03_mock_test_device_readiness_guide",
        "secs": "Section 3: Common Mock-Test Failures and Fixes",
        "ans": "Sit facing a light source (window or lamp) rather than with it behind you. Avoid strong backlighting, which causes your face to appear as a silhouette. Remove sunglasses, hats, or anything obscuring your face during the verification capture."
    },
    "Q11": {
        "docs": "03_mock_test_device_readiness_guide",
        "secs": "Section 3: Common Mock-Test Failures and Fixes",
        "ans": "Move closer to your Wi-Fi router if using wireless. Close other bandwidth-heavy applications (streaming, large downloads) before running the check. If consistently unstable, consider a wired Ethernet connection for your actual assessment — this is the most reliable option where available."
    },
    "Q12": {
        "docs": "03_mock_test_device_readiness_guide",
        "secs": "Section 2: Recommended Timing",
        "ans": "Complete your mock test at least 24 hours before your scheduled assessment window. This is not a hard technical requirement, but it is the single most effective way to avoid last-minute issues — problems found during a mock test can be resolved with Support ahead of time; the same problems found five minutes before your exam window often cannot."
    },
    "Q13": {
        "docs": "01_windows_installation_login_guide|02_mac_installation_login_guide",
        "secs": "Section 2: Common Installation Errors|Section 2: Common Installation Errors",
        "ans": "On Windows, 'Element not found' occurs when a required system component (display or kiosk-mode driver) failed to register during install, resolved by uninstalling via Control Panel, restarting, re-downloading fresh, and installing as Administrator. On Mac, 'Element not found' means one or more required permissions (Camera, Microphone, Screen Recording, Accessibility) was not granted, resolved by opening System Settings → Privacy & Security, manually verifying all four permissions are toggled on, and relaunching PSB."
    },
    "Q14": {
        "docs": "01_windows_installation_login_guide|07_permitted_prohibited_actions_policy",
        "secs": "Section 3: Login Issues|Section 1: Prohibited Actions",
        "ans": "No. Allowing another person to view your screen or assist with your responses during the session, or using a second device (phone, tablet, second laptop) to search for or discuss assessment content during the session, is not permitted at any point during an active assessment, regardless of intent. If the OTP is not received within 2 minutes, check that the phone number/email on your assessment profile matches your registered contact details."
    },
    "Q15": {
        "docs": "03_mock_test_device_readiness_guide",
        "secs": "Section 2: Recommended Timing",
        "ans": "No. Complete your mock test at least 24 hours before your scheduled assessment window. This is not a hard technical requirement, but it is the single most effective way to avoid last-minute issues — problems found during a mock test can be resolved with Support ahead of time; the same problems found five minutes before your exam window often cannot."
    },
    "Q16": {
        "docs": "04_network_connectivity_issue_playbook",
        "secs": "Section 1: If Writing in ProctorIQ Secure Browser (PSB)",
        "ans": "Stay on the assessment screen. Do not close PSB or attempt to reopen the assessment from elsewhere. Use the Wi-Fi/network option available at the bottom of the PSB screen to reconnect. Once the connection is restored, continue your assessment from where you left off — your progress up to the point of disconnection is preserved."
    },
    "Q17": {
        "docs": "04_network_connectivity_issue_playbook",
        "secs": "Section 2: If Writing in a Normal Browser",
        "ans": "Press the Esc key to exit full-screen mode. Reconnect to your Wi-Fi or internet connection through your operating system's network settings. Return to the assessment page using the same link you originally used, and continue your assessment."
    },
    "Q18": {
        "docs": "04_network_connectivity_issue_playbook",
        "secs": "Section 4: When to Report Instead of Retry",
        "ans": "No. If you are unable to reconnect after several attempts, or the disconnection recurs more than twice in the same session, stop attempting to self-resolve and report the issue to the assessment support team immediately with the approximate time(s) the disconnection occurred. This information is needed for any later review of your session."
    },
    "Q19": {
        "docs": "05_device_system_issue_playbook",
        "secs": "Section 1: Assessment Page Freezes or Gets Stuck",
        "ans": "Wait a few seconds first and check whether the page resumes automatically — many freezes are momentary. If the issue is due to your laptop itself becoming unresponsive (not just the assessment page), press and hold the power button to force a restart. After restarting, reopen the assessment using your original assessment link and continue, if the assessment window permits re-entry."
    },
    "Q20": {
        "docs": "05_device_system_issue_playbook",
        "secs": "Section 3: Low Battery or Power Loss",
        "ans": "If your device shuts down due to battery loss, reconnect power, restart the device, and reopen the assessment using your original link as soon as possible. Always assess on a device connected to power where possible; do not rely on battery alone for a full assessment session."
    },
    "Q21": {
        "docs": "06_camera_microphone_face_verification_guide",
        "secs": "Section 1: Responding to a Warning",
        "ans": "If you receive a warning related to your camera, microphone, or face verification while assessing: Ensure your face is clearly and fully visible to the camera. Confirm you are in a well-lit environment — avoid sitting with a light source directly behind you. Verify that camera and microphone permissions are still enabled for PSB. Follow any on-screen instructions displayed alongside the warning, which may differ slightly by warning type."
    },
    "Q22": {
        "docs": "06_camera_microphone_face_verification_guide",
        "secs": "Section 2: Common Causes of Repeated Warnings",
        "ans": "Common causes of repeated warnings: Face partially out of frame: reposition your webcam or your seating so your full face remains visible for the duration of the assessment. Backlighting: a bright window or light source behind you causes the system to lose a clear read on your face; face a light source rather than sitting with your back to it. Multiple faces detected: only the assessing student should be visible in the camera frame. Camera obstruction: ensure nothing (a sticky note, laptop camera cover, hand) is blocking the lens."
    },
    "Q23": {
        "docs": "06_camera_microphone_face_verification_guide",
        "secs": "Section 2: Common Causes of Repeated Warnings",
        "ans": "This was triggered by multiple faces detected: only the assessing student should be visible in the camera frame at any point during the assessment."
    },
    "Q24": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 1: Prohibited Actions",
        "ans": "The following actions are not permitted at any point during an active assessment, regardless of intent: Swiping with three or four fingers on a touchpad. Using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot). Attempting to switch applications, browser tabs, or virtual desktops. Any action that moves focus away from the assessment window. Using a second device (phone, tablet, second laptop) to search for or discuss assessment content during the session. Allowing another person to view your screen or assist with your responses during the session. Attempting to disable, obscure, or interfere with the camera, microphone, or face-verification system."
    },
    "Q25": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 3: Automatic Termination vs. Logged Flag",
        "ans": "Automatic termination: triggered by actions that directly match a defined prohibited pattern (e.g., a detected application-switch attempt, closing PSB, disabling the camera feed). The assessment ends immediately when this occurs. Logged flag, not automatic termination: certain borderline signals (a brief face-verification warning, a momentary loss of camera feed due to lighting) are logged for later review but do not end the assessment on their own."
    },
    "Q26": {
        "docs": "04_network_connectivity_issue_playbook|06_camera_microphone_face_verification_guide",
        "secs": "Section 1: If Writing in ProctorIQ Secure Browser (PSB)|Section 1: Responding to a Warning",
        "ans": "Stay on the assessment screen and use the Wi-Fi/network option at the bottom of the PSB screen to reconnect without closing PSB. For the face verification warning, ensure your face is clearly and fully visible to the camera in a well-lit environment and verify camera permissions are still enabled. Follow any on-screen instructions displayed alongside the warning."
    },
    "Q27": {
        "docs": "05_device_system_issue_playbook|06_camera_microphone_face_verification_guide",
        "secs": "Section 1: Assessment Page Freezes or Gets Stuck|Section 1: Responding to a Warning",
        "ans": "No, this is not a problem. Press and hold the power button to force a restart, then reopen using your original assessment link. When reopening, ensure your face is clearly and fully visible to the camera in a well-lit environment and verify camera permissions. A single or occasional warning is logged but does not end your assessment."
    },
    "Q28": {
        "docs": "07_permitted_prohibited_actions_policy|08_reattempt_request_policy",
        "secs": "Section 2: Why This Matters|Section 1: Reporting an Interruption",
        "ans": "If a candidate accidentally performs a three-finger swipe on their touchpad, the assessment may terminate automatically because the system detects it as an application-switch attempt — this is a known behavior, not a malfunction. If your assessment is interrupted due to this technical issue, report it to the assessment support team immediately with the approximate time, what you observed, and your device type and operating system. Re-attempt decisions require log review and cannot be confirmed immediately."
    },
    "Q29": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 4: This Policy Cannot Be Configured or Bypassed",
        "ans": "No. Face verification cannot be turned off. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
    },
    "Q30": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 4: This Policy Cannot Be Configured or Bypassed",
        "ans": "No. The camera cannot be disabled. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
    },
    "Q31": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 4: This Policy Cannot Be Configured or Bypassed",
        "ans": "No. Attempting to disable, obscure, or interfere with the camera, microphone, or face-verification system is prohibited, and disabling the camera feed triggers automatic termination. There is no setting, workaround, or support-assisted method to disable face verification or reduce the proctoring requirements for an individual assessment, regardless of technical issues. This is a fixed integrity boundary, not a configurable preference."
    },
    "Q32": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 1: Prohibited Actions",
        "ans": "No. Allowing another person to view your screen or assist with your responses during the session is strictly prohibited. Actions that directly match a defined prohibited pattern trigger automatic termination of the assessment."
    },
    "Q33": {
        "docs": "05_device_system_issue_playbook|07_permitted_prohibited_actions_policy",
        "secs": "Section 1: Assessment Page Freezes or Gets Stuck|Section 1: Prohibited Actions",
        "ans": "No, checking notes or switching applications is not permitted. If the issue is due to your laptop itself becoming unresponsive (not just the assessment page), press and hold the power button to force a restart. After restarting, reopen the assessment using your original assessment link and continue, if the assessment window permits re-entry. However, attempting to switch applications, browser tabs, or virtual desktops, using unsupported keyboard shortcuts, or any action that moves focus away from the assessment window is not permitted at any point during an active assessment, regardless of intent."
    },
    "Q34": {
        "docs": "07_permitted_prohibited_actions_policy",
        "secs": "Section 1: Prohibited Actions",
        "ans": "No. There is no keyboard shortcut or option to pause an assessment. Using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot) and any action that moves focus away from the assessment window is strictly prohibited and triggers automatic termination."
    },
    "Q35": {
        "docs": "06_camera_microphone_face_verification_guide|09_status_verification_outcome_guide",
        "secs": "Section 4: What This Document Does Not Cover|Section 4: What This Document Does Not Do",
        "ans": "No. This document does not determine whether a specific flagged warning will affect your assessment outcome — that determination is made through a separate verification review process, not disclosed to the student in real time during the assessment. No outcome can be confirmed in advance of the review described in the Re-attempt Request Policy or the Permitted & Prohibited Actions Policy — this guide only reflects outcomes once a review has actually concluded and been recorded."
    },
    "Q36": {
        "docs": "08_reattempt_request_policy",
        "secs": "Section 1: Reporting an Interruption",
        "ans": "If your assessment is interrupted due to a technical issue, report it to the assessment support team immediately — through the in-platform Help option if you are still able to access it, or through the alternate contact method provided in your original assessment invite if not. Provide the following when reporting: The approximate time the interruption occurred. What you observed (e.g., 'internet disconnected twice,' 'laptop became unresponsive and I had to restart'). Your device type and operating system."
    },
    "Q37": {
        "docs": "08_reattempt_request_policy",
        "secs": "Section 2: Why a Re-attempt Cannot Be Approved Instantly",
        "ans": "A re-attempt cannot be approved at the time of the interruption, during the assessment, or immediately after it ends. This is not a delay in service — it reflects that a decision requires verifying what actually happened during your session (connectivity logs, device event logs, and the specific timing of any flags raised), which is not available for review until after the session has fully ended."
    },
    "Q38": {
        "docs": "08_reattempt_request_policy",
        "secs": "Section 3: What the Review Considers",
        "ans": "The support team verifies the assessment logs and technical details of the reported session before deciding whether a re-attempt can be provided. This includes cross-referencing the timing of any reported interruption against the system's own logs for that session."
    },
    "Q39": {
        "docs": "09_status_verification_outcome_guide",
        "secs": "Section 1: Where to Check Status",
        "ans": "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section of the ProctorIQ platform, under the specific assessment attempt in question."
    },
    "Q40": {
        "docs": "09_status_verification_outcome_guide",
        "secs": "Section 2: Possible Statuses",
        "ans": "Pending Review: your report or request has been received and is queued for review against session logs. No action is needed from you while a request is in this state."
    },
    "Q41": {
        "docs": "09_status_verification_outcome_guide",
        "secs": "Section 2: Possible Statuses",
        "ans": "Resolved — Flag Cleared: an integrity flag raised during your session was reviewed and did not result in any action against your assessment."
    },
    "Q42": {
        "docs": "10_general_support_escalation_guide",
        "secs": "Section 2: What to Provide When Contacting Support",
        "ans": "To help resolve your issue as quickly as possible, provide: Your registered email or student ID. The specific assessment name and scheduled date/time. What you observed, as specifically as possible (exact error message, or a description of what happened and when). Your device type and operating system (Windows/Mac, and version if known)."
    },
    "Q43": {
        "docs": "10_general_support_escalation_guide",
        "secs": "Section 3: What Support Can and Cannot Do in Real Time",
        "ans": "No. Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome of an integrity flag, or override the proctoring/integrity system — these all require the log-review process described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions any individual support agent can make on the spot."
    },
    "Q44": {
        "docs": "09_status_verification_outcome_guide",
        "secs": "Section 2: Possible Statuses",
        "ans": "Status updates for reported issues and re-attempt requests are visible in the 'My Assessments' section under the specific assessment attempt. Possible statuses: Pending Review: your report or request has been received and is queued for review against session logs; no action is needed from you. Under Verification: the support team is actively cross-referencing your report against connectivity, device, or integrity logs. Resolved — Re-attempt Granted: your request was reviewed and a re-attempt has been approved. Resolved — Re-attempt Not Granted: your request was reviewed and, based on the session logs, a re-attempt was not approved. Resolved — Flag Cleared: an integrity flag raised during your session was reviewed and did not result in any action against your assessment. Resolved — Flag Upheld: an integrity flag raised during your session was reviewed and confirmed."
    },
    "Q45": {
        "docs": "09_status_verification_outcome_guide",
        "secs": "Section 3: How Long Review Takes",
        "ans": "Review timelines vary depending on the complexity of the session logs involved and are not fixed to a guaranteed number of hours or days in this document. If your status has remained at 'Pending Review' or 'Under Verification' for an extended period, you may follow up through the in-platform Help option."
    },
    "Q46": {
        "docs": "09_status_verification_outcome_guide|10_general_support_escalation_guide",
        "secs": "Section 2: Possible Statuses|Section 3: What Support Can and Cannot Do in Real Time",
        "ans": "No, support cannot override it. Resolved — Flag Upheld means an integrity flag raised during your session was reviewed and confirmed; this outcome is handled under the assessment's integrity policy, separately from this document. Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome of an integrity flag, or override the proctoring/integrity system — these all require the log-review process described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions any individual support agent can make on the spot."
    },
    "Q47": {
        "docs": "04_network_connectivity_issue_playbook|08_reattempt_request_policy",
        "secs": "Section 5: What This Document Does Not Cover|Section 4: What This Document Cannot Tell You",
        "ans": "No, a re-attempt is not guaranteed and cannot be confirmed now. This document does not cover whether a re-attempt will be granted due to a connectivity issue. Re-attempt decisions require review of your session's connectivity logs by the support team and cannot be confirmed at the time of the issue. This document cannot confirm whether a specific re-attempt request will be approved — that depends on the outcome of the log review on a case-by-case basis and is communicated once that review is complete."
    },
    "Q48": {
        "docs": "08_reattempt_request_policy|10_general_support_escalation_guide",
        "secs": "Section 2: Why a Re-attempt Cannot Be Approved Instantly|Section 3: What Support Can and Cannot Do in Real Time",
        "ans": "No, I cannot personally approve your re-attempt. A re-attempt cannot be approved at the time of the interruption, during the assessment, or immediately after it ends. Support cannot, in real time and during an active assessment, approve a re-attempt, confirm the outcome of an integrity flag, or override the proctoring/integrity system — these all require the log-review process described in the Re-attempt Request Policy and the Status & Verification Outcome Guide, and are not decisions any individual support agent can make on the spot."
    },
    "Q49": {
        "docs": "07_permitted_prohibited_actions_policy|10_general_support_escalation_guide",
        "secs": "Section 4: This Policy Cannot Be Configured or Bypassed|Section 4: A Note on What Support Will Never Do",
        "ans": "No. Support cannot fast-track your case or confirm that it won't count against you. Support will never provide guidance intended to help a student bypass, disable, or reduce face verification, application-lock, or any other integrity control, regardless of the reason given. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
    },
    "Q50": {
        "docs": "07_permitted_prohibited_actions_policy|10_general_support_escalation_guide",
        "secs": "Section 4: This Policy Cannot Be Configured or Bypassed|Section 4: A Note on What Support Will Never Do",
        "ans": "No, this cannot be confirmed. There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. Any request to do so — regardless of the stated reason — is outside what this system or any member of the support team can provide. This is a fixed integrity boundary, not a configurable preference. Support will never provide guidance intended to help a student bypass, disable, or reduce face verification, application-lock, or any other integrity control, regardless of the reason given."
    }
}

output_path = "submission_master_top1.csv"
fieldnames = ["question_id", "answer_text", "cited_docs", "cited_sections"]

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for qid in sorted(solutions.keys()):
        item = solutions[qid]
        writer.writerow({
            "question_id": qid,
            "answer_text": item["ans"],
            "cited_docs": item["docs"],
            "cited_sections": item["secs"]
        })

print(f"Generated {output_path} with {len(solutions)} rows.")

