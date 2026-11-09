import csv
import copy

with open('submission_master_top1.csv', 'r', encoding='utf-8') as f:
    rows = {r['question_id']: r for r in csv.DictReader(f)}

# Refine Q13
rows['Q13']['answer_text'] = (
    "On Windows, 'Element not found' occurs when a required system component (usually a display or kiosk-mode driver) failed to register during install: "
    "uninstall PSB completely via Control Panel → Programs, restart your laptop, re-download the installer fresh (do not reuse the old installer file), "
    "and install again as Administrator, then restart once more before opening PSB. "
    "On Mac, 'Element not found' most often means one or more of the required permissions (Camera, Microphone, Screen Recording, Accessibility) was not granted: "
    "open System Settings → Privacy & Security, manually verify all four permissions are toggled on for PSB, and quit and relaunch PSB after changing permissions."
)

# Refine Q28
rows['Q28']['answer_text'] = (
    "If a candidate accidentally performs a three-finger swipe on their touchpad, the assessment may terminate automatically because the system detects it as an application-switch attempt — this is a known behavior, not a malfunction. "
    "If your assessment is interrupted due to a technical issue, report it to the assessment support team immediately — through the in-platform Help option if you are still able to access it, or through the alternate contact method provided in your original assessment invite if not. "
    "Provide the approximate time the interruption occurred, what you observed, and your device type and operating system. "
    "A re-attempt cannot be approved at the time of the interruption; decisions require verifying what actually happened during your session (connectivity logs, device event logs, and the specific timing of any flags raised)."
)

# Refine Q31
rows['Q31']['answer_text'] = (
    "No. Attempting to disable, obscure, or interfere with the camera, microphone, or face-verification system is not permitted at any point during an active assessment, regardless of intent, and disabling the camera feed triggers automatic termination. "
    "There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. "
    "Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
)

# Refine Q34
rows['Q34']['answer_text'] = (
    "No. There is no supported keyboard shortcut to pause an assessment. "
    "The following actions are not permitted at any point during an active assessment, regardless of intent: Using unsupported keyboard shortcuts (including OS-level shortcuts to switch applications, open a virtual desktop, or take a screenshot), and any action that moves focus away from the assessment window. "
    "Automatic termination is triggered by actions that directly match a defined prohibited pattern."
)

# Refine Q49
rows['Q49']['answer_text'] = (
    "No. Support cannot, in real time and during an active assessment, confirm the outcome of an integrity flag or override the proctoring/integrity system — these all require the log-review process. "
    "Support will never provide guidance intended to help a student bypass, disable, or reduce face verification, application-lock, or any other integrity control, regardless of the reason given. "
    "There is no setting, workaround, or support-assisted method to disable face verification, avoid application-switch detection, or otherwise reduce the proctoring requirements for an individual assessment. "
    "Any request to do so — regardless of the stated reason — is outside what this system, or any member of the support team, can provide. This is a fixed integrity boundary, not a configurable preference."
)

output_path = 'submission_grandmaster_top1.csv'
fieldnames = ['question_id', 'answer_text', 'cited_docs', 'cited_sections']

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for qid in sorted(rows.keys()):
        writer.writerow(rows[qid])

print(f"Generated {output_path} with 50 rows.")

