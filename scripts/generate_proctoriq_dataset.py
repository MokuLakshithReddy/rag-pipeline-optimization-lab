"""Generates standardized ProctorIQ knowledge base, 50-question test set, and sample submission.

Adheres strictly to the NxtWave Academy Gen AI Project 4 specifications:
1. Knowledge Base: ProctorIQ automated proctoring platform documentation.
2. 50 distinct test questions covering identification, audio, lockdown, anomaly, network, infractions, and security.
3. Test CSV: exactly 50 rows (question_id, question).
4. Submission CSV: exactly 50 rows (question_id, answer_text, cited_docs, cited_sections).
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

DOCUMENTS = [
    {
        "id": "doc_piq_01",
        "source": "proctoriq_candidate_verification.md",
        "section": "Section 1.0: Identity Verification Protocols",
        "text": "ProctorIQ enforces automated identity verification before exam launch. Candidates must present a valid government-issued photo ID to the webcam. The system extracts OCR data and matches facial biometrics against the ID photo with a minimum 95% facial similarity threshold. If facial matching fails after three consecutive attempts, the session triggers a Tier-2 manual proctor override."
    },
    {
        "id": "doc_piq_02",
        "source": "proctoriq_candidate_verification.md",
        "section": "Section 1.1: 360-Degree Room Scan Requirements",
        "text": "Prior to initiating the exam timer, ProctorIQ mandates a 360-degree environment room scan. The candidate must rotate their webcam across the entire testing workspace, including beneath the desk and behind monitors. Artificial intelligence models analyze the video stream for unauthorized second screens, mobile phones, or additional individuals."
    },
    {
        "id": "doc_piq_03",
        "source": "proctoriq_audio_monitoring.md",
        "section": "Section 2.0: Ambient Noise & Audio Thresholds",
        "text": "ProctorIQ continuously samples candidate microphone audio at 44.1 kHz. Background noise exceeding 65 dB for more than 3 consecutive seconds is flagged as a potential acoustic violation. Advanced acoustic models isolate vocal speech frequencies from background mechanical hums like air conditioning or fan noise."
    },
    {
        "id": "doc_piq_04",
        "source": "proctoriq_audio_monitoring.md",
        "section": "Section 2.1: Multi-Speaker & Whisper Detection",
        "text": "ProctorIQ employs diarization neural networks to separate multiple speaker voices. If distinct vocal signatures or whispered conversations are detected in the testing perimeter, an immediate High Severity alert is dispatched to the monitoring dashboard, and an audio snippet is cryptographically hashed for post-exam review."
    },
    {
        "id": "doc_piq_05",
        "source": "proctoriq_browser_lockdown.md",
        "section": "Section 3.0: Secure Browser Sandboxing",
        "text": "The ProctorIQ Secure Browser operates in exclusive full-screen kiosk mode. It disables operating system shortcut keys including Alt-Tab, Windows Key, Cmd-Tab, PrintScreen, and function keys. Clipboard history is cleared upon launch, and copy-paste operations between external applications and exam fields are strictly blocked."
    },
    {
        "id": "doc_piq_06",
        "source": "proctoriq_browser_lockdown.md",
        "section": "Section 3.1: Virtual Machine & Remote Desktop Blocking",
        "text": "ProctorIQ inspects system hardware abstraction layers to detect virtualization software. If VMware, VirtualBox, QEMU, Hyper-V, or remote access software like AnyDesk, TeamViewer, or Remote Desktop Protocol (RDP) are running, ProctorIQ halts exam launch with error code PROCTOR-VM-BLOCKED."
    },
    {
        "id": "doc_piq_07",
        "source": "proctoriq_screen_recording.md",
        "section": "Section 4.0: Display Isolation & Multi-Monitor Detection",
        "text": "ProctorIQ enforces single-display testing environments. The browser queries the OS display manager via EDID. If an external secondary monitor or HDMI capture card is detected, the exam screen blanks out until the auxiliary display cable is physically disconnected."
    },
    {
        "id": "doc_piq_08",
        "source": "proctoriq_screen_recording.md",
        "section": "Section 4.1: Screen Capture & Watermarking",
        "text": "ProctorIQ captures display frame buffers at 5 frames per second using hardware-accelerated H.264 encoding. To deter photographic leaks, an invisible steganographic watermark containing candidate ID and timestamp is dynamically tiled across the test window."
    },
    {
        "id": "doc_piq_09",
        "source": "proctoriq_anomaly_detection.md",
        "section": "Section 5.0: Gaze Tracking & Head Pose Estimation",
        "text": "ProctorIQ tracks eye gaze vectors and 3D head pose orientation. Looking away from the central monitor bounds for more than 5 cumulative seconds within a 60-second window triggers a Medium Severity gaze diversion flag. Persistent downward gaze towards lap level triggers an immediate inspection request."
    },
    {
        "id": "doc_piq_10",
        "source": "proctoriq_anomaly_detection.md",
        "section": "Section 5.1: Object Detection in Video Stream",
        "text": "ProctorIQ real-time computer vision models run edge inference on the candidate video feed. Bounding boxes are generated for unauthorized objects including smartphones, smartwatches, textbooks, earphones, and Bluetooth earbuds with confidence thresholds calibrated to 0.85."
    },
    {
        "id": "doc_piq_11",
        "source": "proctoriq_network_resilience.md",
        "section": "Section 6.0: Disconnection Grace Period & Offline Caching",
        "text": "In the event of network packet loss or ISP failure, ProctorIQ activates an automated 10-minute network grace period. The candidate can continue answering pre-loaded exam items while telemetry logs and video chunks are encrypted with AES-256 and cached locally in IndexedDB storage."
    },
    {
        "id": "doc_piq_12",
        "source": "proctoriq_network_resilience.md",
        "section": "Section 6.1: Automatic Session Resumption",
        "text": "When Internet connectivity is restored within the 10-minute window, ProctorIQ synchronizes cached answers and transmits backlogged telemetry. If the network is not restored before the 10-minute timeout expires, the exam session auto-submits current answers to preserve partial credit."
    },
    {
        "id": "doc_piq_13",
        "source": "proctoriq_infraction_severity.md",
        "section": "Section 7.0: Three-Tier Infraction Classification",
        "text": "ProctorIQ categorizes flagged events into three distinct severity tiers: Low Severity (temporary gaze drift, brief ambient noise spike under 3s), Medium Severity (persistent gaze away from screen >5s, secondary background voice), and High Severity (unauthorized person entering room, smartphone detected, VM detected, secondary monitor)."
    },
    {
        "id": "doc_piq_14",
        "source": "proctoriq_infraction_severity.md",
        "section": "Section 7.1: Proctor Intervention & Disqualification Matrix",
        "text": "Low severity flags generate dashboard warnings. Medium severity flags dispatch a two-way text prompt from the proctor. Accumulation of three Medium severity infractions or a single High severity infraction escalates to session suspension pending live proctor review."
    },
    {
        "id": "doc_piq_15",
        "source": "proctoriq_hardware_requirements.md",
        "section": "Section 8.0: Minimum Hardware & Software Specifications",
        "text": "ProctorIQ requires a 64-bit operating system (Windows 10/11 or macOS 12+), a minimum of 8 GB RAM, a dual-core 2.0 GHz processor, 500 MB free disk space, a functional webcam with 720p resolution at 30 fps, a working microphone, and a stable broadband connection with at least 2.0 Mbps upload bandwidth."
    },
    {
        "id": "doc_piq_16",
        "source": "proctoriq_security_and_compliance.md",
        "section": "Section 9.0: Data Privacy & Cryptographic Retention",
        "text": "All biometric templates, video streams, and audio recordings collected by ProctorIQ are encrypted in transit using TLS 1.3 and encrypted at rest with AES-256 keys managed in HSM. ProctorIQ strictly complies with FERPA and GDPR regulations, automatically purging candidate biometric data 30 days after exam certification."
    },
    {
        "id": "doc_piq_17",
        "source": "proctoriq_accessibility_accommodations.md",
        "section": "Section 10.0: Assistive Technology & Accommodations",
        "text": "ProctorIQ supports approved accessibility accommodations including screen magnification software, JAWS and NVDA screen readers, extended testing durations (1.5x or 2.0x time), and ergonomic movement allowances without triggering false positive gaze flags, configured through the Institutional Admin Portal."
    },
    {
        "id": "doc_piq_18",
        "source": "proctoriq_human_proctor_intervention.md",
        "section": "Section 11.0: Real-Time Live Proctor Tools",
        "text": "Live human proctors supervise up to 16 concurrent candidates via a unified matrix dashboard. Proctors can send silent warning messages, pause exam timers, initiate a live 360-degree environment re-scan, or establish secure bidirectional audio communication if anomalies arise."
    }
]

# 50 Detailed Questions mapped to ProctorIQ documentation
QUESTIONS_DATA = [
    # Identity & Verification
    ("q01", "What is the minimum facial similarity threshold required during ID verification in ProctorIQ?",
     "ProctorIQ requires a minimum 95% facial similarity threshold during automated identity matching against the ID photo.",
     "proctoriq_candidate_verification.md", "Section 1.0: Identity Verification Protocols"),
    ("q02", "What happens if candidate facial verification fails after three consecutive attempts?",
     "The session triggers a Tier-2 manual proctor override for human verification.",
     "proctoriq_candidate_verification.md", "Section 1.0: Identity Verification Protocols"),
    ("q03", "What must a candidate do during the pre-exam 360-degree environment room scan?",
     "The candidate must rotate their webcam across the entire workspace, including beneath the desk and behind monitors.",
     "proctoriq_candidate_verification.md", "Section 1.1: 360-Degree Room Scan Requirements"),
    ("q04", "Which unauthorized items does the AI model scan for during the pre-exam room check?",
     "Unauthorized second screens, mobile phones, or additional individuals.",
     "proctoriq_candidate_verification.md", "Section 1.1: 360-Degree Room Scan Requirements"),

    # Audio Monitoring
    ("q05", "What audio sampling rate does ProctorIQ use for microphone recording?",
     "ProctorIQ continuously samples candidate microphone audio at 44.1 kHz.",
     "proctoriq_audio_monitoring.md", "Section 2.0: Ambient Noise & Audio Thresholds"),
    ("q06", "What decibel level and duration triggers an ambient noise violation in ProctorIQ?",
     "Background noise exceeding 65 dB for more than 3 consecutive seconds is flagged as a potential acoustic violation.",
     "proctoriq_audio_monitoring.md", "Section 2.0: Ambient Noise & Audio Thresholds"),
    ("q07", "How does ProctorIQ distinguish human speech from mechanical noise like fans?",
     "Advanced acoustic models isolate vocal speech frequencies from background mechanical hums like air conditioning or fan noise.",
     "proctoriq_audio_monitoring.md", "Section 2.0: Ambient Noise & Audio Thresholds"),
    ("q08", "What action does ProctorIQ take if whispered conversations or multiple voices are detected?",
     "An immediate High Severity alert is dispatched to the dashboard, and an audio snippet is cryptographically hashed for post-exam review.",
     "proctoriq_audio_monitoring.md", "Section 2.1: Multi-Speaker & Whisper Detection"),
    ("q09", "Which neural network architecture separates multiple speaker voices in ProctorIQ?",
     "Diarization neural networks are employed to separate multiple speaker voices.",
     "proctoriq_audio_monitoring.md", "Section 2.1: Multi-Speaker & Whisper Detection"),

    # Browser Lockdown & Virtual Machines
    ("q10", "Which keyboard shortcuts are disabled by the ProctorIQ Secure Browser?",
     "Operating system shortcut keys including Alt-Tab, Windows Key, Cmd-Tab, PrintScreen, and function keys are disabled.",
     "proctoriq_browser_lockdown.md", "Section 3.0: Secure Browser Sandboxing"),
    ("q11", "How does ProctorIQ prevent copy-pasting between external applications and exam fields?",
     "Clipboard history is cleared upon launch and copy-paste operations between external applications and exam fields are strictly blocked.",
     "proctoriq_browser_lockdown.md", "Section 3.0: Secure Browser Sandboxing"),
    ("q12", "What specific error code is raised if virtualization software is detected on the candidate machine?",
     "ProctorIQ halts exam launch with error code PROCTOR-VM-BLOCKED.",
     "proctoriq_browser_lockdown.md", "Section 3.1: Virtual Machine & Remote Desktop Blocking"),
    ("q13", "Name four virtualization or remote access software programs blocked by ProctorIQ.",
     "VMware, VirtualBox, QEMU, Hyper-V, AnyDesk, TeamViewer, or Remote Desktop Protocol (RDP).",
     "proctoriq_browser_lockdown.md", "Section 3.1: Virtual Machine & Remote Desktop Blocking"),

    # Screen Recording & Display Isolation
    ("q14", "How does ProctorIQ detect if an external secondary monitor is plugged into the system?",
     "The browser queries the OS display manager via EDID to detect external displays or capture cards.",
     "proctoriq_screen_recording.md", "Section 4.0: Display Isolation & Multi-Monitor Detection"),
    ("q15", "What happens to the exam display if a secondary monitor is detected during the exam?",
     "The exam screen blanks out until the auxiliary display cable is physically disconnected.",
     "proctoriq_screen_recording.md", "Section 4.0: Display Isolation & Multi-Monitor Detection"),
    ("q16", "What is the frame rate and video encoding standard used for screen capture in ProctorIQ?",
     "Display frame buffers are captured at 5 frames per second using hardware-accelerated H.264 encoding.",
     "proctoriq_screen_recording.md", "Section 4.1: Screen Capture & Watermarking"),
    ("q17", "How does ProctorIQ deter candidates from taking photographs of the screen?",
     "An invisible steganographic watermark containing candidate ID and timestamp is dynamically tiled across the test window.",
     "proctoriq_screen_recording.md", "Section 4.1: Screen Capture & Watermarking"),

    # Anomaly Detection & Gaze Tracking
    ("q18", "What duration of gaze diversion outside monitor bounds triggers a Medium Severity flag?",
     "Looking away from central monitor bounds for more than 5 cumulative seconds within a 60-second window triggers a Medium Severity flag.",
     "proctoriq_anomaly_detection.md", "Section 5.0: Gaze Tracking & Head Pose Estimation"),
    ("q19", "What occurs if a candidate maintains persistent downward gaze toward lap level?",
     "Persistent downward gaze towards lap level triggers an immediate inspection request.",
     "proctoriq_anomaly_detection.md", "Section 5.0: Gaze Tracking & Head Pose Estimation"),
    ("q20", "What confidence threshold is required by computer vision models to flag unauthorized objects?",
     "Bounding boxes for unauthorized objects require confidence thresholds calibrated to 0.85.",
     "proctoriq_anomaly_detection.md", "Section 5.1: Object Detection in Video Stream"),
    ("q21", "Which physical items are detected by the edge computer vision models in ProctorIQ?",
     "Smartphones, smartwatches, textbooks, earphones, and Bluetooth earbuds.",
     "proctoriq_anomaly_detection.md", "Section 5.1: Object Detection in Video Stream"),

    # Network Resilience & Offline Support
    ("q22", "What is the duration of the ProctorIQ network disconnection grace period?",
     "ProctorIQ activates an automated 10-minute network grace period during network failures.",
     "proctoriq_network_resilience.md", "Section 6.0: Disconnection Grace Period & Offline Caching"),
    ("q23", "Where are exam answers and video chunks cached during an internet outage?",
     "Telemetry logs and video chunks are encrypted with AES-256 and cached locally in browser IndexedDB storage.",
     "proctoriq_network_resilience.md", "Section 6.0: Disconnection Grace Period & Offline Caching"),
    ("q24", "What occurs if internet connectivity is not restored within the 10-minute grace window?",
     "The exam session auto-submits current answers to preserve partial credit.",
     "proctoriq_network_resilience.md", "Section 6.1: Automatic Session Resumption"),
    ("q25", "How does ProctorIQ synchronize data when connectivity resumes within the grace period?",
     "ProctorIQ synchronizes cached answers and transmits backlogged telemetry to the server.",
     "proctoriq_network_resilience.md", "Section 6.1: Automatic Session Resumption"),

    # Infractions & Severity Tiers
    ("q26", "What constitutes a Low Severity infraction in ProctorIQ?",
     "Low Severity events include temporary gaze drift or brief ambient noise spikes under 3 seconds.",
     "proctoriq_infraction_severity.md", "Section 7.0: Three-Tier Infraction Classification"),
    ("q27", "Give two examples of events classified as Medium Severity infractions.",
     "Persistent gaze away from the screen for more than 5 seconds and secondary background voices.",
     "proctoriq_infraction_severity.md", "Section 7.0: Three-Tier Infraction Classification"),
    ("q28", "Which infractions are immediately categorized as High Severity in ProctorIQ?",
     "An unauthorized person entering the room, smartphone detected, VM detected, or secondary monitor connected.",
     "proctoriq_infraction_severity.md", "Section 7.0: Three-Tier Infraction Classification"),
    ("q29", "How many Medium severity infractions trigger exam session suspension?",
     "Accumulation of three Medium severity infractions escalates to session suspension pending live proctor review.",
     "proctoriq_infraction_severity.md", "Section 7.1: Proctor Intervention & Disqualification Matrix"),
    ("q30", "What automated proctor intervention occurs when a Medium severity flag is raised?",
     "Medium severity flags dispatch a two-way text prompt from the proctor.",
     "proctoriq_infraction_severity.md", "Section 7.1: Proctor Intervention & Disqualification Matrix"),

    # Hardware & System Requirements
    ("q31", "What are the minimum RAM and operating system requirements for ProctorIQ?",
     "A 64-bit OS (Windows 10/11 or macOS 12+) and a minimum of 8 GB RAM.",
     "proctoriq_hardware_requirements.md", "Section 8.0: Minimum Hardware & Software Specifications"),
    ("q32", "What is the minimum processor speed required to run ProctorIQ?",
     "A dual-core 2.0 GHz processor is required.",
     "proctoriq_hardware_requirements.md", "Section 8.0: Minimum Hardware & Software Specifications"),
    ("q33", "What webcam resolution and frame rate does ProctorIQ require?",
     "A functional webcam with 720p resolution at 30 frames per second.",
     "proctoriq_hardware_requirements.md", "Section 8.0: Minimum Hardware & Software Specifications"),
    ("q34", "What is the minimum broadband upload bandwidth needed for testing in ProctorIQ?",
     "A stable broadband connection with at least 2.0 Mbps upload bandwidth is required.",
     "proctoriq_hardware_requirements.md", "Section 8.0: Minimum Hardware & Software Specifications"),
    ("q35", "How much free disk space is required to install and run the ProctorIQ application?",
     "At least 500 MB of free disk space is required.",
     "proctoriq_hardware_requirements.md", "Section 8.0: Minimum Hardware & Software Specifications"),

    # Security & Compliance
    ("q36", "Which encryption protocol secures ProctorIQ audio and video data in transit?",
     "All data collected by ProctorIQ is encrypted in transit using TLS 1.3.",
     "proctoriq_security_and_compliance.md", "Section 9.0: Data Privacy & Cryptographic Retention"),
    ("q37", "What encryption algorithm and key management system protects candidate recordings at rest?",
     "Data at rest is encrypted with AES-256 keys managed in Hardware Security Modules (HSM).",
     "proctoriq_security_and_compliance.md", "Section 9.0: Data Privacy & Cryptographic Retention"),
    ("q38", "Which student privacy and data protection regulations does ProctorIQ comply with?",
     "ProctorIQ strictly complies with FERPA and GDPR regulations.",
     "proctoriq_security_and_compliance.md", "Section 9.0: Data Privacy & Cryptographic Retention"),
    ("q39", "How long does ProctorIQ retain candidate biometric templates before purging them?",
     "ProctorIQ automatically purges candidate biometric data 30 days after exam certification.",
     "proctoriq_security_and_compliance.md", "Section 9.0: Data Privacy & Cryptographic Retention"),

    # Accessibility & Accommodations
    ("q40", "Which popular screen readers are supported by ProctorIQ for visually impaired candidates?",
     "JAWS and NVDA screen readers are fully supported by ProctorIQ.",
     "proctoriq_accessibility_accommodations.md", "Section 10.0: Assistive Technology & Accommodations"),
    ("q41", "What timing extensions can be configured for candidates with approved accommodations?",
     "Extended testing durations of 1.5x or 2.0x time can be configured.",
     "proctoriq_accessibility_accommodations.md", "Section 10.0: Assistive Technology & Accommodations"),
    ("q42", "How does ProctorIQ prevent false positive gaze violations for candidates with ergonomic movement accommodations?",
     "Ergonomic movement allowances are configured through the Institutional Admin Portal to suppress false positive gaze flags.",
     "proctoriq_accessibility_accommodations.md", "Section 10.0: Assistive Technology & Accommodations"),
    ("q43", "Where are candidate accessibility accommodations officially managed and configured?",
     "Accommodations are configured through the Institutional Admin Portal.",
     "proctoriq_accessibility_accommodations.md", "Section 10.0: Assistive Technology & Accommodations"),

    # Human Proctoring & Intervention
    ("q44", "How many concurrent candidates can a single human proctor supervise on the ProctorIQ dashboard?",
     "A single human proctor can supervise up to 16 concurrent candidates via the matrix dashboard.",
     "proctoriq_human_proctor_intervention.md", "Section 11.0: Real-Time Live Proctor Tools"),
    ("q45", "What live intervention tools are available to a human proctor during an active session?",
     "Proctors can send silent warnings, pause exam timers, initiate a live 360 room re-scan, or start secure audio communication.",
     "proctoriq_human_proctor_intervention.md", "Section 11.0: Real-Time Live Proctor Tools"),
    ("q46", "Can a live proctor pause a student exam timer, and under what circumstances?",
     "Yes, proctors can pause exam timers if environmental anomalies or technical checks require investigation.",
     "proctoriq_human_proctor_intervention.md", "Section 11.0: Real-Time Live Proctor Tools"),
    ("q47", "What protocol is used if a candidate attempts to communicate with someone off-camera?",
     "Multi-speaker audio detection alerts the proctor, who can initiate bidirectional audio and request a 360-degree re-scan.",
     "proctoriq_human_proctor_intervention.md", "Section 11.0: Real-Time Live Proctor Tools"),

    # Complex & Multi-Section Synthesis
    ("q48", "Compare the consequences of a single High severity infraction versus three Medium severity infractions in ProctorIQ.",
     "Both a single High severity infraction and the accumulation of three Medium severity infractions escalate to session suspension pending live proctor review.",
     "proctoriq_infraction_severity.md", "Section 7.1: Proctor Intervention & Disqualification Matrix"),
    ("q49", "How do offline IndexedDB caching and AES-256 encryption work together during network loss in ProctorIQ?",
     "During network outages, telemetry and video chunks are encrypted using AES-256 and stored in IndexedDB for up to 10 minutes before auto-submitting.",
     "proctoriq_network_resilience.md", "Section 6.0: Disconnection Grace Period & Offline Caching"),
    ("q50", "Why does ProctorIQ tile steganographic watermarks and restrict secondary displays simultaneously?",
     "Steganographic watermarks deter photographic screen leaks while display isolation prevents HDMI capture cards and multi-monitor cheating.",
     "proctoriq_screen_recording.md", "Section 4.1: Screen Capture & Watermarking")
]


def generate_all():
    benchmarks_dir = Path("data/benchmarks")
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full JSON benchmark
    json_path = benchmarks_dir / "proctoriq_knowledge_base.json"
    queries_json = [
        {
            "question_id": q[0],
            "query": q[1],
            "ground_truth_answer": q[2],
            "cited_docs": q[3],
            "cited_sections": q[4],
            "relevant_doc_ids": [q[3]]
        }
        for q in QUESTIONS_DATA
    ]
    data = {
        "knowledge_base_name": "ProctorIQ Automated Proctoring Platform",
        "corpus": DOCUMENTS,
        "benchmark_queries": queries_json
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ProctorIQ knowledge base to {json_path} ({len(DOCUMENTS)} docs, {len(queries_json)} queries)")

    # 2. Save competition test.csv (exactly 50 rows: question_id, question)
    test_csv_path = benchmarks_dir / "proctoriq_test.csv"
    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question"])
        for q in QUESTIONS_DATA:
            writer.writerow([q[0], q[1]])
    print(f"Saved 50-row competition test set to {test_csv_path}")

    # 3. Save official sample_submission.csv (exactly 50 rows: question_id, answer_text, cited_docs, cited_sections)
    sample_sub_path = benchmarks_dir / "sample_submission.csv"
    with open(sample_sub_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "answer_text", "cited_docs", "cited_sections"])
        for q in QUESTIONS_DATA:
            writer.writerow([q[0], q[2], q[3], q[4]])
    print(f"Saved sample competition submission to {sample_sub_path}")


if __name__ == "__main__":
    generate_all()

