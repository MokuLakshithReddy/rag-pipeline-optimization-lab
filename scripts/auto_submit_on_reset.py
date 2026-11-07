import os
import subprocess
import time
import sys

KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN", "KGAT_78265fddbf5f5d92551afbd2646bc4f7")
COMPETITION = "gen-ai-project-4-building-optimizing-rag-pipelines"
SUBMISSION_FILE = "submission_golden_rank1.csv"
MESSAGE = "NxtWave Top-1 Verbatim Grounded Solution"

def check_remaining():
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    res = subprocess.run(
        ["kaggle", "competitions", "submission-limits", "-c", COMPETITION],
        capture_output=True,
        text=True,
        env=env
    )
    for line in res.stdout.splitlines():
        if "Remaining today:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    return int(parts[1].strip())
                except ValueError:
                    pass
    return 0

def submit():
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION,
        "-f", SUBMISSION_FILE,
        "-m", MESSAGE
    ]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    return res.returncode == 0

if __name__ == "__main__":
    print("Checking current Kaggle submission quota...")
    remaining = check_remaining()
    print(f"Remaining submissions: {remaining}")
    if remaining > 0:
        print("Submitting immediately...")
        success = submit()
        if success:
            print("Submission successful!")
        else:
            print("Submission failed.")
    else:
        print("Daily quota exhausted. Waiting for 00:00 UTC reset...")
