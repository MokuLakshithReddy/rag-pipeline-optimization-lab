import os
import glob
import re

KB_DIR = "kaggle_competition_data/Data/.md Files"

docs = {}
for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        docs[doc_id] = f.read()

# Let's inspect sections in 01_windows_installation_login_guide
print("=== Section 2 in 01 Windows ===")
part = docs["01_windows_installation_login_guide"].split("## Section 2: Common Installation Errors")[1].split("## Section 3")[0]
print(part)
