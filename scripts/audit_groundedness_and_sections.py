import csv
import glob
import re
from pathlib import Path

with open("kaggle_competition_data/Data/Test CSV/test.csv", encoding="utf-8") as f:
    tests = {r["question_id"]: r["question"] for r in csv.DictReader(f)}

with open("submission.csv", encoding="utf-8") as f:
    subs = {r["question_id"]: r for r in csv.DictReader(f)}

docs = {}
sec_texts = {}
for p in sorted(glob.glob("kaggle_competition_data/Data/.md Files/*.md")):
    stem = Path(p).stem
    txt = open(p, encoding="utf-8").read()
    docs[stem] = txt
    for part in txt.split("\n## "):
        if not part.strip():
            continue
        lines = part.strip().split("\n", 1)
        title = lines[0].strip().lstrip("#").strip()
        body = lines[1] if len(lines) > 1 else ""
        sec_texts[(stem, title)] = body

stop = set(
    "a an the is are was were to in on of and or for it its you your my i this that "
    "not do does did can cannot could would should will be by as at with from if when "
    "what how why which who where have has had into than then also only just make may "
    "might must need use using any all one two no yes per does".split()
)


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_/-]{3,}", s.lower())) - stop


print("=== GROUNDEDNESS & SECTION OVERLAP AUDIT (submission.csv) ===\n")
for qid in sorted(tests):
    r = subs[qid]
    cdocs = r["cited_docs"].split("|")
    csecs = r["cited_sections"].split("|")
    cited_body = ""
    for d, s in zip(cdocs, csecs):
        matched = ""
        for (kd, ks), v in sec_texts.items():
            if kd == d and (ks == s or ks.startswith(s) or s in ks):
                matched += "\n" + ks + "\n" + v
        cited_body += matched
    ans_t = toks(r["answer_text"])
    cit_t = toks(cited_body)
    ungrounded = sorted(ans_t - cit_t)
    other_hits = []
    for (kd, ks), v in sec_texts.items():
        if kd in cdocs and ks not in csecs:
            ov = ans_t & (toks(v) - cit_t)
            if len(ov) >= 2:
                other_hits.append(f"{kd}::{ks}({len(ov)}:{sorted(ov)[:6]})")
    # Also check uncited docs that strongly match question tokens!
    q_t = toks(tests[qid])
    doc_q_matches = []
    for (kd, ks), v in sec_texts.items():
        ov = q_t & (toks(ks) | toks(v))
        if len(ov) >= 4 and (kd not in cdocs or ks not in csecs):
            doc_q_matches.append((len(ov), f"{kd}::{ks}", sorted(ov)[:5]))
    doc_q_matches.sort(reverse=True)
    if len(ungrounded) > 2 or other_hits:
        print(f"[{qid}] docs={r['cited_docs']} | secs={r['cited_sections']}")
        print(f"  Q: {tests[qid]}")
        print(f"  A: {r['answer_text']}")
        print(f"  Ungrounded ({len(ungrounded)}): {ungrounded}")
        if other_hits:
            print(f"  Other sec leaks: {other_hits}")
        if doc_q_matches[:3]:
            print(f"  Alt Q matches: {[(m[0], m[1], m[2]) for m in doc_q_matches[:3]]}")
        print()

