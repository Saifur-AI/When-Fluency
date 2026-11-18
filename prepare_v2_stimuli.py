#!/usr/bin/env python
r"""
prepare_v2_stimuli.py  --  convert the v2 stimulus packages into the input format of
stutter_rq_lmstudio.py.

INPUT   exp1_stimuli_v2_package/fluencybank_stimuli_v2.csv   (C1-C6, 126 excerpts)
        exp2_stimuli_v2_package/exp2_stimuli_v2.csv          (F, S3/S8/S15, T3/T8/T15, 120 answers)
OUTPUT  data/stimuli_v2/stimuli_exp1.csv
        data/stimuli_v2/stimuli_exp2.csv

USAGE
    python prepare_v2_stimuli.py                      # primary Exp 1 set (46 excerpts) + Exp 2
    python prepare_v2_stimuli.py --exp1-secondary     # also add stuttering-topic excerpts, flagged topic_stutter
"""
import argparse
from pathlib import Path

import pandas as pd

ROLE = {"office_analyst": "analyst", "client_facing": "client"}
# Exp 1 interview questions, matched to what each excerpt is about (exp1_question_map.csv)
QUESTIONS = {
    "ABOUT": "Tell me a little about yourself.",
    "COMM": "What does good communication mean to you? Can you give an example?",
    "WORK": "Tell me about your work experience.",
    "CHALLENGE": "Tell me about a challenge you have faced and how you handled it.",
    "OUTSIDE": "What do you enjoy doing outside of work?",
    "PRESENT": "Tell me about a time you presented or spoke in front of others.",
    "SHAPED": "How have your life experiences shaped who you are today?",
    "ADVICE": "What advice would you give to someone starting out?",
}
STUTTER_TOPIC_QUESTION = "Tell me a little about yourself."   # secondary set, analysed separately
# Exp 2 version code -> (dose_kind, dose_level)
DOSE = {"F": ("fluent", "none"),
        "S3": ("stutter", "low"), "S8": ("stutter", "medium"), "S15": ("stutter", "high"),
        "T3": ("typical", "low"), "T8": ("typical", "medium"), "T15": ("typical", "high")}


def exp1(path, secondary, qmap_path, drop_poor):
    s = pd.read_csv(path, encoding="utf-8-sig")
    keep = s.exp1_primary.astype(str).str.lower().eq("true")
    if secondary:
        # stuttering-topic windows are analysed separately (topic_stutter=True); other exclusions stay out
        keep |= s.neutral_review.eq("stuttering_content")
    s = s[keep].copy()
    qmap = pd.read_csv(qmap_path).set_index("excerpt_id")
    unknown = set(qmap.qid) - set(QUESTIONS)
    assert not unknown, f"unknown qid in {qmap_path}: {unknown}"
    primary = s.exp1_primary.astype(str).str.lower().eq("true")
    missing = set(s[primary].excerpt_id) - set(qmap.index)
    assert not missing, f"primary excerpts without a question in {qmap_path}: {sorted(missing)}"
    s["question_id"] = s.excerpt_id.map(qmap.qid)
    s["question_fit"] = s.excerpt_id.map(qmap.fit).fillna("n/a")
    s["question"] = s.question_id.map(QUESTIONS).fillna(STUTTER_TOPIC_QUESTION)
    if drop_poor:
        s = s[s.question_fit != "poor"]
    return pd.DataFrame({
        "experiment": 1,
        "speaker": s.speaker_id.astype(str),
        "excerpt": s.excerpt_id.astype(str),
        "role": s.role.map(ROLE),
        "condition": s.version,
        "text": s.text,
        "question": s.question,
        "question_id": s.question_id,
        "question_fit": s.question_fit,
        "pss": s.pss,
        "topic_stutter": ~s.exp1_primary.astype(str).str.lower().eq("true"),
        "n_sld": s.n_sld,
        "n_td": s.n_td,
        "n_words": s.word_count,
        "readability": s.readability,
    })


def exp2(path):
    s = pd.read_csv(path)
    # a T version has fillers at the positions and count of its S version, so it takes that version's dose
    st = s[s.version.str.startswith("S")].assign(lvl=lambda d: d.version.str[1:])
    dose = st.set_index(["answer_id", "lvl"]).pss_actual
    lvl = s.version.str[1:]
    s["dose_pss"] = [dose.get((a, l), 0.0) if v != "F" else 0.0 for a, l, v in zip(s.answer_id, lvl, s.version)]
    rows = []
    # every answer is rated under both roles; the role goes into the excerpt id so job ids stay unique,
    # and speaker = answer so clustering is by answer
    for role in ("analyst", "client"):
        d = s.assign(role=role)
        rows.append(pd.DataFrame({
            "experiment": 2,
            "speaker": d.answer_id,
            "excerpt": d.answer_id + "__" + role,
            "role": role,
            "condition": d.version,
            "text": d.text,
            "question": d.question,   # each answer was written for its own question
            "pss": d.pss_actual,
            "topic_stutter": False,
            "dose_kind": d.version.map(lambda v: DOSE[v][0]),
            "dose_level": d.version.map(lambda v: DOSE[v][1]),
            "dose_pss": d.dose_pss,
            "question_id": d.question_id,
            "n_words": d.word_count,
            "readability": d.readability,
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp1", default="exp1_stimuli_v2_package/fluencybank_stimuli_v2_c5v3.csv",
                    help="v2 table with C5 rebuilt by rebuild_c5.py (protocol v3 rule)")
    ap.add_argument("--exp2", default="exp2_stimuli_v2_package/exp2_stimuli_v2.csv")
    ap.add_argument("--out", default="data/stimuli_v2")
    ap.add_argument("--qmap", default="exp1_stimuli_v2_package/exp1_question_map.csv",
                    help="excerpt -> matched interview question")
    ap.add_argument("--exp1-secondary", action="store_true", help="add stuttering-topic excerpts (topic_stutter=True)")
    ap.add_argument("--drop-poor-fit", action="store_true", help="leave out excerpts whose question fit is 'poor'")
    ap.add_argument("--no-exp2", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    s1 = exp1(a.exp1, a.exp1_secondary, a.qmap, a.drop_poor_fit)
    assert s1.role.notna().all(), "unknown role in Exp 1 file"
    s1.to_csv(out / "stimuli_exp1.csv", index=False)
    n = s1[~s1.topic_stutter]
    print(f"Exp 1: {len(s1)} rows; neutral {n.excerpt.nunique()} excerpts from {n.speaker.nunique()} speakers; "
          f"stuttering-topic {s1[s1.topic_stutter].excerpt.nunique()} excerpts")
    fits = n.drop_duplicates("excerpt").question_fit.value_counts().to_dict()
    print(f"       question fit (neutral excerpts): {fits}")

    if not a.no_exp2:
        s2 = exp2(a.exp2)
        s2.to_csv(out / "stimuli_exp2.csv", index=False)
        print(f"Exp 2: {len(s2)} rows; {s2.speaker.nunique()} answers x {s2.condition.nunique()} versions x 2 roles")
    print(f"Written to {out}/")


if __name__ == "__main__":
    main()
