"""Rebuild C5 in the v2 stimulus table with the protocol v3 rule.

C5 (v3) = C1 with every stuttering-like event replaced by one filler (alternating um / uh) at the same
position; typical disfluencies (fillers, revisions, Valente MUR) are kept as in C1. C1, C2, C3, C4 and C6
are unchanged; the rebuild checks that it reproduces C1 and C2 exactly before replacing C5.

  python rebuild_c5.py --romana ../fluencybank/Romana/TimeStamped/metadata.csv \
      --valente ../fluencybank/Valente/Valente/interview \
      --stimuli fluencybank_stimuli_v2.csv --out fluencybank_stimuli_v2_c5v3.csv
"""
import argparse

import pandas as pd

from build_stimuli import build_excerpt, flesch, load_valente, tok
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--romana", default="../fluencybank/Romana/TimeStamped/metadata.csv")
    ap.add_argument("--valente", default="../fluencybank/Valente/Valente/interview")
    ap.add_argument("--stimuli", default="fluencybank_stimuli_v2.csv")
    ap.add_argument("--out", default="fluencybank_stimuli_v2_c5v3.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.romana)
    df["spk"] = df.segid.str.rsplit("_", n=1).str[0]
    df["verbatim"] = df.verbatim.fillna("")
    df["intended"] = df.intended.fillna("")
    seg_ix = df.set_index("segid")
    ev = load_valente(a.valente)
    st = pd.read_csv(a.stimuli, encoding="utf-8-sig")

    new_c5, n_fill, mismatch = {}, {}, []
    for r in st[st.version == "C1"].itertuples():
        segs = seg_ix.loc[r.segids.split()].reset_index()
        c1, c2, c5, stats = build_excerpt(segs, ev[ev.spk == r.speaker_id], c5_mode="v3")
        old_c2 = st[(st.excerpt_id == r.excerpt_id) & (st.version == "C2")].text.iloc[0]
        if c1 != r.text or c2 != old_c2:
            mismatch.append(r.excerpt_id)
        new_c5[r.excerpt_id] = c5
        n_fill[r.excerpt_id] = stats["n_c5_fillers"]
    if mismatch:
        raise SystemExit(f"C1/C2 not reproduced for {len(mismatch)} excerpts, C5 not replaced: {mismatch[:10]}")

    m = st.version == "C5"
    st.loc[m, "text"] = st.loc[m, "excerpt_id"].map(new_c5)
    st.loc[m, "version_label"] = "typical-disfluency control (v3: C1 with stutters as fillers)"
    st.loc[m, "word_count"] = st.loc[m, "text"].map(lambda t: len(re.findall(r"[A-Za-z0-9']+", t)))
    st.loc[m, "readability"] = st.loc[m, "text"].map(flesch)
    st["n_c5_fillers"] = st.excerpt_id.map(n_fill)
    st.to_csv(a.out, index=False, encoding="utf-8-sig")

    # checks: C5 keeps C1's words outside the stuttering events, and adds one filler per replaced event
    c1 = st[st.version == "C1"].set_index("excerpt_id").text
    c5 = st[st.version == "C5"].set_index("excerpt_id").text
    added = {e: sum(w in ("um", "uh") for w in tok(c5[e])) - sum(w in ("um", "uh") for w in tok(c1[e])) for e in c5.index}
    bad = [e for e in c5.index if added[e] != n_fill[e]]
    print(f"rebuilt C5 for {len(new_c5)} excerpts; C1 and C2 reproduced exactly")
    print(f"fillers added per excerpt: median {pd.Series(n_fill).median():.0f}, range {min(n_fill.values())}-{max(n_fill.values())}")
    print(f"filler-count check (C5 um/uh minus C1 um/uh == events replaced): {len(c5) - len(bad)}/{len(c5)} ok")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
