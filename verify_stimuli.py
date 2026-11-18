#!/usr/bin/env python
"""verify_stimuli.py  --  check that rebuilt stimuli are identical to the ones used in the paper.

Experiment 1 texts are compared by SHA-256 against data/stimuli_v2/stimuli_exp1_manifest.csv, because the
FluencyBank text itself cannot be redistributed. Every job id (and so every cached model call) is a hash of
the stimulus text, so identical hashes mean a rerun matches the released outputs row for row.

    python verify_stimuli.py                       # checks data/stimuli_v2/stimuli_exp1.csv and stimuli_exp2.csv
    python verify_stimuli.py --exp2-only           # without FluencyBank access
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

KEY = ["excerpt", "condition"]


def sha(t):
    return hashlib.sha256(str(t).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", default="data/stimuli_v2")
    ap.add_argument("--exp2-only", action="store_true")
    ap.add_argument("--exp2-reference", default="exp2_stimuli_v2_package/exp2_stimuli_v2.csv",
                    help="released Exp 2 table; a rebuilt table is compared against it")
    ap.add_argument("--exp2-rebuilt", default=None, help="Exp 2 table written by build_exp2.py, to compare")
    a = ap.parse_args()
    ok = True
    stim = Path(a.stim)

    if not a.exp2_only:
        man = pd.read_csv(stim / "stimuli_exp1_manifest.csv")
        s1 = pd.read_csv(stim / "stimuli_exp1.csv")
        s1 = s1.assign(sha=s1.text.map(sha))
        m = man.merge(s1[KEY + ["sha"]], on=KEY, how="outer", indicator=True)
        missing, extra = (m._merge == "left_only").sum(), (m._merge == "right_only").sum()
        diff = m[(m._merge == "both") & (m.text_sha256 != m.sha)]
        print(f"Exp 1: {len(man)} manifest rows; missing {missing}, unexpected {extra}, text differs {len(diff)}")
        if len(diff):
            print("  first differing rows:", diff[KEY].head(10).to_dict("records"))
        ok &= missing == 0 and extra == 0 and len(diff) == 0

    if a.exp2_rebuilt:
        ref, new = pd.read_csv(a.exp2_reference), pd.read_csv(a.exp2_rebuilt)
        if ref.shape != new.shape:
            print(f"Exp 2 rebuild vs released table: DIFFERENT shape {new.shape} vs {ref.shape}")
            ok = False
        else:
            differ = int((ref.text.values != new.text.values).sum())
            print(f"Exp 2 rebuild vs released table: {differ} rows differ")
            if differ and new.text.str.contains("Ã").any():
                print("  the rebuild read UTF-8 as cp1252 ('café' -> 'cafÃ©'); on Windows run build_exp2.py "
                      "with `python -X utf8`")
            ok &= differ == 0
    s2 = pd.read_csv(stim / "stimuli_exp2.csv")
    print(f"Exp 2: {len(s2)} rows, {s2.speaker.nunique()} answers, versions {sorted(s2.condition.unique())}")
    print("OK: stimuli match the paper" if ok else "MISMATCH: see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
