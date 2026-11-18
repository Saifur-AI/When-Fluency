"""Apply the second topic screen to the Experiment 1 table written by build_stimuli.py.

build_stimuli.py applies the hand review (review_labels.csv). A second, independent screen flagged three
windows that the hand review had kept (507#n033, 54m#n012, 68m#n064). An excerpt enters the primary set only
if neither screen flags it. This script adds the `second_screen` column from second_screen_labels.csv and
drops the flagged excerpts from the primary set.

  python apply_second_screen.py --stimuli fluencybank_stimuli_v2.csv --labels second_screen_labels.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", default="fluencybank_stimuli_v2.csv")
    ap.add_argument("--labels", default="second_screen_labels.csv")
    ap.add_argument("--out", default=None, help="default: overwrite --stimuli")
    a = ap.parse_args()

    st = pd.read_csv(a.stimuli, encoding="utf-8-sig")
    lab = pd.read_csv(a.labels).set_index("excerpt_id").second_screen
    st["second_screen"] = st.excerpt_id.map(lab)
    primary = st.exp1_primary.astype(str).str.lower().eq("true")
    flagged = primary & st.second_screen.eq("llm_screen_stuttering")
    st.loc[flagged, "exp1_primary"] = False
    st.loc[flagged, "exclusion_reason"] = "flagged by second screen"
    st.to_csv(a.out or a.stimuli, index=False, encoding="utf-8-sig")
    print(f"second screen: {st[flagged].excerpt_id.nunique()} excerpts removed from the primary set; "
          f"primary now {st[st.exp1_primary.astype(str).str.lower().eq('true')].excerpt_id.nunique()} excerpts")


if __name__ == "__main__":
    main()
