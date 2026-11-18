#!/usr/bin/env python
"""paper_extras.py  --  numbers quoted in the paper that `stutter_rq_lmstudio.py analyze` does not print.

Reads only the released tidy tables, so it needs no stimuli and no model.

    python paper_extras.py                       # reads results_v3/, writes results_v3/extra_*.csv
    python paper_extras.py --tidy results_v3 --out results_check

Outputs
  extra_exp1_means.csv               Exp 1 mean primary ratings by condition C1-C6 (Sect. 5.2)
  extra_gap_by_model.csv             Exp 1 C1-C2 gap per model (Sect. 5.1)
  extra_exp2_means.csv               Exp 2 mean primary ratings by version (Fig. 2b)
  extra_exp2_hire_by_model.csv       Exp 2 hireability by model and version (Sect. 5.3)
  extra_exp2_mitigation_by_model.csv Exp 2 S15-F and T15-F gap per model and framing (Sect. 5.4)
  extra_agent_decisions.csv          shortlist rate per agent, experiment, framing and version (Sect. 5.5)
  extra_agent_pairs.csv              discordant shortlist pairs and exact McNemar p per agent (Sect. 5.5)
  extra_agent_vs_single.csv          A1 vs A2 gap per agent model on identical answers (Sect. 5.5)
  extra_agent_runs.csv               completion and step-11 reminder rates (Sect. 4, 5.5)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

PRIMARY = ["competence", "intelligence", "hireability"]
KEYS = ["model", "arch", "exp", "framing", "task", "speaker", "excerpt", "sample", "trait"]


def load(d):
    R, A = pd.read_csv(d / "tidy_ratings.csv"), pd.read_csv(d / "tidy_agent.csv")
    for df in (R, A):
        df["topic_stutter"] = df.topic_stutter.astype(str).str.lower().eq("true")
    return R[(R["sample"] == 0) & ~R.topic_stutter], A[(A["sample"] == 0) & ~A.topic_stutter]


def gap(df, a, b, by):
    """Mean paired difference a - b within answer, role and model."""
    k = [c for c in KEYS if c in df.columns]
    x = df[df.condition == a][k + ["rating"]].merge(df[df.condition == b][k + ["rating"]], on=k, suffixes=("_a", "_b"))
    x["d"] = x.rating_a - x.rating_b
    return x.dropna(subset=["d"]).groupby(by).d.mean()


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tidy", default="results_v3")
    ap.add_argument("--out", default=None, help="default: same folder as --tidy")
    a = ap.parse_args()
    tidy = Path(a.tidy)
    out = Path(a.out or a.tidy)
    out.mkdir(parents=True, exist_ok=True)
    R, A = load(tidy)
    prim = R[R.trait.isin(PRIMARY)]
    a1 = prim[(prim.arch == "A1") & (prim.task == "A") & (prim.framing == "F1")]

    # Exp 1 means and per-model gaps
    e1 = a1[a1.exp == 1]
    e1.pivot_table(index="trait", columns="condition", values="rating", aggfunc="mean").round(2) \
        .reset_index().rename_axis(None, axis=1).to_csv(out / "extra_exp1_means.csv", index=False)
    gap(e1, "C1", "C2", ["model", "trait"]).unstack().round(2).reset_index().rename_axis(None, axis=1) \
        .to_csv(out / "extra_gap_by_model.csv", index=False)

    # Exp 2 dose means
    e2 = a1[a1.exp == 2]
    order = ["F", "S3", "S8", "S15", "T3", "T8", "T15"]
    e2.pivot_table(index="trait", columns="condition", values="rating", aggfunc="mean")[order].round(2) \
        .reset_index().rename_axis(None, axis=1).to_csv(out / "extra_exp2_means.csv", index=False)
    h = e2[e2.trait == "hireability"]
    h.pivot_table(index="model", columns="condition", values="rating", aggfunc="mean")[order].round(2) \
        .reset_index().rename_axis(None, axis=1).to_csv(out / "extra_exp2_hire_by_model.csv", index=False)

    # Exp 2 mitigation per model
    m2 = prim[(prim.exp == 2) & (prim.task == "A")]
    rows = []
    for cmp_ in ("S15", "T15"):
        for arch in ("A1", "A2"):
            s = m2[m2.arch == arch]
            if cmp_ == "T15" and arch == "A2":
                continue
            g = gap(s, cmp_, "F", ["arch", "model", "trait", "framing"]).unstack()
            g = g.reindex(columns=["F1", "F2", "F3", "F4"]).round(2).reset_index().rename_axis(None, axis=1)
            rows.append(g.assign(cmp=f"{cmp_}-F"))
    pd.concat(rows).to_csv(out / "extra_exp2_mitigation_by_model.csv", index=False)

    # agent decisions
    done = A[A.completed == 1]
    dec = done.groupby(["exp", "framing", "model", "condition"]).agg(
        n=("shortlist", "size"), shortlist=("shortlist", "mean")).reset_index()
    dec["shortlist"] = dec.shortlist.round(3)
    dec.to_csv(out / "extra_agent_decisions.csv", index=False)

    pairs = []
    for (exp, fr, m), s in done.groupby(["exp", "framing", "model"]):
        st, fl = ("S15", "F") if exp == 2 else ("C1", "C2")
        p = s[s.condition == st][["excerpt", "shortlist"]].merge(
            s[s.condition == fl][["excerpt", "shortlist"]], on="excerpt", suffixes=("_s", "_f"))
        if not len(p):
            continue
        ps, pf = p.shortlist_s.astype(bool), p.shortlist_f.astype(bool)
        n10, n01, pv = mcnemar(pf, ps)
        lat = norm.ppf(np.clip(pf.mean(), 1e-3, 1 - 1e-3)) - norm.ppf(np.clip(ps.mean(), 1e-3, 1 - 1e-3))
        pairs.append(dict(exp=exp, framing=fr, model=m, contrast=f"{st} vs {fl}", n_pairs=len(p),
                          shortlist_stuttered=round(ps.mean(), 3), shortlist_fluent=round(pf.mean(), 3),
                          only_fluent=n10, only_stuttered=n01, mcnemar_p=pv, latent_shift_sd=round(lat, 2)))
    pd.DataFrame(pairs).to_csv(out / "extra_agent_pairs.csv", index=False)

    # A1 vs A2 on the same answers and models, per agent model
    vs = []
    for exp, (st, fl) in ((1, ("C1", "C2")), (2, ("S15", "F"))):
        s = prim[(prim.exp == exp) & (prim.task == "A") & (prim.framing == "F1")]
        agent_items = set(A[(A.exp == exp) & (A.framing == "F1")].excerpt)
        for m in sorted(A.model.unique()):
            sm = s[(s.model == m) & s.excerpt.isin(agent_items)]
            g = gap(sm, st, fl, ["arch", "trait"]).unstack("arch")
            for t, r in g.iterrows():
                vs.append(dict(exp=exp, model=m, trait=t, contrast=f"{st}-{fl}", A1=round(r.get("A1", np.nan), 2),
                               A2=round(r.get("A2", np.nan), 2), psi=round(r.get("A2", np.nan) - r.get("A1", np.nan), 2)))
    pd.DataFrame(vs).to_csv(out / "extra_agent_vs_single.csv", index=False)

    # run completion and the step-11 reminder (a run that needs more than 10 steps gets the reminder)
    runs = A.assign(reminder=A.n_steps.fillna(0) >= 11)
    runs["form"] = np.where(runs.condition.isin(["C1", "S15"]), "stuttered",
                            np.where(runs.condition.isin(["C2", "F"]), "fluent", "disclosure"))
    runs["experiment"] = np.where(runs.exp == 2, "Exp 2", "Exp 1")   # exp 3 = Exp 1 framings F3/F4
    sm = runs.groupby(["experiment", "form"]).agg(runs=("completed", "size"), completed=("completed", "mean"),
                                                  reminder=("reminder", "mean")).reset_index()
    tot = pd.DataFrame([dict(experiment="all", form="all", runs=len(runs), completed=runs.completed.mean(),
                             reminder=runs.reminder.mean())])
    pd.concat([sm, tot]).round(4).to_csv(out / "extra_agent_runs.csv", index=False)
    print(f"incomplete agent runs: {int((runs.completed == 0).sum())} of {len(runs)}")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
