# When Fluency Becomes a Proxy for Competence: Disparate Impact in LLM Hiring

Code, stimuli, model outputs and analysis for the paper (PAKDD 2027 submission).

We audit five open-weight LLMs as interview screeners. Counterfactual text operators change the *form* of a
candidate's answer (stuttering-like events, fillers at the same positions, a disclosure sentence) but not its
content. Paired contrasts then split the stuttering penalty into a generic disfluency part and a
stuttering-specific part. The audit covers 37,964 model outputs on two stimulus sets:

* **Experiment 1:** 44 real interview excerpts from 20 adults who stutter (FluencyBank), in six conditions C1–C6.
* **Experiment 2:** 120 written answers with controlled disfluency doses of about 3, 8 and 15 %SD
  (versions F, S3/S8/S15, T3/T8/T15).

Every number in the paper can be recomputed from this repository without a GPU, a model or FluencyBank access
(see [Quick start](#quick-start-reproduce-the-tables-no-model-needed)). The full pipeline, from raw
FluencyBank files through model calls to tables, is also here. We checked that it rebuilds the paper's
stimuli byte for byte and regenerates all 37,964 job ids.

## Contents

```
stutter_rq_lmstudio.py      the whole pipeline: prompts, LM Studio client, agent, measures, statistics, figures
paper_extras.py             numbers quoted in the paper that `analyze` does not print (per-model, agent decisions)
prepare_v2_stimuli.py       stimulus packages -> model-ready tables in data/stimuli_v2
verify_stimuli.py           checks rebuilt stimuli against the paper's (SHA-256 for Experiment 1)
scripts/
  build_stimuli.ps1 / .sh   rebuild all stimuli from source and verify them
  run_all.ps1 / .sh         every model call, in the order used for the released outputs
exp1_stimuli_v2_package/    Experiment 1 builder (FluencyBank -> C1-C6) and its text-free inputs
exp2_stimuli_v2_package/    Experiment 2 answers, stuttering inventory, builder and the built table
data/stimuli_v2/
  stimuli_exp2.csv          Experiment 2 stimuli as given to the models
  stimuli_exp1_manifest.csv Experiment 1 design without the text (text_sha256 instead)
data/runs_v3/
  results.jsonl.gz          every model call (37,964 records; see "Data and privacy")
  judge.jsonl.gz            LLM-judge stereotype labels (25,020)
results_v3/
  tidy_ratings.csv          one row per output x trait (ratings, limited-evidence flag)
  tidy_texts.csv            one row per single-call text (stereotype lexicon, judge label, speech mention)
  tidy_agent.csv            one row per agent run (decision, steps, trace measures)
  rq*.csv, RQ*.png          tables and figures per research question
  extra_*.csv               output of paper_extras.py
  SUMMARY.md                all headline estimates in one file
```

## Quick start: reproduce the tables (no model needed)

```bash
pip install -r requirements.txt
python stutter_rq_lmstudio.py analyze --from-tidy results_v3 --out results_check
python paper_extras.py --tidy results_v3 --out results_check
```

This rebuilds every `rq*.csv`, `RQ*.png`, `extra_*.csv` and `SUMMARY.md` from the three tidy tables, with the
same cluster bootstrap (B = 5000, fixed seed). We checked that all 24 `rq*.csv` tables and `SUMMARY.md` come
out identical to the released `results_v3/`.

## Where each result in the paper comes from

| Paper | File(s) in `results_v3/` |
| --- | --- |
| Sect. 5.1, penalty C1 − C2, intervals, Holm p, d_z | `rq1_c1_c2.csv` (rows `model = pooled`, `arch = A1`), `rq1_mixed_model.csv` |
| Sect. 5.1, per-model gap, fluent means, limited-evidence rate | `extra_gap_by_model.csv`, `extra_exp1_means.csv`, `rq1_insufficient_rate.csv` |
| Sect. 5.1, stereotype language (judge, lexicon, first impression), term counts | `rq2_c1_c2.csv`, `rq2_rate_by_condition.csv`, `rq2_terms.csv` |
| Sect. 5.2, disclosure means and contrasts | `extra_exp1_means.csv`, `rq3_contrasts.csv`, `rq3_cell_means.csv` |
| Sect. 5.2, agent shortlists after disclosure | `extra_agent_decisions.csv` (exp 1, C4 vs C2) |
| Fig. 2a, decomposition and generic shares | `rq4_specificity.csv` |
| Fig. 2b, Eq. (2), dose slopes β_T and β_S | `extra_exp2_means.csv`, `rq4_dose_slopes.csv` |
| Sect. 5.3, per-model dose response | `extra_exp2_hire_by_model.csv` |
| Table 1, framings on Experiment 2 (TOST, 90% CI) | `rq5_tost_exp2.csv`, `rq5_verdict_exp2.csv`; fillers: `rq5_tost_exp2_fillers.csv` |
| Sect. 5.4, framings per model | `extra_exp2_mitigation_by_model.csv` |
| Table 2, A1 vs A2 and Ψ | `rq6_gap_by_arch*.csv`, `rq6_did.csv` (Exp. 1), `rq6_did_exp2.csv` (Exp. 2) |
| Table 2, trace rows | `rq6_trace.csv`, `rq6_trace_exp2.csv` |
| Sect. 5.5, shortlist rates, McNemar, latent shift, gpt-oss alone | `extra_agent_pairs.csv`, `extra_agent_vs_single.csv`, `extra_agent_decisions.csv` |
| Sect. 4 and 5.5, incomplete runs (5 of 2,144) and step-11 reminder | `extra_agent_runs.csv` |
| Experiment 1 framings (baseline already inside ±0.3) | `rq5_tost.csv`, `rq5_verdict.csv` |

Coding used in the tables:

* `exp = 1` is Experiment 1 under F1, `exp = 3` is Experiment 1 under framings F2–F4, and `exp = 2` is Experiment 2.
* `arch = A1` is a single call and `A2` is the tool-using agent. `task = A` is the hiring assessment and `B` is the first impression.
* Differences are disfluent minus fluent, so a negative value is a penalty. For nervousness, higher means worse.

## Full reproduction

### 1. Stimuli

The Experiment 2 stimuli are in the repository, so nothing needs to be built for it. To rebuild them from the
written answers and check the result anyway:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_stimuli.ps1       # Windows
bash scripts/build_stimuli.sh                                            # Linux / macOS
```

Experiment 1 needs FluencyBank. Request access at <https://fluency.talkbank.org>, then download two
things: the **Valente** adult interview CHAT files, which carry the `%xstu` stuttering-event tier, and the
**FluencyBank Timestamped** segment table (Romana et al., 2024), which has `metadata.csv` with verbatim and
intended text and word times. Arrange them as:

```
<fb>/Romana/TimeStamped/metadata.csv
<fb>/Valente/Valente/interview/*.cha
```

and run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_stimuli.ps1 -FluencyBank <fb>
bash scripts/build_stimuli.sh <fb>
```

This builds C1–C6 for 126 candidate windows. It then applies the two topic screens, rebuilds C5 (each
stuttering-like event replaced by exactly one filler) and writes `data/stimuli_v2/stimuli_exp1.csv`.
`verify_stimuli.py` should print `OK: stimuli match the paper`, meaning all 690 Experiment 1 texts have the
expected SHA-256.

On Windows, keep the `-X utf8` flag that the scripts pass. Without it, Python reads the answer files as
cp1252 and two Experiment 2 answers change ("café").

### 2. Models

All calls go to [LM Studio](https://lmstudio.ai)'s OpenAI-compatible server at `http://localhost:1234/v1`.
The models are:

| Short name | LM Studio model id | Role |
| --- | --- | --- |
| `qwen` | `qwen/qwen3-8b` | rater, agent, stereotype judge |
| `llama` | `meta-llama-3.1-8b-instruct` | rater |
| `mistral` | `mistralai/mistral-7b-instruct-v0.3` | rater |
| `gemma` | `google/gemma-3-4b` | rater |
| `gptoss` | `openai/gpt-oss-20b` | rater, agent |

Settings are in the `CONFIGURATION` block at the top of `stutter_rq_lmstudio.py`: temperature 0, seed
20271, `max_tokens` 1200, at most 12 agent steps, and `/no_think` appended for Qwen3. Check that the server
is reachable with

```bash
python stutter_rq_lmstudio.py check
```

### 3. Run, judge, analyze

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1    # or: bash scripts/run_all.sh
```

The script runs one model at a time, in the order used for the paper:

1. Single calls.
2. Agents on Experiment 1.
3. The stereotype judge.
4. The Experiment 2 framing and agent runs.
5. `analyze` and `paper_extras.py`.

Each finished call is appended to `data/runs_v3/results.jsonl`, and each model response is cached in
SQLite under a SHA-256 of the request. Re-running the script therefore skips finished work.

Job ids are hashes of the prompt version, condition and stimulus text, so rebuilt stimuli give the same
37,964 ids as the released outputs. To compare a fresh run with ours, decompress the released file and
match records on `job_id`:

```bash
gunzip -k data/runs_v3/results.jsonl.gz data/runs_v3/judge.jsonl.gz
```

With temperature 0 and a fixed seed, outputs should be close to ours, but exact identity depends on the
quantisation, LM Studio version and hardware.

Running `analyze` on the full `results.jsonl` needs both stimulus files, because the stereotype lexicon and
the evidence-grounding check compare each output with the fluent transcript.

## Data and privacy

* **FluencyBank text is not redistributed.** The FluencyBank transcripts are password-protected under the
  [TalkBank Ground Rules](https://talkbank.org/share/rules.html). This repository therefore contains no
  Experiment 1 stimulus text: only segment ids, review labels, the question map and SHA-256 hashes.
* **Experiment 1 free-text outputs are redacted.** Model assessments, first impressions, agent notes,
  justifications, concerns, traces and exact-quote evidence can repeat the speaker's words. For Experiment 1
  records (`exp` 1 and 3), these fields are replaced by a placeholder in `results.jsonl.gz` and in
  `tidy_texts.csv`. That covers 8,924 of the 37,964 records.
* **Every derived measure is kept**: ratings, limited-evidence flags, decisions, judge labels, lexicon hits,
  speech mentions and trace measures. The analysis does not read the redacted text, which is why the
  quick start reproduces every table exactly.
* **Experiment 2 is released in full.** Its answers were written for this study and contain no FluencyBank
  speech; only event types and repetition counts were sampled from the FluencyBank codings
  (`exp2_stimuli_v2_package/stutter_inventory.json`).
* Researchers with FluencyBank access can rebuild the Experiment 1 stimuli as described above.
* **Human annotation.** The two annotation tasks (the stimulus meaning-preservation check and the
  stereotype labels on 140 outputs) and their agreement statistics are reported in the paper. Their label
  files are not yet part of this repository.

## Notes on the method

* **Operators.** In C1 the original transcript is rendered with repetitions as `s-s-set`, prolongations as
  `ssso` and blocks as `p... project`. C2 removes the stuttering-like events. C5 replaces each event with
  one filler (alternating *um* / *uh*). C3 and C4 add "Before we start, I should mention that I stutter."
  C6 adds "...that I'm left-handed."
* **Disfluency rate.** `pss` is the percentage of syllables disfluent (%SD), counted with CMU dictionary
  syllables. For stuttering-like events it equals clinical %SS.
* **Inference.** Contrasts are paired within excerpt (or answer), role and model. Intervals come from a
  cluster bootstrap over speakers (Experiment 1) or answers (Experiment 2): percentile intervals,
  B = 5000. Holm correction is applied over the three primary traits, McNemar tests are used for binary
  measures, and TOST uses δ = 0.3 with 90% intervals.
* **Dose model (Eq. 2).** A mixed model with model fixed effects and item random intercepts, fitted with
  statsmodels `MixedLM`.

## Citation

```bibtex
@inproceedings{anonymous2027fluency,
  title     = {When Fluency Becomes a Proxy for Competence: Disparate Impact in {LLM} Hiring},
  author    = {Anonymous},
  booktitle = {Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year      = {2027},
  note      = {Under review}
}
```

If you use the stimuli, please also cite FluencyBank (Bernstein Ratner & MacWhinney, 2018) and the
FluencyBank Timestamped release (Romana et al., 2024), and acknowledge NIH NIDCD R01-DC015494, as the
TalkBank Ground Rules require.

## License

Code: MIT. Experiment 2 answers, model outputs and derived tables: CC BY-NC-SA 4.0. The non-commercial,
share-alike licence matches FluencyBank, since the Experiment 2 events are sampled from its codings.
