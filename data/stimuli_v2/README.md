# Model-ready stimuli

These are the files `stutter_rq_lmstudio.py` reads. `prepare_v2_stimuli.py --exp1-secondary --drop-poor-fit`
writes them from the two stimulus packages.

## stimuli_exp2.csv (in this repository, 1,680 rows)

- 120 written answers x 7 versions (F, S3, S8, S15, T3, T8, T15) x 2 roles (analyst, client).
- `question`: the interview question the answer was written for (12 questions).
- `excerpt` = answer id + role; `speaker` = answer id (the clustering unit).
- `pss` is the stuttering-like disfluency rate of the text itself (0 for F and T versions).
  `dose_pss` is the dose used in the analysis: S versions take their own %SD, and each T version takes the
  %SD of its matching S version, because its fillers are at the same positions and count.
  Median dose: 3.1, 8.1 and 15.0.

## stimuli_exp1.csv (NOT in this repository, 690 rows)

Derived from FluencyBank, which is password-protected under the TalkBank Ground Rules. Rebuild it with
`scripts/build_stimuli.ps1` or `scripts/build_stimuli.sh` and check it with `python verify_stimuli.py`.

`stimuli_exp1_manifest.csv` has every column of that file except `text`, plus `text_sha256`, so you can
see the design (speakers, excerpts, conditions C1-C6, questions, %SD, event counts) without the speech.

- Neutral set: 44 excerpts from 20 speakers, conditions C1-C6 (`topic_stutter = False`). This is the
  Experiment 1 set analysed in the paper.
- Stuttering-topic set: 71 excerpts (`topic_stutter = True`), run but analysed separately (not in the paper).
- `question_fit`: good (29) or partial (15).
