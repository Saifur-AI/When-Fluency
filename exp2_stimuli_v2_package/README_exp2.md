# Experiment 2 stimuli v2 (job answers with transplanted stuttering)

120 written answers (X2-Q01-01 to X2-Q12-10), 12 interview questions x 10 answers, 53 to 102 words, varied jobs, no mention of speech or stuttering.

## Versions (7 per answer, 840 rows)

| Code | Content |
| --- | --- |
| F | Fluent original |
| S3, S8, S15 | Stuttered at about 3, 8 and 15 %SS (achieved: 2.4 to 3.6, 7.4 to 8.5, 14.5 to 15.6) |
| T3, T8, T15 | "um" / "uh" at the same positions and count as the matching S version |

- Stutter positions are nested: S3 events are a subset of S8, which are a subset of S15, so the dose contrast only adds stuttering.
- Event types and repetition counts are sampled from the Valente-coded events in FluencyBank (`stutter_inventory.json`). Achieved mix at S15: ISR 36%, B 32%, SR 16%, P 11%, MUR 5% (corpus: 33%, 30%, 15%, 17%, 5%). Prolongations are lower because they are only placed on sounds that can be held (s, f, m, n, l, vowels and similar).
- About 17% of stutters fall on function words, matching the corpus; neighbouring words are not both stuttered where avoidable.
- Rendering matches Experiment 1: `b-b-book`, `ba-baby`, `b... book`, `sssso`, `book, book`.

## Columns

`answer_id, question_id, question, version, version_label, level, text, positions` (word indices of events), `event_types` (index:type x repetitions), `n_events, syllables_fluent, pss_actual, word_count, readability, seed, source`.

## Run plan

Each answer under both roles (office analyst, client-facing), one fresh call per version, temperature 0:
120 answers x 7 versions x 2 roles x 5 models = 8,400 calls for Task A (the same again for Task B). The same fixed seed (2027) rebuilds identical stimuli.

## Rebuild

```
python3 build_exp2.py --answers answers_part1.json answers_part2.json \
  --inventory stutter_inventory.json --out exp2_stimuli_v2.csv
```
