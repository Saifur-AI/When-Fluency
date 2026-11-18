# FluencyBank stimuli v2 (Experiment 1)

Built from FluencyBank Timestamped (Romana segment file, 37 speakers) and the Valente interview codes (36 speakers), with the 22 excerpts of v1 kept as seeds.

## Use

Filter `exp1_primary == True` for the canonical Experiment 1 set (46 excerpts, 20 speakers, 276 rows).
The other 80 excerpts are kept for reference and an optional secondary analysis (`exclusion_reason` says why).

## Selection rules

- Windows of consecutive speaker segments, gap 3 s or less, 40 to 180 fluent words, at least one Valente core event.
- Segments where the speaker reads an interview prompt, or mentions stuttering, speech, therapy or fluency, break a window.
- Every new window was read by hand (`neutral_review`): `clean`, `implicit` (speaking or communication theme, no stuttering cue) or `stuttering_content` (causes, therapy, listener advice, counterfactuals about stuttering). Only `clean` and `implicit` enter the primary set.
- The three v1 excerpts that answer a stuttering question (508#01, 50fa#03, 60m#02) are dropped from the primary set. For new windows, `question_mentions_stuttering` is kept as a moderator.
- Second screen: windows that overlap the earlier pipeline's rejected list carry its reason in `second_screen`. Of 42 windows I judged to be about stuttering, the earlier screen also rejected all 42. Three windows I had kept were flagged by its LLM screen (507#n033, 54m#n012, 68m#n064) and are dropped, so an excerpt enters the primary set only if neither screen flags it. Its `no_topic` rejections come from a topic filter, not a neutrality judgement, so they do not exclude.
- Cap of 5 primary excerpts per speaker (v1 excerpts first, then clean, then longer).

## Versions

- C1 Romana `verbatim`, with Valente events rendered: part-word and word repetitions as `b-b-book`, blocks as `b... book`, prolongations as `bbbook`, multisyllabic repetitions as `book, book`. Typical revisions end with a comma; fillers stay as spoken.
- C2 Romana `intended` (fluent).
- C3, C4: C1 and C2 with "Before we start, I should mention that I stutter."
- C5: C2 with alternating "um" / "uh" at the positions of C1's stuttering events.
- C6: C1 with "Before we start, I should mention that I'm left-handed."

## Measures

- `pss` = Valente core events (SR, ISR, MUR, P, B) in the excerpt's time span / syllables in C2 x 100 (CMU dictionary syllables).
- `readability` = Flesch reading ease of that version.
- `topic` is keyword-based and approximate.

## Alignment

895 Valente events fell inside selected excerpts: 534 matched to a disfluency written in the verbatim transcript, 353 placed on the nearest word by timing (mostly blocks and prolongations, which leave no trace in a word transcript), 8 could not be placed. Events placed by timing are rendered with the conventions above, so C1 shows every coded stutter. Per-excerpt counts are in `fluencybank_stimuli_v2_alignment_log.csv`.

## Rebuild

The stimulus table itself is not in this repository (FluencyBank text). It is rebuilt from the
FluencyBank files with the text-free inputs in this folder:

| File | Content |
| --- | --- |
| `exp1_seed_segments.csv` | segment ids of the 22 excerpts kept from v1 (corpus question wording replaced by a placeholder; only "mentions stuttering" is used) |
| `review_labels.csv` | hand topic review of new windows |
| `second_screen_labels.csv` | result of the second, independent topic screen |
| `exp1_question_map.csv` | interview question matched to each neutral excerpt, with fit rating |
| `fluencybank_stimuli_v2_alignment_log.csv` | event alignment counts per excerpt |

```
python -X utf8 build_stimuli.py --romana <fb>/Romana/TimeStamped/metadata.csv --valente <fb>/Valente/Valente/interview \
  --old exp1_seed_segments.csv --out fluencybank_stimuli_v2.csv \
  --extra 507_013,507_021,44m_265,57m_059,43m_044 --labels review_labels.csv
python -X utf8 apply_second_screen.py --stimuli fluencybank_stimuli_v2.csv --labels second_screen_labels.csv
python -X utf8 rebuild_c5.py --romana <fb>/Romana/TimeStamped/metadata.csv --valente <fb>/Valente/Valente/interview \
  --stimuli fluencybank_stimuli_v2.csv --out fluencybank_stimuli_v2_c5v3.csv
```

Then run `prepare_v2_stimuli.py --exp1-secondary --drop-poor-fit` and `verify_stimuli.py` from the repository root.
