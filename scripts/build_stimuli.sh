#!/usr/bin/env bash
# Rebuild all stimuli from source and check them against the paper (Linux / macOS / Git Bash).
#   bash scripts/build_stimuli.sh /path/to/fluencybank      # Experiments 1 and 2 (needs FluencyBank access)
#   bash scripts/build_stimuli.sh                           # Experiment 2 only
# FluencyBank folder layout expected:
#   <fb>/Romana/TimeStamped/metadata.csv   and   <fb>/Valente/Valente/interview/*.cha
set -euo pipefail
PY="${PY:-python} -X utf8"
FB=${1:-}

cd exp2_stimuli_v2_package
$PY build_exp2.py --answers answers_part1.json answers_part2.json \
    --inventory stutter_inventory.json --out exp2_stimuli_v2_rebuilt.csv
cd ..

if [ -z "$FB" ]; then
  # data/stimuli_v2/stimuli_exp2.csv is already in the repository; this only checks the rebuild against it
  echo "No FluencyBank folder given: Experiment 1 skipped."
  $PY verify_stimuli.py --exp2-only --exp2-rebuilt exp2_stimuli_v2_package/exp2_stimuli_v2_rebuilt.csv
  exit 0
fi

cd exp1_stimuli_v2_package
$PY build_stimuli.py --romana "$FB/Romana/TimeStamped/metadata.csv" --valente "$FB/Valente/Valente/interview" \
    --old exp1_seed_segments.csv --out fluencybank_stimuli_v2.csv \
    --extra 507_013,507_021,44m_265,57m_059,43m_044 --labels review_labels.csv
$PY apply_second_screen.py --stimuli fluencybank_stimuli_v2.csv --labels second_screen_labels.csv
$PY rebuild_c5.py --romana "$FB/Romana/TimeStamped/metadata.csv" --valente "$FB/Valente/Valente/interview" \
    --stimuli fluencybank_stimuli_v2.csv --out fluencybank_stimuli_v2_c5v3.csv
cd ..
$PY prepare_v2_stimuli.py --exp1-secondary --drop-poor-fit --exp2 exp2_stimuli_v2_package/exp2_stimuli_v2_rebuilt.csv
$PY verify_stimuli.py --exp2-rebuilt exp2_stimuli_v2_package/exp2_stimuli_v2_rebuilt.csv
