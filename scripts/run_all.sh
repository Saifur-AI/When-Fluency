#!/usr/bin/env bash
# Full model run for the paper, in the order used for the released outputs (Linux / macOS).
# Needs LM Studio serving the five models (see MODELS in stutter_rq_lmstudio.py) and both stimulus files in
# data/stimuli_v2. One model at a time, so LM Studio does not swap models between calls.
# Resumable: re-running skips every job already in data/runs_v3/results.jsonl.
# Run from the repository root:  bash scripts/run_all.sh
set -euo pipefail
PY=${PY:-python}
MODELS="qwen llama mistral gemma gptoss"
AGENTS="qwen gptoss"

# 1. Experiments 1 and 2, framing F1, plus Experiment 1 framings F2-F4 (single call, A1)
for m in $MODELS; do echo "=== A1 $m $(date +%T)"; $PY stutter_rq_lmstudio.py run --models "$m" --no-agent; done
# 2. Hiring agent (A2) on Experiment 1
for m in $AGENTS; do echo "=== A2 agent $m $(date +%T)"; $PY stutter_rq_lmstudio.py run --models "$m" --agent-models "$m"; done
# 3. Stereotype judge (Qwen3 8B) on all single-call texts so far
echo "=== judge $(date +%T)"; $PY stutter_rq_lmstudio.py judge --judge qwen
# 4. Experiment 2 mitigation: F2-F4 on F/S15/T15 (A1), and agents under F1/F3/F4 on F/S15
for m in $MODELS; do echo "=== A1 Exp 2 framings $m $(date +%T)"; $PY stutter_rq_lmstudio.py run --models "$m" --no-agent --exp2-mitigation; done
for m in $AGENTS; do echo "=== A2 Exp 2 agent $m $(date +%T)"; $PY stutter_rq_lmstudio.py run --models "$m" --agent-models "$m" --exp2-mitigation; done
# 5. Tables, figures and the extra numbers quoted in the paper
echo "=== analyze $(date +%T)"
$PY stutter_rq_lmstudio.py analyze
$PY paper_extras.py
echo "=== done $(date +%T)"
