# Full model run for the paper, in the order used for the released outputs (Windows PowerShell).
# Needs LM Studio serving the five models (see MODELS in stutter_rq_lmstudio.py) and both stimulus files in
# data/stimuli_v2. One model at a time, so LM Studio does not swap models between calls.
# Resumable: re-running skips every job already in data/runs_v3/results.jsonl.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
$py = "python"
$models = @("qwen", "llama", "mistral", "gemma", "gptoss")
$agents = @("qwen", "gptoss")

# 1. Experiments 1 and 2, framing F1, plus Experiment 1 framings F2-F4 (single call, A1)
foreach ($m in $models) {
    Write-Output "=== A1 $m  $(Get-Date -Format T)"
    & $py stutter_rq_lmstudio.py run --models $m --no-agent
}
# 2. Hiring agent (A2) on Experiment 1
foreach ($m in $agents) {
    Write-Output "=== A2 agent $m  $(Get-Date -Format T)"
    & $py stutter_rq_lmstudio.py run --models $m --agent-models $m
}
# 3. Stereotype judge (Qwen3 8B) on all single-call texts so far
Write-Output "=== judge  $(Get-Date -Format T)"
& $py stutter_rq_lmstudio.py judge --judge qwen
# 4. Experiment 2 mitigation: F2-F4 on F/S15/T15 (A1), and agents under F1/F3/F4 on F/S15
foreach ($m in $models) {
    Write-Output "=== A1 Exp 2 framings $m  $(Get-Date -Format T)"
    & $py stutter_rq_lmstudio.py run --models $m --no-agent --exp2-mitigation
}
foreach ($m in $agents) {
    Write-Output "=== A2 Exp 2 agent $m  $(Get-Date -Format T)"
    & $py stutter_rq_lmstudio.py run --models $m --agent-models $m --exp2-mitigation
}
# 5. Tables, figures and the extra numbers quoted in the paper
Write-Output "=== analyze  $(Get-Date -Format T)"
& $py stutter_rq_lmstudio.py analyze
& $py paper_extras.py
Write-Output "=== done  $(Get-Date -Format T)"
