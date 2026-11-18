# Rebuild all stimuli from source and check them against the paper (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\build_stimuli.ps1 -FluencyBank D:\fluencybank   # Exp 1 and 2
#   powershell -ExecutionPolicy Bypass -File scripts\build_stimuli.ps1                               # Exp 2 only
# FluencyBank folder layout expected:
#   <fb>\Romana\TimeStamped\metadata.csv   and   <fb>\Valente\Valente\interview\*.cha
# -X utf8 matters on Windows: without it the answer files are read as cp1252 and two texts change.
param([string]$FluencyBank = "")
$py = "python"
$ErrorActionPreference = "Stop"

Push-Location exp2_stimuli_v2_package
& $py -X utf8 build_exp2.py --answers answers_part1.json answers_part2.json --inventory stutter_inventory.json --out exp2_stimuli_v2_rebuilt.csv
Pop-Location

if ($FluencyBank -eq "") {
    Write-Output "No FluencyBank folder given: Experiment 1 skipped."
    & $py -X utf8 verify_stimuli.py --exp2-only --exp2-rebuilt exp2_stimuli_v2_package\exp2_stimuli_v2_rebuilt.csv
    exit $LASTEXITCODE
}

$romana = Join-Path $FluencyBank "Romana\TimeStamped\metadata.csv"
$valente = Join-Path $FluencyBank "Valente\Valente\interview"
Push-Location exp1_stimuli_v2_package
& $py -X utf8 build_stimuli.py --romana $romana --valente $valente --old exp1_seed_segments.csv --out fluencybank_stimuli_v2.csv --extra 507_013,507_021,44m_265,57m_059,43m_044 --labels review_labels.csv
& $py -X utf8 apply_second_screen.py --stimuli fluencybank_stimuli_v2.csv --labels second_screen_labels.csv
& $py -X utf8 rebuild_c5.py --romana $romana --valente $valente --stimuli fluencybank_stimuli_v2.csv --out fluencybank_stimuli_v2_c5v3.csv
Pop-Location
& $py -X utf8 prepare_v2_stimuli.py --exp1-secondary --drop-poor-fit --exp2 exp2_stimuli_v2_package\exp2_stimuli_v2_rebuilt.csv
& $py -X utf8 verify_stimuli.py --exp2-rebuilt exp2_stimuli_v2_package\exp2_stimuli_v2_rebuilt.csv
exit $LASTEXITCODE
