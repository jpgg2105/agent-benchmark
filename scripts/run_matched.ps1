# Reruns the baseline with the agent's rule on missing information (no invitation to abstain),
# then compares it with the agent's published answers. About 200 Gemini calls, ~20 minutes.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\run_matched.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts\run_gemini.ps1 once first to set up .venv" }
if (-not ((Test-Path ".env") -and (Select-String -Path ".env" -Pattern "^GEMINI_API_KEY=." -Quiet))) { throw "Put GEMINI_API_KEY=... in .env" }
& $py -m agent_benchmark.cli run --config configs\hotpotqa-gemini-matched.yaml --resume
if ($LASTEXITCODE -ne 0) { throw "The run stopped. If it hit the daily quota, run this script again tomorrow; it resumes." }
$latest = Get-ChildItem results\results-*.json | Sort-Object LastWriteTime | Select-Object -Last 1
& $py scripts\compare_matched.py results\published\hotpotqa-gemini-2026-10-31.json $latest.FullName
