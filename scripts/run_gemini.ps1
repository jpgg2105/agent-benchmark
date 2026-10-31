# Runs the HotpotQA benchmark on Gemini's free tier, start to finish.
# Safe to run again: it skips what's already set up and resumes where the
# daily quota stopped it. Run it once a day until it prints the leaderboard
# with no errors.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1            # full run / resume
#   powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1 -Limit 5   # 5 questions first
param([int]$Limit = 0)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Failed: $exe $($argv -join ' ')" }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python not found. Install Python 3.11 or newer from python.org (tick 'Add to PATH')." }
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.10") { throw "Python $ver found; 3.10 or newer is needed." }
if (-not (Test-Path "..\research-agent\pyproject.toml")) { throw "Expected the research-agent repo next to this one (..\research-agent)." }

if (-not (Test-Path ".venv")) { Write-Host "Creating the virtual environment..."; Run python @("-m", "venv", ".venv") }
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path ".venv\.installed")) {
    Write-Host "Installing dependencies..."
    Run $py @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
    Run $py @("-m", "pip", "install", "--quiet", "-e", "..\research-agent[openai]", "-e", ".[dev]")
    Write-Host "Running the tests..."
    Run $py @("-m", "pytest", "-q")
    New-Item ".venv\.installed" -ItemType File | Out-Null
}

if (-not (Test-Path "data\hotpotqa_dev100.jsonl")) {
    Write-Host "Downloading 100 HotpotQA questions (takes a few minutes)..."
    Run $py @("scripts\prepare_hotpotqa.py")
    Write-Host "Dry run with a fake model..."
    Run $py @("-m", "agent_benchmark.cli", "run", "--config", "configs\hotpotqa-offline.yaml", "--limit", "5")
}

if (-not ((Test-Path ".env") -and (Select-String -Path ".env" -Pattern "^GEMINI_API_KEY=." -Quiet))) {
    $key = Read-Host "Paste your Gemini API key (aistudio.google.com/apikey)"
    Add-Content ".env" "GEMINI_API_KEY=$key"
}

$argv = @("-m", "agent_benchmark.cli", "run", "--config", "configs\hotpotqa-gemini.yaml", "--resume")
if ($Limit -gt 0) { $argv += @("--limit", "$Limit") }
Write-Host "Running the benchmark (it waits between calls to stay under the free quota)..."
Run $py $argv
Write-Host "`nDone for now. Results are in the results folder."
