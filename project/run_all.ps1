param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",
    [switch]$SkipTests,
    [switch]$DryRun,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @("$PSScriptRoot\scripts\run_all.py", "--mode", $Mode)

if ($SkipTests) { $arguments += "--skip-tests" }
if ($DryRun) { $arguments += "--dry-run" }
if ($Resume) { $arguments += "--resume" }

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Task 1 pipeline failed with exit code $LASTEXITCODE"
}
