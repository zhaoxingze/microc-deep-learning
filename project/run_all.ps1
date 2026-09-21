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
    $logDirectory = Join-Path $PSScriptRoot "outputs\logs"
    $latestLog = Get-ChildItem -LiteralPath $logDirectory -Filter "run_all_*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    $message = "Task 1 pipeline failed with exit code $LASTEXITCODE"
    if ($null -ne $latestLog) {
        $message += ". See the real failure in: $($latestLog.FullName)"
    }
    throw $message
}
