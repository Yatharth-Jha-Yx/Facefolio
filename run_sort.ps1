# Runs sort_faces.py and restarts it whenever it skipped a hung file (exit code 3).
# Usage: .\run_sort.ps1 <input> <output> [sort_faces.py options...]
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "sort_faces.py"
do {
    & $py $script @args
    $code = $LASTEXITCODE
    if ($code -eq 3) { Write-Host "Restarting after skipping a hung file..." }
} while ($code -eq 3)
exit $code
