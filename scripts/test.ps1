$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try { & (Join-Path $ProjectRoot '.venv\Scripts\python.exe') -m unittest discover -s tests -v; $TestExit=$LASTEXITCODE }
finally { Pop-Location }
exit $TestExit
