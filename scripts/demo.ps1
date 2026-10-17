$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') (Join-Path $ProjectRoot 'realtime_detection.py') --events (Join-Path $ProjectRoot 'demos\events.jsonl')
exit $LASTEXITCODE
