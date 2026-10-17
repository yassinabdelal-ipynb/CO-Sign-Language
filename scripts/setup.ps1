param(
    [string]$Python = 'python',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cpu',
    [switch]$Collection,
    [switch]$Training
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" } }
& $Python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12 for pinned collection/training compatibility"'
Check-Exit
$LocalPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $LocalPython)) { & $Python -m venv (Join-Path $ProjectRoot '.venv'); Check-Exit }
$TorchIndex = if ($Backend -eq 'cuda') { 'https://download.pytorch.org/whl/cu130' } else { 'https://download.pytorch.org/whl/cpu' }
& $LocalPython -m pip install --index-url $TorchIndex 'torch==2.14.1'
Check-Exit
& $LocalPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
Check-Exit
if ($Collection) {
    & $LocalPython -m pip install -r (Join-Path $ProjectRoot 'requirements-collection.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Collection install failed. dlib may require manually installed MSVC Build Tools/CMake. Core CPU commands remain usable. No compiler/driver was installed by this script.' }
}
if ($Training) { & $LocalPython -m pip install -r (Join-Path $ProjectRoot 'requirements-training.txt'); Check-Exit }
$LocalFolder = Join-Path $ProjectRoot 'local'
New-Item -ItemType Directory -Force -Path $LocalFolder | Out-Null
& $LocalPython -m pip freeze | Set-Content -LiteralPath (Join-Path $LocalFolder 'resolved-requirements.txt') -Encoding UTF8
Check-Exit
Write-Host "Ready: $LocalPython (no activation required)"
