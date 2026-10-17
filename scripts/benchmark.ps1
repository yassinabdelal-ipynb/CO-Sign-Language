param([Parameter(Mandatory=$true)][string]$Checkpoint,[ValidateSet('cpu','cuda')][string]$Device='cpu')
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path -Parent $PSScriptRoot
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'verify_checkpoint.py') --checkpoint $Checkpoint --device $Device --output (Join-Path $ProjectRoot "local\checkpoint-$Device.json")
exit $LASTEXITCODE
