param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Predictor,
    [int[]]$Roi,
    [ValidateSet('cpu','cuda')][string]$Device='cpu',
    [ValidateSet('dark','light')][string]$Theme='dark'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path -Parent $PSScriptRoot
$RunArgs=@((Join-Path $ProjectRoot 'realtime_detection.py'),'--checkpoint',$Checkpoint,'--device',$Device,'--theme',$Theme)
if ($Predictor) { $RunArgs+=@('--predictor',$Predictor) }
elseif ($Roi.Count -eq 4) { $RunArgs+=@('--roi')+$Roi }
else { throw 'Supply -Predictor PATH or -Roi X,Y,WIDTH,HEIGHT (explicit approximate manual crop).' }
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') @RunArgs
exit $LASTEXITCODE
