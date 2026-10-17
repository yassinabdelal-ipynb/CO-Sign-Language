param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Predictor,
    [string]$Roi,
    [ValidateSet('cpu','cuda')][string]$Device='cpu',
    [ValidateSet('dark','light')][string]$Theme='dark'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path -Parent $PSScriptRoot
$RunArgs=@((Join-Path $ProjectRoot 'realtime_detection.py'),'--checkpoint',$Checkpoint,'--device',$Device,'--theme',$Theme)
if ($Predictor) { $RunArgs+=@('--predictor',$Predictor) }
elseif ($Roi) {
    $Values = $Roi.Split(',')
    if ($Values.Count -ne 4) { throw 'Roi must be four comma-separated integers: X,Y,WIDTH,HEIGHT' }
    $ParsedRoi = @($Values | ForEach-Object { [int]$_ })
    $RunArgs+=@('--roi')+$ParsedRoi
}
else { throw 'Supply -Predictor PATH or -Roi X,Y,WIDTH,HEIGHT (explicit approximate manual crop).' }
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') @RunArgs
exit $LASTEXITCODE
