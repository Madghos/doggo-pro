
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Scene = Join-Path $Root "dog_sim\scene.xml"

Write-Host "Python: $Python"
Write-Host "Scene:  $Scene"

if (-not (Test-Path $Python)) {
    throw "Nie znaleziono Pythona: $Python"
}

if (-not (Test-Path $Scene)) {
    throw "Nie znaleziono sceny: $Scene"
}

$Viewer = Join-Path $Root "dog_sim\viewer.py"
& $Python $Viewer

if ($LASTEXITCODE -ne 0) {
    throw "MuJoCo zakonczyl dzialanie z bledem: $LASTEXITCODE"
}
