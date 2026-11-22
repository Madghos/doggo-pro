param(
  [Parameter(Position=0)][string]$Checkpoint='outputs/train/act-dog-target/checkpoints/last/pretrained_model',
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Rollout=Join-Path $Root '.venv\Scripts\lerobot-rollout.exe'
$Path=$Checkpoint
if (-not [System.IO.Path]::IsPathRooted($Path) -and -not $Path.StartsWith('hf://') -and -not ($Path -match '^[^/]+/[^/]+$')) { $Path=Join-Path $Root $Path }
$ArgsList=@('--strategy.type=base',"--policy.path=$Path",'--robot.type=dog_sim','--robot.discover_packages_path=dog_sim','--task=Move the dog to the green disk','--fps=30','--robot.fps=30','--duration=20','--display_data=true') + @($ExtraArgs)
Push-Location $Root
try {& $Rollout @ArgsList; if ($LASTEXITCODE -ne 0) {throw "Eval failed $LASTEXITCODE"}}
finally {Pop-Location}
