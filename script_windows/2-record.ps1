param(
  [Parameter(Mandatory=$true,Position=0)][string]$Repo,
  [Parameter(Position=1)][int]$Episodes=10,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Cmd=Join-Path $Root '.venv\Scripts\lerobot-record.exe'
if (!(Test-Path $Cmd)) { throw "Missing $Cmd" }
$Name=($Repo -split '/')[-1]
$ArgsList=@(
 '--robot.type=dog_sim','--robot.discover_packages_path=dog_sim',
 '--teleop.type=dog_keyboard',"--dataset.repo_id=$Repo",
 "--dataset.root=outputs/datasets/$Name",'--dataset.no_stamp=true',
 '--dataset.push_to_hub=false',"--dataset.num_episodes=$Episodes",
 '--dataset.single_task=Move the dog to the green disk',
 '--dataset.video=false','--dataset.fps=30','--robot.fps=30',
 '--dataset.episode_time_s=60','--dataset.reset_time_s=5',
 '--display_data=true','--play_sounds=false'
) + @($ExtraArgs)
Push-Location $Root
try { & $Cmd @ArgsList; if ($LASTEXITCODE -ne 0) {throw "Record failed $LASTEXITCODE"} }
finally {Pop-Location}
