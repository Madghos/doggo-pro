param(
  [Parameter(Mandatory=$true,Position=0)][string]$Repo,
  [Parameter(Position=1)][int]$Episodes=50,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Python=Join-Path $Root '.venv\Scripts\python.exe'
$Name=($Repo -split '/')[-1]
$ArgsList=@('-m','dog_sim.record_auto',"--dataset.repo_id=$Repo", "--dataset.root=outputs/datasets/$Name", '--dataset.no_stamp=true','--dataset.push_to_hub=false',"--dataset.num_episodes=$Episodes",'--dataset.single_task=Move the dog to the green disk','--dataset.video=false','--dataset.fps=30') + @($ExtraArgs)
Push-Location $Root
try {& $Python @ArgsList; if ($LASTEXITCODE -ne 0) {throw "Auto-record failed $LASTEXITCODE"}}
finally {Pop-Location}
