param(
  [Parameter(Mandatory=$true,Position=0)][string]$Repo,
  [Parameter(Position=1)][string]$Job='act-dog-target',
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Python=Join-Path $Root '.venv\Scripts\python.exe'
$Train=Join-Path $Root '.venv\Scripts\lerobot-train.exe'
& $Python -c 'import torch,sys; print("CUDA:",torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 2)'
if ($LASTEXITCODE -ne 0) {throw 'CUDA not available: install CUDA-enabled PyTorch'}
$ArgsList=@("--dataset.repo_id=$Repo",'--policy.type=act','--policy.device=cuda','--policy.private=true','--policy.push_to_hub=false',"--output_dir=outputs/train/$Job","--job_name=$Job",'--batch_size=4','--steps=4000','--save_freq=1000') + @($ExtraArgs)
Push-Location $Root
try { & $Train @ArgsList; if ($LASTEXITCODE -ne 0) {throw "Train failed $LASTEXITCODE"} }
finally {Pop-Location}
