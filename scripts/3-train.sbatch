#!/usr/bin/env bash
#SBATCH --job-name=act-sim-pick-cube
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Train ACT on one GPU of the WMI cluster (https://cluster.wmi.amu.edu.pl).
#
# Usage. Run this from the example02 directory on access.cluster.wmi.amu.edu.pl:
#   sbatch scripts/3-train.sbatch <hf-user>/<dataset-name> [job-name] [extra lerobot-train args...]
#
# The #SBATCH lines must come directly after the shebang.
#
# The gpu partition has RTX 3090, 4090 and 5090 cards. uv.lock installs a
# PyTorch build for CUDA 13.0, which supports all three. If you change the
# environment and PyTorch rejects the RTX 5090 (sm_120), keep the job on the
# RTX 3090 and 4090 nodes:
#   sbatch --constraint="cc8.6|cc8.9" scripts/3-train.sbatch <hf-user>/<dataset-name>
#
# Before the first job, log in to the Hugging Face Hub on the access node.
# The token goes to ~/.cache/huggingface/token, where the job finds it:
#   module load uv
#   uvx --from huggingface_hub hf auth login
#
# The job writes its log to slurm-<job-id>.out in this directory. To see the
# job in the queue, run `squeue -u $USER`. To stop it, run `scancel <job-id>`.
set -euo pipefail

# Slurm copies this file to a spool directory, so $0 does not point to the
# project. The job starts in the directory of the sbatch command.
cd "$SLURM_SUBMIT_DIR"

module load uv
# torchcodec decodes the MP4 frames of the dataset with the FFmpeg libraries
# of this module (FFmpeg 8.1). The cluster admins ask to load it in every job.
module load ffmpeg

# Keep the uv cache in /work, on the same file system as .venv
CACHE="/work/$USER/.cache"
export UV_CACHE_DIR="$CACHE/uv"
export HF_LEROBOT_HOME="$CACHE/huggingface/lerobot"

# evdev is a Linux keyboard library that pynput needs. Training does not use
# it, and it has no wheel, so uv must build it with a compiler.
uv sync --no-install-package evdev

# Stop `uv run` in 3-train.sh from a second sync. That sync installs evdev.
export UV_NO_SYNC=1

# Write the GPU, the driver and the versions of LeRobot, PyTorch, CUDA and
# FFmpeg to the log.
nvidia-smi
uv run lerobot-info
exec bash scripts/3-train.sh "$@"
