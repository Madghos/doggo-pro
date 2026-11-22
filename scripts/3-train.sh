#!/usr/bin/env bash
# Train an ACT policy on a dataset from the Hub. Run this on the cluster.
#
# Usage:
#   bash scripts/3-train.sh <hf-user>/<dataset-name> [job-name] [extra lerobot-train args...]
#
# The checkpoints go to outputs/train/<job-name>/checkpoints/. To train on a
# laptop, add --policy.device=mps (Apple) or --policy.device=cpu.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:?give the dataset repo id, for example my-hf-user/sim-pick-cube}"
JOB="${2:-act-sim-pick-cube2}"
shift $(( $# > 2 ? 2 : $# ))

cd "$ROOT"
exec uv run lerobot-train \
    --dataset.repo_id="$REPO" \
    --policy.type=act \
    --policy.device=mps \
    --policy.private=true \
    --policy.push_to_hub=false \
    --output_dir="outputs/train/$JOB" \
    --job_name="$JOB" \
    --batch_size=16 \
    --steps=40000 \
    --save_freq=5000 \
    "$@"
