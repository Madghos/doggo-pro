#!/usr/bin/env bash
# Usage: bash scripts/2-record_auto.sh <user>/<dataset> [episodes] [extra args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:?give the dataset repo id, for example local/a1-walk}"
EPISODES="${2:-50}"
shift $(( $# > 2 ? 2 : $# ))
cd "$ROOT"

exec uv run --locked --no-install-project python -m dog_sim.record_auto \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT/outputs/datasets/${REPO#*/}" \
    --dataset.no_stamp=true \
    --dataset.push_to_hub=false \
    --dataset.num_episodes="$EPISODES" \
    --dataset.single_task="Walk to point B" \
    --dataset.video=false \
    --dataset.fps=30 \
    --dataset.episode_time_s=20 \
    "$@"
