#!/usr/bin/env bash
# Let a trained policy drive the simulated SO-101 for 40 seconds.
#
# Usage:
#   bash scripts/4-eval.sh [checkpoint-dir] [extra lerobot-rollout args...]
#
# Copy the checkpoint from the cluster first, for example with rsync. The
# default is outputs/train/act-sim-pick-cube/checkpoints/last/pretrained_model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:-$ROOT/outputs/train/act-sim-pick-cube/checkpoints/last/pretrained_model}"
shift $(( $# > 1 ? 1 : $# ))

bash "$ROOT/scripts/0-fetch-model.sh" >/dev/null
cd "$ROOT"

if [ "$(uname)" = "Darwin" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib
fi

# so101_sim sets its physics steps from --robot.fps (default 30), not from
# --fps. If you change --fps, give the same value to --robot.fps.
exec uv run lerobot-rollout \
    --strategy.type=base \
    --policy.path="$CKPT" \
    --robot.type=so101_sim \
    --robot.discover_packages_path=so101_sim \
    --task="Pick up the red cube and put it on the green disk" \
    --fps=30 \
    --robot.fps=30 \
    --duration=20 \
    --display_data=true \
    "$@"
