#!/usr/bin/env bash
# Record pick-and-place episodes in the simulation without a human, and upload
# the dataset to a private Hugging Face Hub repo at the end.
#
# A scripted expert reads the true cube pose from the simulation. It moves the
# gripper over the cube, grasps it, carries it to the green disk and goes back
# home (so101_sim/expert.py). After each attempt the script tests whether the
# cube lies on the disk. It saves a correct episode and discards a failed one,
# until the dataset has the requested number of correct episodes.
#
# Usage:
#   bash scripts/2-record_auto.sh <hf-user>/<dataset-name> [num-episodes] [extra args...]
#
# Log in first with `uv run hf auth login`. The default is 50 episodes.
#
# The extra args are the --dataset.* flags of lerobot-record, and also:
#
#   --seed=0                      the same cube positions at every run
#   --display_data=true           show the cameras in Rerun. The run gets slower.
#   --dataset.push_to_hub=false   keep the dataset only on this computer
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:?give the dataset repo id, for example my-hf-user/sim-pick-cube-auto}"
EPISODES="${2:-50}"
shift $(( $# > 2 ? 2 : $# ))  # moves arguments so everything left gets passed to the recorder

bash "$ROOT/scripts/0-fetch-model.sh" >/dev/null
cd "$ROOT"

# torchcodec 0.11 supports FFmpeg 4 to 8. Homebrew ships FFmpeg 9, so use ffmpeg@7 on macOS.
if [ "$(uname)" = "Darwin" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib
fi

# --dataset.fps (default 30) also sets the physics steps of the simulation.
exec uv run python -m so101_sim.record_auto \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT/outputs/datasets/${REPO#*/}" \
    --dataset.no_stamp=true \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --dataset.num_episodes="$EPISODES" \
    --dataset.single_task="Pick up the red cube and put it on the green disk" \
    --dataset.streaming_encoding=true \
    --dataset.rgb_encoder.vcodec=auto \
    "$@"
