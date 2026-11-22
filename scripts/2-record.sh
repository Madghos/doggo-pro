#!/usr/bin/env bash
# Teleoperate the simulated SO-101 with the keyboard, record a dataset, and
# upload it to a private Hugging Face Hub repo at the end.
#
# Usage:
#   bash scripts/2-record.sh <hf-user>/<dataset-name> [num-episodes] [extra lerobot-record args...]
#
# Log in first with `uv run hf auth login`.
#
# Arm keys. The listener is global, so it captures these letters even
# when the terminal is not focused. Do not type them elsewhere!
#
#   a / d   shoulder_pan
#   w / s   shoulder_lift
#   e / c   elbow_flex
#   t / g   wrist_flex
#   f / h   wrist_roll
#   o / p   gripper open / close
#   x       put the arm at home and the cube at a new random position
#
# Episode keys
#
#   n   end the episode, start the reset. Press it again to save and continue.
#   r   discard this episode, record it again
#   q   save this episode, then stop the run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:?give the dataset repo id, for example my-hf-user/sim-pick-cube}"
EPISODES="${2:-10}"
shift $(( $# > 2 ? 2 : $# ))  # moves arguments so everything left gets passed to LeRobot

bash "$ROOT/scripts/0-fetch-model.sh" >/dev/null
cd "$ROOT"

# torchcodec 0.11 supports FFmpeg 4 to 8. Homebrew ships FFmpeg 9, so use ffmpeg@7 on macOS.
if [ "$(uname)" = "Darwin" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib
fi

# so101_sim sets its physics steps from --robot.fps (default 30), not from
# --dataset.fps (default 30). If you change one of them, change the other too.
exec uv run lerobot-record \
    --robot.type=so101_sim \
    --robot.discover_packages_path=so101_sim \
    --teleop.type=keyboard_joints \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT/outputs/datasets/${REPO#*/}" \
    --dataset.no_stamp=true \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --dataset.num_episodes="$EPISODES" \
    --dataset.episode_time_s=200 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Pick up the red cube and put it on the green disk" \
    --dataset.streaming_encoding=true \
    --dataset.rgb_encoder.vcodec=auto \
    --display_data=true \
    "$@"
