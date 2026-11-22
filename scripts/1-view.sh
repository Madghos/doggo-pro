#!/usr/bin/env bash
# Open the scene in the interactive MuJoCo viewer.
#
# Usage:
#   bash scripts/1-view.sh
#
# The viewer runs under plain python, also on macOS. Only the passive viewer
# (mujoco.viewer.launch_passive) needs mjpython.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="$ROOT/assets/menagerie/robotstudio_so101/pick_cube.xml"

bash "$ROOT/scripts/0-fetch-model.sh" >/dev/null
cd "$ROOT"
exec uv run python -m mujoco.viewer --mjcf="$SCENE"
