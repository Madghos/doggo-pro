#!/usr/bin/env bash
# Download the SO-101 model from mujoco_menagerie.
#
# Usage:
#   bash scripts/0-fetch-model.sh

# -e: exit if any command fails
# -u: fail if we ask for an unset env var
# -o pipefail: fails the pipeline (`cmd1 | cmd2 | ...`) if any of its commands fails (not the only last one)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/assets/menagerie/robotstudio_so101"

if [ ! -d "$MODEL_DIR" ]; then  # if MODEL_DIR dir does not exist, then clone selected files
    # --depth 1           only the last commit
    # --filter=blob:none  do not fetch file contents yet
    # --sparse            check out only the top-level files
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/google-deepmind/mujoco_menagerie.git "$ROOT/assets/menagerie"
    # -C dir              run git as if we're in this folder
    # sparse-checkout set dir   add dir to the working tree
    git -C "$ROOT/assets/menagerie" sparse-checkout set robotstudio_so101
fi

# MuJoCo resolves <include> and mesh paths from the folder of the main file, so we copy our scene
cp "$ROOT/so101_sim/scene.xml" "$MODEL_DIR/pick_cube.xml"
