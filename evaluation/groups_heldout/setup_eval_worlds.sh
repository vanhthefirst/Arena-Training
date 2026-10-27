#!/usr/bin/env bash
# setup_eval_worlds.sh <arena_src> <out_root>
# Copies the three held-out worlds next to the training workspace and writes the held-out scenarios into
# the copies, so evaluation can start (ARENA_WORLD_PATH=<out_root>) without a build in the training workspace.
set -euo pipefail
src=$1/arena_simulation_setup/worlds out=$2
py=${ARENA_VENV_DIR:+$ARENA_VENV_DIR/bin/python}
mkdir -p "$out"
for w in hospital_1_bare sbend_corridor reception; do
  rm -rf "${out:?}/$w"
  cp -r "$src/$w" "$out/$w"
done
"${py:-python3}" "$(dirname "$(readlink -f "$0")")/make_heldout.py" "$out"
