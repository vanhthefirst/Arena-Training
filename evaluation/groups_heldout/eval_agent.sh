#!/usr/bin/env bash
# eval_agent.sh <agents_dir> <agent> <tick>
# The rosnav_rl driver always loads <agent>/best_model.zip, so a fixed checkpoint gets its own folder.
set -euo pipefail
dir=$1 agent=$2 tick=$3
src=$dir/$agent dst=$dir/${agent}_t$((tick / 1000))k
[ -f "$src/tick_$tick.zip" ] || { echo "missing $src/tick_$tick.zip" >&2; exit 1; }
mkdir -p "$dst"
cp "$src/training_config.yaml" "$dst/"
cp "$src/tick_$tick.zip" "$dst/best_model.zip"
sha256sum "$src/tick_$tick.zip" "$dst/best_model.zip"
