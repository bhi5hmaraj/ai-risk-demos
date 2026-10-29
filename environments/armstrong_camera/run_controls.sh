#!/usr/bin/env bash
# Five control cells of the per-hit Armstrong eval on two models, with a given number of games per cell.
# Run with: bash run_controls.sh 2 for a smoke test, then bash run_controls.sh 30.
# Needs OPENROUTER_API_KEY. A keeps the original wording. B renames every noun. C is B without the camera.
# D is B on random boards. E is D without the camera, on the same boards.
set -euo pipefail
games=${1:?games per cell}
cells=("A false true false" "B true true false" "C true false false" "D true true true" "E true false true")
for model in anthropic/claude-sonnet-5.5 google/gemini-3.5-flash; do
  for cell in "${cells[@]}"; do
    read -r name neutral camera random_boards <<< "$cell"
    uv run vf-eval @ openrouter.toml --model "$model" --env.taskset.per_hit true --env.taskset.macros true \
      --env.taskset.neutral "$neutral" --env.taskset.camera "$camera" --env.taskset.random_boards "$random_boards" \
      -n "$games" -r 1 -c 5 --no-push > "outputs/control-$name-${model//\//-}.log" 2>&1 &
  done
done
wait
