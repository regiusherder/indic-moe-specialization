#!/usr/bin/env bash
# Entrypoint for a rented GPU pod (RunPod/Lambda). Run this over SSH inside
# a `tmux`/`screen` session so the job survives an SSH disconnect, e.g.:
#
#   tmux new -s indic-moe
#   bash run_all.sh
#   # Ctrl+B, D to detach; `tmux attach -t indic-moe` to check back in later
#
# Does NOT use Docker by default (faster to iterate on a rented pod that
# already has CUDA/drivers set up) — use the Dockerfile instead if you want
# full OS-level reproducibility beyond just the Python environment.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Environment check ==="
python3 --version
nvidia-smi || { echo "No GPU visible — aborting, this pipeline requires a GPU."; exit 1; }

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== Running full pipeline (olmoe -> qwen_moe -> deepseek_v2lite) ==="
python3 scripts/run_all_models.py

echo "=== Done. Results in ./results/ — sync this directory off the pod before terminating it. ==="
