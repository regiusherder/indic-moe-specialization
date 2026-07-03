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

# HF Hub's "xet" fast-download backend was observed to fail reproducibly
# mid-shard on a DeepSeek-V2-Lite download (RunPod RTX 4090, 2026-07-03):
# "RuntimeError: Internal Writer Error: Failed to send data: receiver
# dropped" at the same shard/offset on two separate attempts. Disabling it
# falls back to the standard HTTP downloader, which succeeded immediately.
# This MUST be set before any transformers/huggingface_hub import happens —
# an unattended run has nobody to notice a silent hang or retry a crash.
export HF_HUB_DISABLE_XET=1

echo "=== Environment check ==="
python3 --version
nvidia-smi || { echo "No GPU visible — aborting, this pipeline requires a GPU."; exit 1; }

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

# Pre-download all three models' weights BEFORE the pipeline starts. On a
# RunPod RTX 4090 (2026-07-03), huggingface_hub's downloader (both xet and
# its standard HTTP path) either failed reproducibly mid-shard or stalled
# indefinitely on multi-GB safetensors files, while plain curl to the same
# URLs succeeded (slowly). scripts/prefetch_model.py uses snapshot_download
# with max_workers=1 and a longer etag_timeout to mimic curl's simpler,
# single-connection behavior, with automatic resume-on-retry. Doing this as
# a separate up-front step (rather than relying on from_pretrained()'s
# on-demand download inside the pipeline) means an unattended run doesn't
# silently hang for hours on a stalled shard with nobody watching.
echo "=== Pre-fetching all model weights (see scripts/prefetch_model.py for why) ==="
python3 scripts/prefetch_model.py --all

echo "=== Running full pipeline (olmoe -> qwen_moe -> deepseek_moe) ==="
python3 scripts/run_all_models.py

echo "=== Done. Results in ./results/ — sync this directory off the pod before terminating it. ==="
