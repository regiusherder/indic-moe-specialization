#!/usr/bin/env bash
# Single entrypoint for the whole study on a rented GPU pod (RunPod/Lambda).
# Run this ONCE inside a tmux/screen session and walk away:
#
#   tmux new -s indic-moe
#   bash run_all.sh
#   # Ctrl+B, D to detach; `tmux attach -t indic-moe` to check back in later
#
# What this script does, in order, so nothing needs a second manual step:
#   1. Sanity-checks: GPU present, not accidentally running on a quota-limited
#      network volume, enough free disk for all three models + FLORES + results
#   2. Installs dependencies
#   3. Pre-fetches all three models' weights via scripts/prefetch_model.py,
#      which downloads files with plain curl subprocesses rather than
#      huggingface_hub's own downloader. On 2026-07-03, huggingface_hub's
#      downloader (both its xet backend and its standard HTTP backend) hung
#      indefinitely with no exception on TWO different models' first shard,
#      while curl to the identical URLs succeeded — so this bypasses
#      huggingface_hub's download machinery entirely and builds the local
#      HF cache directory by hand (see prefetch_model.py's docstring for
#      exactly how). Each file download runs under a hard timeout + retry,
#      so a stuck transfer gets killed and resumed rather than hanging
#      forever with nobody watching.
#   4. Runs the actual pipeline (olmoe -> qwen_moe -> deepseek_moe)
#
# Does NOT use Docker by default (faster to iterate on a rented pod that
# already has CUDA/drivers set up) — use the Dockerfile instead if you want
# full OS-level reproducibility beyond just the Python environment.

set -euo pipefail

cd "$(dirname "$0")"

# HF Hub's "xet" fast-download backend failed reproducibly on this study's
# models (RunPod RTX 4090, 2026-07-03). Disabled globally, before any
# transformers/huggingface_hub import happens anywhere in this run.
export HF_HUB_DISABLE_XET=1

# THE ROOT-CAUSE FIX for nearly every infrastructure failure hit on
# 2026-07-03: RunPod's PyTorch template sets HF_HOME to
# /workspace/.cache/huggingface — the SHARED, QUOTA-LIMITED, SLOW NETWORK
# VOLUME. Every model download was silently landing there, which explains:
#   - "Disk quota exceeded" errors while the container disk sat empty
#   - downloads hanging at 0% / mid-shard (multi-GB writes onto a slow
#     network filesystem, through huggingface_hub's downloader)
#   - very slow curl transfers (bottlenecked by network-mount writes,
#     not by the connection to HF's CDN)
#   - stale half-finished cache state (refs/main pointing at snapshot dirs
#     missing their safetensors) surviving across pod restarts
# Overriding HF_HOME to a directory inside this repo (which the disk checks
# below guarantee is on the pod's fast local container disk) fixes all of
# these at once, and gives every fresh clone a clean cache with no stale
# refs/ or .no_exist markers left over from previous attempts.
export HF_HOME="$PWD/.hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME overridden to $HF_HOME (local disk, not the pod's network volume)"

echo "=== [1/4] Environment checks ==="
python3 --version

if ! command -v curl > /dev/null 2>&1; then
    echo "FATAL: curl not found. scripts/prefetch_model.py requires it (huggingface_hub's" >&2
    echo "own downloader was unreliable on the pod this was developed against — see" >&2
    echo "prefetch_model.py's docstring). Install curl before continuing." >&2
    exit 1
fi

if ! nvidia-smi > /dev/null 2>&1; then
    echo "FATAL: no GPU visible. This pipeline requires a GPU (tested against a single 24GB RTX 4090)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# FATAL (not just a warning) if the repo sits on RunPod's shared network
# volume: HF_HOME now lives inside this repo, so a network-mounted PWD would
# put ~75GB of model weights onto the quota-limited, slow shared mount —
# the exact root cause of the 2026-07-03 "Disk quota exceeded" and
# hung-download failures. Better to refuse up front than fail hours in.
CWD_FS=$(df "$PWD" | tail -1)
if echo "$CWD_FS" | grep -qE "mfs#|nfs|:/workspace|runpod-volume"; then
    echo "FATAL: this repo is on a network-mounted filesystem:" >&2
    echo "    $CWD_FS" >&2
    echo "Network volumes on RunPod are quota-limited per user and slow for multi-GB" >&2
    echo "writes. Re-clone and run from the pod's local container disk instead:" >&2
    echo "    cd /root && git clone <this repo> && cd indic-moe-specialization && bash run_all.sh" >&2
    exit 1
fi

AVAIL_GB=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 100 ]; then
    echo "FATAL: only ${AVAIL_GB}GB free at $PWD. This study needs ~100GB+ (three" >&2
    echo "models cached simultaneously at ~14+28+33GB, plus FLORES, results, and" >&2
    echo "working headroom). Resize the volume before running." >&2
    exit 1
fi
echo "Disk check OK: ${AVAIL_GB}GB free at $PWD"

echo ""
echo "=== [2/4] Installing dependencies ==="
pip install -q -r requirements.txt

# aria2c downloads each file over 16 parallel connections — much faster than
# single-connection curl (which was correct but slow), and it's a separate
# battle-tested client unaffected by huggingface_hub's downloader problems.
# Best-effort install: prefetch_model.py falls back to curl if aria2c is
# unavailable (e.g. no apt access), so this never blocks the run.
if ! command -v aria2c > /dev/null 2>&1; then
    echo "Installing aria2 for parallel downloads (falls back to curl if this fails)..."
    (apt-get update -qq && apt-get install -y -qq aria2) || echo "aria2 install failed — prefetch will use curl (slower but works)"
fi

echo ""
echo "=== [3/4] Pre-fetching all model weights (hardened against download hangs) ==="
python3 scripts/prefetch_model.py --all

# Once prefetch confirms every file is cached locally, force ALL subsequent
# huggingface_hub/transformers calls to skip the network entirely — including
# the lightweight etag/HEAD "check for updates" request from_pretrained()
# normally makes even when a local file already exists. That HEAD check is
# the most likely explanation for the pipeline re-downloading full shards
# after prefetch had already cached them via curl (2026-07-03): if THAT
# small request hung/failed the same way the big downloads did, transformers
# can fall through to re-fetching the whole file. HF_HUB_OFFLINE=1 is
# huggingface_hub's own documented "never touch the network" switch, so this
# is the correct fix rather than guessing at cache-path mismatches.
export HF_HUB_OFFLINE=1
echo "HF_HUB_OFFLINE=1 set — pipeline will only read from the cache prefetch just populated."

echo ""
echo "=== [4/4] Running full pipeline (olmoe -> qwen_moe -> deepseek_moe) ==="
python3 scripts/run_all_models.py

echo ""
echo "=== Done. Results in ./results/ — sync this directory off the pod before terminating it. ==="
echo "    Example: scp -r <pod-user>@<pod-host>:$(pwd)/results ./results-from-runpod"
