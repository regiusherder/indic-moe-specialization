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

# NOT pinning HF_HOME here deliberately: whatever it already resolves to
# (e.g. /workspace/.cache/huggingface on this RunPod image) is left alone,
# so files prefetch_model.py already downloaded there are reused, not
# re-fetched under a different path. See scripts/run_model.py and the
# adapters for how the pipeline is told to trust that cache and skip the
# network entirely once prefetch confirms the files are present.

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

# Warn (don't block — some pods legitimately run everything from a network
# mount) if the current directory looks like RunPod's shared network volume
# rather than the pod's own container disk. Downloading ~75GB of model
# weights onto a quota-limited network mount was the root cause of a
# "Disk quota exceeded" error on 2026-07-03 even with the container disk
# almost empty.
CWD_FS=$(df "$PWD" | tail -1)
if echo "$CWD_FS" | grep -qE "mfs#|nfs|:/workspace|runpod-volume"; then
    echo ""
    echo "!!! WARNING: current directory appears to be on a network-mounted filesystem:"
    echo "    $CWD_FS"
    echo "!!! Network volumes on RunPod are often quota-limited per user even when the"
    echo "!!! pool shows huge free space. If this run later fails with 'Disk quota"
    echo "!!! exceeded', re-clone and run this repo from the pod's local container disk"
    echo "!!! instead (commonly /root or /)."
    echo ""
    sleep 5
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
