#!/usr/bin/env bash
# GPU entrypoint for the study on a rented GPU pod (RunPod/Lambda). This does
# ONLY the GPU-bound work and splits it into two phases around the CPU-bound
# analysis, which you run OFF the pod (on a laptop) via analyse_all.sh:
#
#   ┌ pod instance A ─────────────┐   ┌ laptop ──────────┐   ┌ pod instance B ─┐
#   │ bash run_all.sh             │   │ bash analyse_all │   │ bash run_all.sh │
#   │  -> extraction (GPU)        │ → │  -> analysis     │ → │  -> ablation    │
#   │  -> stops (analysis not     │   │  (CPU, parallel, │   │  (GPU; sees     │
#   │     ready) OR ablates if    │   │   free)          │   │   analysis      │
#   │     analysis already present│   │                  │   │   present)      │
#   └─────────────────────────────┘   └──────────────────┘   └─────────────────┘
#
# One command, run the SAME way on both pod instances. It figures out what to do
# from the results/ tree:
#   * Extraction not finished for the whole matrix  -> run extraction, then stop
#     (or, if extraction just completed AND analysis is somehow already present,
#     fall through to ablation).
#   * Extraction done + analysis present for all     -> run ablation.
#   * Extraction done + analysis NOT present          -> stop and tell you to run
#     analyse_all.sh on the laptop, then re-run this on a fresh pod.
#
# Between the two pod instances you must move results/ off the pod, analyze on
# the laptop, and move it back (the pod's local disk does not persist):
#   pod A done:  scp -r <pod>:$(pwd)/results ./results-from-pod
#   laptop:      bash analyse_all.sh ./results-from-pod
#   pod B:       scp -r ./results-from-pod <podB>:$(pwd)/results  (before run_all.sh)
#
# Run inside tmux and walk away:
#   tmux new -s indic-moe ; bash run_all.sh    # Ctrl+B,D to detach
#
# Steps 1-3 (env checks, deps, model prefetch) are identical to before; only the
# final execution step is phase-aware. Extraction needs the models loaded; the
# ablation phase does too, so we prefetch in both cases.

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

# The ~100GB requirement counts data ALREADY in the local cache toward the
# total — otherwise a resumed run whose models are fully downloaded (~75GB
# in .hf_cache) fails the check precisely BECAUSE the downloads succeeded.
AVAIL_GB=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc '0-9')
CACHED_GB=$(du -s -BG "$HF_HOME" 2>/dev/null | cut -f1 | tr -dc '0-9')
CACHED_GB=${CACHED_GB:-0}
EFFECTIVE_GB=$((AVAIL_GB + CACHED_GB))
if [ "$EFFECTIVE_GB" -lt 100 ]; then
    echo "FATAL: ${AVAIL_GB}GB free + ${CACHED_GB}GB already cached = ${EFFECTIVE_GB}GB at $PWD." >&2
    echo "This study needs ~100GB total (three models at ~14+28+33GB, plus FLORES," >&2
    echo "results, and working headroom). Resize the volume before running." >&2
    exit 1
fi
echo "Disk check OK: ${AVAIL_GB}GB free + ${CACHED_GB}GB already cached (${EFFECTIVE_GB}GB effective)"

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
echo "=== [4/4] GPU pipeline (phase-aware) ==="
echo "    Current state of results/:"
python3 scripts/check_phase_ready.py status results || true

# Decide the phase from the results tree.
#   - If analysis is already present for every extracted cell AND every cell is
#     extracted, there's nothing left for extraction to do -> go straight to
#     ablation.
#   - Otherwise run extraction (idempotent: cells already extracted are skipped),
#     then re-check: if analysis is now present for all, ablate; else stop.
run_ablation_if_ready () {
    if python3 scripts/check_phase_ready.py analysis-all results; then
        echo ""
        echo "=== Analysis present for all extracted cells -> running ABLATION phase (GPU) ==="
        python3 scripts/run_all_models.py --phase ablate
        echo ""
        echo "=== Ablation complete. Full results in ./results/. ==="
        python3 scripts/check_phase_ready.py status results || true
        return 0
    fi
    return 1
}

if python3 scripts/check_phase_ready.py extracted-all results; then
    echo ""
    echo "All cells already extracted — skipping extraction phase."
    if run_ablation_if_ready; then
        exit 0
    fi
    echo ""
    echo "############################################################################"
    echo "# Extraction is done, but ANALYSIS is not present for all cells yet.        #"
    echo "# Next step (OFF the pod, e.g. on your laptop):                             #"
    echo "#   1. Sync results/ down:   scp -r <pod>:$(pwd)/results ./results-from-pod #"
    echo "#   2. Run analysis:         bash analyse_all.sh ./results-from-pod         #"
    echo "#   3. Sync results/ back to a fresh pod, then run:  bash run_all.sh        #"
    echo "############################################################################"
    exit 0
fi

echo ""
echo "=== Running EXTRACTION phase (GPU): routing capture only, stops before analysis ==="
echo "    olmoe -> qwen_moe -> deepseek_moe, each loaded once across the matrix:"
echo "    2 conditions (token_capped, aligned) x 2 seeds. Cells already extracted"
echo "    are skipped. Results land in results/<condition>/seed<N>/<model>/02_routing_raw/."
python3 scripts/run_all_models.py --phase extract

echo ""
echo "=== Extraction phase finished. ==="
python3 scripts/check_phase_ready.py status results || true

# It's possible (e.g. a resumed pod that already carried analysis back) that
# analysis is present for everything now; if so, ablate in the same run.
if run_ablation_if_ready; then
    exit 0
fi

echo ""
echo "############################################################################"
echo "# EXTRACTION DONE. Analysis (CPU) is next and runs OFF the pod:             #"
echo "#   1. Sync results/ down:   scp -r <pod>:$(pwd)/results ./results-from-pod #"
echo "#   2. On the laptop:        bash analyse_all.sh ./results-from-pod         #"
echo "#   3. Sync results/ back to a fresh pod and run again:  bash run_all.sh    #"
echo "#      (it will detect analysis is present and run the ablation phase)      #"
echo "#                                                                           #"
echo "# You can terminate THIS pod now to stop paying for the GPU during analysis.#"
echo "############################################################################"
