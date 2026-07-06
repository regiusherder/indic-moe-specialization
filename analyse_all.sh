#!/usr/bin/env bash
# LAPTOP entrypoint for the CPU-bound analysis stage. Runs OFF the GPU pod, for
# free, in parallel across cores. This is step 2 of the split pipeline:
#
#   pod A:   bash run_all.sh                 # extraction (GPU) -> 02_routing_raw/
#   laptop:  bash analyse_all.sh [RESULTS]   # analysis (CPU)   -> 03_analysis/   <-- you are here
#   pod B:   bash run_all.sh                 # ablation (GPU)   -> 04_ablation/
#
# It scans the results tree for cells that have extracted routing (.pkl) but no
# valid analysis yet, and computes per-layer JSD + permutation tests + bootstrap
# CIs for each, parallelized across the (independent) layers. NO GPU, NO model,
# NO torch required — only numpy/scipy/pandas. Same code path as the pod's inline
# analysis, so numbers are identical and reproducible.
#
# Usage:
#   bash analyse_all.sh                       # analyzes ./results
#   bash analyse_all.sh ./results-from-pod    # analyzes a synced-down tree
#   bash analyse_all.sh ./results --workers 6 # override worker count
#   bash analyse_all.sh ./results --force     # recompute even if present
#
# Windows note: run under Git Bash / WSL. The heavy lifting is Python, which is
# cross-platform; this wrapper only needs a POSIX shell.

set -euo pipefail
cd "$(dirname "$0")"

# First arg is the RESULTS dir ONLY if it doesn't look like a flag; otherwise
# default to ./results and treat all args as flags for run_analysis.py.
RESULTS="results"
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    RESULTS="$1"
    shift
fi

if [ ! -d "$RESULTS" ]; then
    echo "FATAL: results dir '$RESULTS' not found." >&2
    echo "Pass the path to the synced-down results tree, e.g.:" >&2
    echo "    bash analyse_all.sh ./results-from-pod" >&2
    exit 1
fi

# Pick a python (python3 on unix, python on Windows/Git Bash).
PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

echo "=== Dependency check (numpy/scipy/pandas/pyyaml; NO torch needed) ==="
if ! "$PY" -c "import numpy, scipy, pandas, yaml" 2>/dev/null; then
    echo "Installing analysis-only dependencies ..."
    "$PY" -m pip install -q numpy scipy pandas pyyaml
fi

echo ""
echo "=== Current state of $RESULTS ==="
"$PY" scripts/check_phase_ready.py status "$RESULTS" || true

echo ""
echo "=== Running analysis on cells that need it (parallel across layers) ==="
# forward any extra flags (--workers N, --force) straight to run_analysis.py
"$PY" scripts/run_analysis.py "$RESULTS" "$@"

echo ""
echo "=== Analysis done. State now: ==="
"$PY" scripts/check_phase_ready.py status "$RESULTS" || true

echo ""
echo "############################################################################"
echo "# Next: sync '$RESULTS' back to a fresh GPU pod and run:  bash run_all.sh   #"
echo "# It will detect analysis is present and run the ablation phase (GPU).      #"
echo "############################################################################"
