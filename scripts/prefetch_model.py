#!/usr/bin/env python
"""Pre-download a model's weights into the HF cache BEFORE the main pipeline
runs, using huggingface_hub's snapshot_download with settings tuned for flaky
large-file transfers, wrapped in a hard per-attempt timeout + retry loop.

Why this exists: on a RunPod RTX 4090 pod (2026-07-03), model downloads
through both huggingface_hub's "xet" fast-download backend AND its standard
HTTP backend either failed with an exception mid-shard OR — the harder case —
hung indefinitely with NO exception and NO progress, surviving Ctrl+C, while
a plain `curl` to the identical URL succeeded (slowly). A try/except retry
loop cannot recover from the hang case: nothing is ever raised, so the
except branch never fires and the process just sits there forever.

The fix: run each attempt in a SEPARATE PROCESS under a hard wall-clock
timeout. If an attempt exceeds the timeout, it is killed (SIGKILL, no
possibility of it swallowing SIGINT/SIGTERM the way the interactive hang
did) and the next attempt starts fresh. snapshot_download()'s own file-level
resume means a killed attempt doesn't lose completed shards — the next
attempt picks up mid-download. Combined with `max_workers=1`, this mimics
the single-connection behavior that worked reliably via curl.

Usage:
    python scripts/prefetch_model.py --model olmoe
    python scripts/prefetch_model.py --model qwen_moe
    python scripts/prefetch_model.py --model deepseek_moe
    python scripts/prefetch_model.py --all

Called automatically by run_all.sh before the main pipeline starts.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-attempt wall-clock budget. Generous enough for a genuinely slow (but
# progressing) transfer of the largest shard in this study's models
# (~8-9GB) even at the ~5-10 MB/s "slow but happening" rate observed via
# curl on the flaky pod, with margin — this is a timeout for STALLS, not a
# realistic-speed budget, so err high rather than killing a slow-but-working
# transfer.
ATTEMPT_TIMEOUT_SECONDS = 45 * 60
MAX_ATTEMPTS = 6


def _worker_snippet(hf_id: str, revision: str | None) -> str:
    """Source for the single-shot child process: try ONE snapshot_download
    call and exit. Runs in its own process so the parent can SIGKILL it on
    a timeout regardless of what the download call is doing internally."""
    return f"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={hf_id!r},
    revision={revision!r},
    max_workers=1,
    etag_timeout=30,
)
print("PREFETCH_OK")
"""


def prefetch(hf_id: str, revision: str | None):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[{hf_id}] attempt {attempt}/{MAX_ATTEMPTS} "
              f"(revision={revision or 'main'}, timeout={ATTEMPT_TIMEOUT_SECONDS}s) ...")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _worker_snippet(hf_id, revision)],
                timeout=ATTEMPT_TIMEOUT_SECONDS,
                cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            print(f"[{hf_id}] attempt {attempt} HUNG past {ATTEMPT_TIMEOUT_SECONDS}s — "
                  f"killed. Retrying (already-downloaded shards are NOT lost, "
                  f"snapshot_download resumes them).")
            continue

        if result.returncode == 0:
            print(f"[{hf_id}] prefetch succeeded on attempt {attempt}.")
            return
        print(f"[{hf_id}] attempt {attempt} exited with code {result.returncode} — retrying.")
        if attempt < MAX_ATTEMPTS:
            backoff = min(20 * attempt, 120)
            print(f"[{hf_id}] waiting {backoff}s before retry...")
            time.sleep(backoff)

    raise RuntimeError(
        f"[{hf_id}] failed to download after {MAX_ATTEMPTS} attempts. "
        f"This is the same failure pattern that forced the DeepSeek-V2-Lite -> "
        f"deepseek-moe-16b-base swap (see README) — if this keeps happening for "
        f"a DIFFERENT model, the problem is this pod's network, not the specific "
        f"HF repo. Consider a different RunPod region/pod, or run "
        f"`curl -L -o /tmp/test.bin <a large file URL from this repo>` to confirm "
        f"raw connectivity before retrying the pipeline."
    )


def check_not_on_network_volume():
    """RunPod separates container disk (fast, per-pod) from an optional
    mounted network volume (shared, quota-limited, e.g. /workspace on some
    pods). Downloading multi-GB model weights onto the network volume by
    accident — because the repo was cloned there — caused a
    'Disk quota exceeded' error on 2026-07-03 even though the container's
    own disk was almost empty. Fail loudly here rather than silently eating
    hours of download time before hitting the same quota wall."""
    cwd = Path.cwd()
    try:
        result = subprocess.run(["df", str(cwd)], capture_output=True, text=True, timeout=10)
        output = result.stdout
    except Exception:
        return  # best-effort; don't block a run over a diagnostic that itself failed

    suspicious_markers = ["mfs#", "nfs", ":/workspace", "runpod-volume"]
    if any(marker in output for marker in suspicious_markers):
        print(f"\n{'!'*70}")
        print(f"WARNING: current directory ({cwd}) appears to be on a network-mounted")
        print(f"filesystem, not the pod's local container disk:\n{output}")
        print(f"Network volumes on RunPod are often quota-limited per user even when")
        print(f"the pool shows huge free space. If downloads fail with 'Disk quota")
        print(f"exceeded', clone/run this repo from the container's local disk instead")
        print(f"(commonly /root or / — check with `df -h .` after cd'ing there).")
        print(f"{'!'*70}\n")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["olmoe", "qwen_moe", "deepseek_moe"])
    group.add_argument("--all", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    check_not_on_network_volume()

    config = yaml.safe_load(Path(args.config).read_text())
    models_to_fetch = list(config["models"].keys()) if args.all else [args.model]

    for model_key in models_to_fetch:
        model_cfg = config["models"][model_key]
        prefetch(model_cfg["hf_id"], model_cfg.get("revision"))

    print("\nAll requested models cached. Running the pipeline now reads from local")
    print("disk — from_pretrained() will not need the network for these models again")
    print("unless the cache is deleted.")


if __name__ == "__main__":
    main()
