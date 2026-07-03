#!/usr/bin/env python
"""Pre-download a model's weights into the HF cache BEFORE the main pipeline
runs, using plain curl subprocesses instead of huggingface_hub's downloader.

Why this exists: on a RunPod RTX 4090 pod (2026-07-03), huggingface_hub's
downloader — both its "xet" fast-download backend AND its standard HTTP
backend — hung indefinitely with NO exception (survived Ctrl+C, needed
Ctrl+Z + kill -9) on TWO DIFFERENT models' first shard, stuck at 0% with no
progress. A direct `curl` to the identical file URLs succeeded both times
(slowly, ~5-10 MB/s, but making real progress). This pointed at
huggingface_hub's download machinery itself (connection pooling / retry
logic / xet client) misbehaving on this pod's network, not the network
being down. The fix: stop using huggingface_hub's downloader entirely and
build the local HF cache directory by hand with curl.

How it works: this script (1) lists a repo's files via the small, fast HF
API `GET /api/models/{repo_id}` call (JSON, not the large-file path), then
(2) curls each file directly into
`~/.cache/huggingface/hub/models--{org}--{name}/snapshots/main/{filename}`.
huggingface_hub's cache resolution only checks that this path exists — it
does not require the blobs/ + symlink structure it creates itself, and
falls back to treating the revision string literally as the snapshot
folder name when no `refs/{revision}` file is present. Once cached this
way, `from_pretrained(repo_id)` finds everything locally with zero
network calls (verified against huggingface_hub's cache resolution logic,
2026-07-03).

Usage:
    python scripts/prefetch_model.py --model olmoe
    python scripts/prefetch_model.py --model qwen_moe
    python scripts/prefetch_model.py --model deepseek_moe
    python scripts/prefetch_model.py --all

Called automatically by run_all.sh before the main pipeline starts.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

CURL_TIMEOUT_SECONDS = 45 * 60  # per-file hard wall-clock budget
MAX_ATTEMPTS_PER_FILE = 4

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", HF_HOME / "hub"))


def cache_dir_for(hf_id: str) -> Path:
    org_name = hf_id.replace("/", "--")
    return HF_HUB_CACHE / f"models--{org_name}"


def list_repo_files(hf_id: str, revision: str | None) -> list[str]:
    """Small, fast metadata call — not the large-file download path that hangs."""
    ref = revision or "main"
    url = f"https://huggingface.co/api/models/{hf_id}?revision={ref}"
    print(f"[{hf_id}] listing files via {url} ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    files = [s["rfilename"] for s in data.get("siblings", [])]
    if not files:
        raise RuntimeError(f"[{hf_id}] API returned no file list — check the repo ID and revision.")
    print(f"[{hf_id}] {len(files)} files found: {files}")
    return files


def curl_file(url: str, dest: Path) -> bool:
    """Returns True on success. Uses curl -C - for resume-on-retry (partial
    files from a killed/timed-out attempt are continued, not restarted)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".partial")

    for attempt in range(1, MAX_ATTEMPTS_PER_FILE + 1):
        print(f"    attempt {attempt}/{MAX_ATTEMPTS_PER_FILE}: curl -> {dest.name}")
        try:
            result = subprocess.run(
                ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5",
                 "-C", "-",  # resume partial download if tmp_dest already exists
                 "-o", str(tmp_dest), url],
                timeout=CURL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print(f"    attempt {attempt} exceeded {CURL_TIMEOUT_SECONDS}s — killed, retrying "
                  f"(curl -C - resumes from where this attempt left off, not from scratch)")
            continue

        if result.returncode == 0:
            tmp_dest.rename(dest)
            return True
        print(f"    curl exited {result.returncode} — retrying")
        time.sleep(min(10 * attempt, 60))

    return False


SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".md", ".gitattributes")


def prefetch(hf_id: str, revision: str | None):
    files = list_repo_files(hf_id, revision)
    snapshot_dir = cache_dir_for(hf_id) / "snapshots" / (revision or "main")

    for filename in files:
        if any(filename.lower().endswith(suf) for suf in SKIP_SUFFIXES):
            print(f"[{hf_id}] {filename}: skipping (not needed for from_pretrained — logo/readme/gitattributes)")
            continue
        dest = snapshot_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{hf_id}] {filename}: already cached ({dest.stat().st_size / 1e6:.1f} MB), skipping")
            continue
        url = f"https://huggingface.co/{hf_id}/resolve/{revision or 'main'}/{filename}"
        ok = curl_file(url, dest)
        if not ok:
            raise RuntimeError(
                f"[{hf_id}] failed to download {filename} after {MAX_ATTEMPTS_PER_FILE} attempts via curl. "
                f"Since curl (not huggingface_hub) is now the download path, a failure here means a "
                f"genuine network problem to huggingface.co from this pod — test with: "
                f"curl -o /dev/null -w '%{{http_code}} %{{speed_download}}\\n' -L {url}"
            )
        size_mb = dest.stat().st_size / 1e6
        print(f"[{hf_id}] {filename}: done ({size_mb:.1f} MB)")

    print(f"[{hf_id}] all {len(files)} files cached at {snapshot_dir}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["olmoe", "qwen_moe", "deepseek_moe"])
    group.add_argument("--all", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    models_to_fetch = list(config["models"].keys()) if args.all else [args.model]

    for model_key in models_to_fetch:
        model_cfg = config["models"][model_key]
        prefetch(model_cfg["hf_id"], model_cfg.get("revision"))

    print("\nAll requested models cached via curl. from_pretrained() will read these")
    print("files from local disk with no network call, since huggingface_hub's cache")
    print("resolution only checks that snapshots/{revision}/{filename} exists — it does")
    print("not require the blobs/+symlink structure it would normally create itself.")


if __name__ == "__main__":
    main()
