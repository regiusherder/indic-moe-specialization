#!/usr/bin/env python
"""Pre-download a model's weights into the HF cache BEFORE the main pipeline
runs, using aria2c (16 parallel connections; falls back to plain curl)
instead of huggingface_hub's downloader.

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
`{HF_HUB_CACHE}/models--{org}--{name}/snapshots/{snapshot_name}/{filename}`.

IMPORTANT correction (2026-07-03, second incident): an earlier version of
this script always used the literal string "main" as the snapshot folder
name, assuming huggingface_hub falls back to treating the revision string
literally when no `refs/{revision}` pointer file exists. That assumption
broke in practice: a PARTIAL huggingface_hub download attempt had already
created `refs/main` pointing at a real commit hash (e.g.
`9b0c1aa8...`), with a snapshot dir under that hash containing only the
small config/tokenizer files (the safetensors shards never finished). Once
that `refs/main` file exists, huggingface_hub's offline resolver ALWAYS
follows it to that hash-named directory — it never falls back to checking
a literal `snapshots/main/` folder, no matter what's in it. So this
script's `snapshots/main/` (fully populated by curl) was silently ignored,
and `HF_HUB_OFFLINE=1` correctly reported "file not found" because it was
looking in the OTHER (incomplete) snapshot directory the whole time.

Fix: before writing anything, this script now checks whether
`refs/{revision}` already exists. If it does, files are written into
THAT hash-named snapshot directory (so any pre-existing partial download's
pointer is honored and completed, not orphaned). If it doesn't, this script
creates `refs/{revision}` itself, pointing at a snapshot directory named
after the revision string — so a completely fresh cache is unambiguous too.

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


def list_repo_files(hf_id: str, revision: str | None) -> tuple[list[str], str]:
    """Small, fast metadata call — not the large-file download path that hangs.
    Returns (files, commit_sha). The sha matters: transformers' dynamic-module
    loader (trust_remote_code models) extracts the commit hash from the cache
    path with a regex requiring a 40-hex snapshot folder name — a folder
    literally named "main" makes that extraction return None and crash with
    `unsupported operand type(s) for /: 'PosixPath' and 'NoneType'`
    (hit live 2026-07-04 on deepseek-moe-16b-base)."""
    ref = revision or "main"
    url = f"https://huggingface.co/api/models/{hf_id}?revision={ref}"
    print(f"[{hf_id}] listing files via {url} ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    files = [s["rfilename"] for s in data.get("siblings", [])]
    sha = data.get("sha")
    if not files:
        raise RuntimeError(f"[{hf_id}] API returned no file list — check the repo ID and revision.")
    if not sha:
        raise RuntimeError(f"[{hf_id}] API response has no 'sha' — cannot build a cache layout "
                           f"transformers' dynamic-module loader will accept.")
    print(f"[{hf_id}] {len(files)} files found at commit {sha}")
    return files, sha


def _have_aria2() -> bool:
    import shutil
    return shutil.which("aria2c") is not None


def download_file(url: str, dest: Path) -> bool:
    """Returns True on success. Prefers aria2c (16 parallel connections per
    file — single-connection curl was correct but very slow on the pod this
    was developed against); falls back to curl if aria2c isn't installed.
    Both paths resume partial downloads across retries rather than restarting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".partial")
    use_aria2 = _have_aria2()

    for attempt in range(1, MAX_ATTEMPTS_PER_FILE + 1):
        tool = "aria2c" if use_aria2 else "curl"
        print(f"    attempt {attempt}/{MAX_ATTEMPTS_PER_FILE}: {tool} -> {dest.name}")
        try:
            if use_aria2:
                # -x/-s 4: HF's CDN issues pre-signed redirect URLs locked to a
                # specific byte range; with many parallel connections, the ones
                # that don't match their URL's signed range get 403'd (observed
                # live 2026-07-04 — downloads still completed via retry, but
                # noisily and with degraded parallelism). 4 connections keeps
                # most of the speed with far fewer signed-range rejections;
                # actual observed throughput was write-bound, not
                # connection-bound, once HF_HOME moved to local disk.
                # -c: resume; --file-allocation=none: skip slow preallocation;
                # summary lines instead of a progress bar (tmux/log friendly)
                cmd = ["aria2c", "-x", "4", "-s", "4", "-k", "1M", "-c",
                       "--retry-wait=2", "--max-tries=15",
                       "--file-allocation=none", "--summary-interval=15",
                       "--console-log-level=warn", "--download-result=hide",
                       "-d", str(tmp_dest.parent), "-o", tmp_dest.name, url]
            else:
                cmd = ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5",
                       "-C", "-", "-o", str(tmp_dest), url]
            result = subprocess.run(cmd, timeout=CURL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"    attempt {attempt} exceeded {CURL_TIMEOUT_SECONDS}s — killed, retrying "
                  f"(partial file is resumed, not restarted)")
            continue

        if result.returncode == 0:
            # aria2c leaves a .aria2 control file next to a completed download
            # only on failure; remove any stale one defensively
            control = tmp_dest.parent / (tmp_dest.name + ".aria2")
            if control.exists():
                control.unlink()
            tmp_dest.rename(dest)
            return True
        print(f"    {tool} exited {result.returncode} — retrying")
        if use_aria2 and attempt >= 2:
            # aria2c failing twice: drop to curl for the remaining attempts —
            # slower, but a different client/code path
            print("    switching to curl fallback for remaining attempts")
            use_aria2 = False
        time.sleep(min(10 * attempt, 60))

    return False


SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".md", ".gitattributes")


def resolve_snapshot_dir(hf_id: str, revision: str | None, sha: str) -> Path:
    """The snapshot directory MUST be named with the real 40-hex commit sha —
    transformers' trust_remote_code machinery regex-extracts the commit hash
    from this path and crashes on a non-hex folder name (see list_repo_files).

    Also migrates two kinds of pre-existing state without re-downloading:
      - a legacy `snapshots/main/` folder from this script's earlier versions
        (renamed to snapshots/{sha}, files preserved)
      - a partial huggingface_hub-created snapshots/{sha} (files merged into
        by the download loop's per-file exists-check)
    refs/{revision} is (re)written to point at the sha either way."""
    ref_name = revision or "main"
    repo_cache = cache_dir_for(hf_id)
    snapshot_dir = repo_cache / "snapshots" / sha

    # migrate a legacy literal-named snapshot dir (e.g. snapshots/main) so
    # its already-downloaded multi-GB shards are reused, not re-fetched
    legacy_dir = repo_cache / "snapshots" / ref_name
    if legacy_dir.exists() and legacy_dir != snapshot_dir:
        if not snapshot_dir.exists():
            print(f"[{hf_id}] migrating legacy snapshots/{ref_name} -> snapshots/{sha} (rename, no re-download)")
            legacy_dir.rename(snapshot_dir)
        else:
            print(f"[{hf_id}] both snapshots/{ref_name} and snapshots/{sha} exist — moving missing files over")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for item in legacy_dir.iterdir():
                target = snapshot_dir / item.name
                if not target.exists():
                    item.rename(target)
            import shutil
            shutil.rmtree(legacy_dir)

    ref_path = repo_cache / "refs" / ref_name
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(sha)
    print(f"[{hf_id}] refs/{ref_name} -> {sha}; snapshot dir: {snapshot_dir}")
    return snapshot_dir


def clean_stale_cache_state(hf_id: str, files: list[str]):
    """Remove a `.no_exist` marker directory if huggingface_hub previously
    recorded (incorrectly, from this script's point of view) that some of
    this repo's files don't exist upstream — that marker makes
    huggingface_hub refuse to look for them again even after this script
    downloads them. Also drops any `.incomplete` blob fragments from an
    earlier partial huggingface_hub attempt so they don't get mistaken for
    real content."""
    repo_cache = cache_dir_for(hf_id)
    no_exist_dir = repo_cache / ".no_exist"
    if no_exist_dir.exists():
        import shutil
        print(f"[{hf_id}] removing stale .no_exist marker directory ({no_exist_dir}) — "
              f"leftover from an earlier failed huggingface_hub attempt, would otherwise "
              f"make huggingface_hub refuse to recognize files this script downloads")
        shutil.rmtree(no_exist_dir)

    blobs_dir = repo_cache / "blobs"
    if blobs_dir.exists():
        for incomplete in blobs_dir.glob("*.incomplete"):
            print(f"[{hf_id}] removing stale incomplete blob fragment: {incomplete.name}")
            incomplete.unlink()


def prefetch(hf_id: str, revision: str | None):
    files, sha = list_repo_files(hf_id, revision)
    clean_stale_cache_state(hf_id, files)
    snapshot_dir = resolve_snapshot_dir(hf_id, revision, sha)

    for filename in files:
        if any(filename.lower().endswith(suf) for suf in SKIP_SUFFIXES):
            print(f"[{hf_id}] {filename}: skipping (not needed for from_pretrained — logo/readme/gitattributes)")
            continue
        dest = snapshot_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{hf_id}] {filename}: already cached ({dest.stat().st_size / 1e6:.1f} MB), skipping")
            continue
        # resolve by the exact sha, not the branch name — if the repo gets a
        # new commit mid-run, all files still come from one consistent revision
        url = f"https://huggingface.co/{hf_id}/resolve/{sha}/{filename}"
        ok = download_file(url, dest)
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

    print("\nAll requested models cached via curl, into the exact snapshot directory")
    print("huggingface_hub's refs/{revision} pointer resolves to (creating that pointer")
    print("if none existed yet). from_pretrained() with HF_HUB_OFFLINE=1 will read these")
    print("files from local disk with no network call.")


if __name__ == "__main__":
    main()
