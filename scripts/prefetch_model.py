#!/usr/bin/env python
"""Pre-download a model's weights into the HF cache BEFORE the main pipeline
runs, using huggingface_hub's snapshot_download with settings tuned for flaky
large-file transfers rather than transformers' from_pretrained() default path.

Why this exists: on a RunPod RTX 4090 pod (2026-07-03), model downloads
through both huggingface_hub's "xet" fast-download backend AND its standard
HTTP backend either failed reproducibly mid-shard or stalled indefinitely on
multi-GB safetensors shards — while a plain `curl` to the same URLs succeeded
(slowly). snapshot_download() with `max_workers=1` and `etag_timeout` bumped
up mimics curl's simpler, single-connection request pattern instead of
huggingface_hub's default multi-connection/xet-first behavior, and its
built-in resume-on-retry means a killed/stalled download picks up where it
left off rather than restarting.

Usage:
    python scripts/prefetch_model.py --model olmoe
    python scripts/prefetch_model.py --model qwen_moe
    python scripts/prefetch_model.py --model deepseek_moe
    python scripts/prefetch_model.py --all

Run this BEFORE `run_all.sh` / `run_model.py` on a pod where downloads have
been unreliable. Once cached, from_pretrained() calls in the main pipeline
read from local disk and touch the network only to check for updates
(which revision=<pinned commit> avoids entirely).
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import yaml
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def prefetch(hf_id: str, revision: str | None, max_retries: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{hf_id}] attempt {attempt}/{max_retries} (revision={revision or 'main'}) ...")
            path = snapshot_download(
                repo_id=hf_id,
                revision=revision,
                max_workers=1,          # one connection at a time, like the curl test that worked
                etag_timeout=30,        # HF's default (10s) is too aggressive for a slow/flaky link
                # deliberately NOT passing allow_patterns/ignore_patterns — we
                # want the full snapshot (config, tokenizer, all shards) so
                # the later from_pretrained() call never needs the network.
            )
            print(f"[{hf_id}] cached at: {path}")
            return path
        except Exception as e:
            print(f"[{hf_id}] attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            backoff = min(30 * attempt, 180)
            print(f"[{hf_id}] retrying in {backoff}s (snapshot_download resumes partial files, doesn't restart)...")
            time.sleep(backoff)


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

    print("\nAll requested models cached. You can now run scripts/run_model.py or run_all.sh —")
    print("from_pretrained() will load from local cache without re-downloading.")


if __name__ == "__main__":
    main()
