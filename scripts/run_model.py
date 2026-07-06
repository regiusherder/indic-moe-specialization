#!/usr/bin/env python
"""Run the full pipeline (data -> routing -> analysis -> ablation) for one model.

Usage:
    python scripts/run_model.py --model olmoe
    python scripts/run_model.py --model qwen_moe
    python scripts/run_model.py --model deepseek_moe

Re-running with the same --model after a crash resumes from the last
completed checkpoint (see src/pipeline.py's stage-by-stage artifact writes).
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

# Must be set before transformers/huggingface_hub is imported anywhere
# (including transitively via src.pipeline -> adapters). The "xet" fast-
# download backend failed reproducibly mid-shard on DeepSeek-V2-Lite on a
# RunPod RTX 4090 (2026-07-03) — same offset, two separate attempts. Falling
# back to the standard HTTP downloader fixed it immediately. run_all.sh also
# sets this at the shell level, but this covers running this script directly.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_model_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["olmoe", "qwen_moe", "deepseek_moe"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None, help="override output.results_dir from config")
    parser.add_argument("--phase", default="full", choices=["extract", "ablate", "full"],
                        help="GPU phase: 'extract' (routing only, stop before analysis), "
                             "'ablate' (needs extraction + laptop-produced analysis present), "
                             "'full' (extract+analysis+ablation on one box). Default: full.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())

    results_root = Path(args.results_dir or config["output"]["results_dir"]).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    log_path = results_root / f"{args.model}.log"
    print(f"Logging to stdout AND {log_path}")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

        def isatty(self):
            return False

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    try:
        run_model_pipeline(args.model, config, config_path, results_root, phase=args.phase)
    except Exception:
        print(f"\n{'!'*70}\nFATAL ERROR in {args.model} pipeline — full traceback below.\n"
              f"Re-running this command will resume from the last completed checkpoint\n"
              f"(check {results_root / args.model / '_checkpoint.json'} and the numbered\n"
              f"artifact directories to see what already finished).\n{'!'*70}\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
