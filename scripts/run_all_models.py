#!/usr/bin/env python
"""Runs olmoe -> qwen_moe -> deepseek_moe sequentially (fits one 24GB GPU
at a time; only one model is ever loaded into memory). If one model's run
fails, this logs the failure and CONTINUES to the next model rather than
aborting the whole batch — a bug in the DeepSeek adapter (the least-verified
of the three, see adapters/deepseek_moe.py, reused for deepseek-moe-16b-base
after DeepSeek-V2-Lite's download failed reproducibly) shouldn't block
OLMoE/Qwen results that already succeeded.

Exit code is non-zero if ANY model failed, so calling infrastructure
(GitHub Actions, a shell script's `set -e`, cron) can still detect partial failure
even though the run continues.
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS = ["olmoe", "qwen_moe", "deepseek_moe"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="full", choices=["extract", "ablate", "full"],
                        help="GPU phase to run for every model (passed to run_model.py).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    failures = []

    for model in MODELS:
        print(f"\n{'#'*70}\n# Starting {model} (phase={args.phase})\n{'#'*70}")
        result = subprocess.run(
            [sys.executable, str(script_dir / "run_model.py"),
             "--model", model, "--phase", args.phase],
            cwd=script_dir.parent,
        )
        if result.returncode != 0:
            print(f"\n*** {model} FAILED (exit code {result.returncode}) — see results/{model}.log ***")
            failures.append(model)
        else:
            print(f"\n*** {model} completed successfully ***")

    print(f"\n{'='*70}\nBatch summary: {len(MODELS) - len(failures)}/{len(MODELS)} models succeeded")
    if failures:
        print(f"Failed: {failures}")
        print("Re-run `python scripts/run_model.py --model <name>` for each failed model —")
        print("it will resume from the last completed checkpoint, not restart from scratch.")
        sys.exit(1)
    print("All models completed. Results under results/<model_name>/")


if __name__ == "__main__":
    main()
