"""Run manifest: every output directory gets a manifest.json recording exactly
what produced it, so a result found later can be traced back without guessing.
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return True  # unknown state — treat conservatively as "dirty"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def write_manifest(output_dir: Path, config: dict, config_path: Path, extra: dict = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": hash_config(config),
        "config_file": str(config_path),
        "config_snapshot": config,
    }
    try:
        import torch
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["gpu_name"] = torch.cuda.get_device_name(0)
            manifest["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    if extra:
        manifest.update(extra)

    # manifest.json always reflects the LATEST invocation; every invocation
    # (including resumes after a crash) is also appended to manifest_history.jsonl
    # so the original run's provenance is never silently overwritten.
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(output_dir / "manifest_history.jsonl", "a") as f:
        f.write(json.dumps(manifest, default=str) + "\n")
    return manifest_path
