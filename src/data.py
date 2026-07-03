"""FLORES-200 acquisition and token-capped sampling.

Sampling is capped by TOKEN COUNT per language, not sentence count or char
count. This is deliberate: tokenization fertility varies up to 9x across
scripts in this study (English ~0.20 tok/char vs Punjabi ~1.78 tok/char).
Capping by sentence count would give high-fertility languages far more
routing decisions (= more statistical power) than low-fertility ones purely
as an artifact of orthography, contaminating any cross-language JSD comparison.
"""
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

FLORES_DIRNAME = "flores200_dataset"


def download_flores(cache_dir: Path, url: str, expected_sha256: str = None) -> tuple[Path, str]:
    """Download + extract FLORES-200, verifying the tarball checksum.

    The sha256 of the first download is persisted next to the extracted data
    (flores_sha256.txt) so every later run — including on a different machine
    with `expected_sha256` still null in config — verifies against the same
    bytes the study started with, and the hash lands in each run's manifest.

    Returns (extract_dir, sha256).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "flores200_dataset.tar.gz"
    extract_dir = cache_dir / FLORES_DIRNAME
    sha_path = cache_dir / "flores_sha256.txt"

    if extract_dir.exists():
        recorded = sha_path.read_text().strip() if sha_path.exists() else "unrecorded (extracted before checksum persistence was added)"
        if expected_sha256 and sha_path.exists() and recorded != expected_sha256:
            raise RuntimeError(
                f"Cached FLORES-200 was downloaded with sha256 {recorded}, but config "
                f"expects {expected_sha256}. Delete {cache_dir} to re-download, or fix the config."
            )
        return extract_dir, recorded

    print(f"Downloading FLORES-200 from {url} ...")
    urllib.request.urlretrieve(url, tar_path)

    sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    if expected_sha256 and sha256 != expected_sha256:
        raise RuntimeError(
            f"FLORES-200 tarball checksum mismatch.\n"
            f"  expected: {expected_sha256}\n  got:      {sha256}\n"
            f"The dataset may have changed upstream or the download was corrupted. "
            f"Refusing to proceed with unverified data."
        )
    print(f"Downloaded. sha256={sha256}")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache_dir)
    tar_path.unlink()
    sha_path.write_text(sha256 + "\n")
    return extract_dir, sha256


def load_language_sentences(
    flores_dir: Path,
    lang_code: str,
    split: str,
    max_sentences_pool: int,
) -> list[str]:
    filepath = flores_dir / split / f"{lang_code}.{split}"
    if not filepath.exists():
        raise FileNotFoundError(
            f"FLORES file missing for language code '{lang_code}': {filepath}\n"
            f"Check the code against the FLORES-200 language list — a typo here "
            f"fails loudly rather than silently skipping a language."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines[:max_sentences_pool]


def build_token_capped_sample(
    sentences: list[str],
    tokenizer,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Concatenate sentences one at a time until the token budget is hit.

    Returns (text, n_sentences_used, n_tokens_actual). n_tokens_actual may be
    slightly under max_tokens (stops before exceeding, doesn't truncate mid-sentence)
    so token counts are comparable but not necessarily byte-identical across languages.
    """
    text = ""
    n_sentences = 0
    n_tokens = 0
    for sentence in sentences:
        candidate = (text + " " + sentence).strip() if text else sentence
        candidate_tokens = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if candidate_tokens > max_tokens and n_sentences > 0:
            break
        text = candidate
        n_tokens = candidate_tokens
        n_sentences += 1
        if n_tokens >= max_tokens:
            break

    if n_sentences == 0:
        raise RuntimeError("Token budget too small to fit even one sentence — increase max_tokens_per_language.")

    return text, n_sentences, n_tokens
