"""FLORES-200 acquisition and two sampling conditions.

The study runs each model under TWO sampling schemes, because they control
complementary confounds and no single scheme controls both:

  token_capped -- equal TOKEN COUNT per language (~20k). Tokenization fertility
    varies up to ~9x across scripts (English ~0.20 tok/char vs Punjabi ~1.78),
    so capping by SENTENCE count would give high-fertility languages far more
    routing decisions (= more statistical power) than low-fertility ones purely
    as an artifact of orthography. Capping by tokens equalizes statistical
    precision. COST: each language then covers different CONTENT -- English
    spans ~736 FLORES sentences to reach the budget, Malayalam ~73 -- so the
    languages are compared on non-overlapping slices of the parallel corpus
    (critique #5).

  aligned -- the SAME aligned sentence indices for every language. FLORES-200
    is sentence-aligned (line i is the same meaning in every language), so this
    controls content perfectly: every language routes on identical material
    (critique #6). COST: token counts then differ by fertility, reintroducing
    the precision imbalance token_capped removed -- which is exactly why we run
    both and require the family-structure finding to survive under each.
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
) -> tuple[list[str], int]:
    """Take sentences in order until adding the next one would exceed the token
    budget. Returns (selected_sentences, n_tokens_estimate).

    Returns the SENTENCE LIST (not a concatenated blob) so the pipeline can feed
    them to the model one at a time -- routing is captured per-sentence, which
    the bootstrap and permutation tests need. Token counting uses the same
    per-sentence, no-special-token convention the extraction loop's counts are
    compared against; the earlier version counted the growing *concatenated*
    string, which double-counted separators and disagreed with what the model
    was actually fed (critique #7 -- consistency between budgeting and extraction).
    """
    selected = []
    n_tokens = 0
    for sentence in sentences:
        # count this sentence on its own (matches how it will actually be fed:
        # each sentence is a separate forward pass, not concatenated)
        st = len(tokenizer(sentence, add_special_tokens=False, verbose=False)["input_ids"])
        if n_tokens + st > max_tokens and selected:
            break
        selected.append(sentence)
        n_tokens += st
        if n_tokens >= max_tokens:
            break

    if not selected:
        raise RuntimeError("Token budget too small to fit even one sentence — increase max_tokens_per_language.")

    return selected, n_tokens


def build_aligned_sample(
    sentences: list[str],
    n_aligned_sentences: int,
) -> tuple[list[str], int]:
    """The first n_aligned_sentences sentences, verbatim, no tokenizer involved.

    Because FLORES-200 is line-aligned, taking the first N lines gives the SAME
    content (same source sentences) for every language -- the whole point of the
    aligned condition. No token budget: token counts will vary by fertility, and
    that's the honest cost we report. Returns (selected_sentences, n_selected).
    """
    selected = sentences[:n_aligned_sentences]
    if not selected:
        raise RuntimeError("No sentences available for aligned sampling.")
    return selected, len(selected)
