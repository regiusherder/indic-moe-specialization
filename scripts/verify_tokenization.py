#!/usr/bin/env python
"""Sanity check: confirm each model's tokenizer actually TOKENIZES every
language, rather than collapsing whole scripts to <unk>.

Why this matters: if a tokenizer emitted <unk> for, say, all Malayalam
characters, the "routing distribution" for Malayalam would reflect the model
choking on an unknown token, not genuine language processing -- and the whole
specialization result would be an artifact. This script rules that out
directly by tokenizing each language's real sampled text (from
results/<model>/01_samples.json, the exact text the pipeline fed the model)
and counting how many tokens are the tokenizer's <unk> id.

Uses only the lightweight `tokenizers` library (pure Rust binding, no torch,
no transformers, no GPU) against each model's small tokenizer.json (a few MB,
fetched from the HF Hub on first run and cached in .tokcheck_cache/) -- this
runs on a laptop in seconds.

Verified result (2026-07-05, against the actual FLORES-200 samples used in
this study): 0.0000% UNK for all 11 languages across all 3 models. Every
script was genuinely tokenized into subwords; none collapsed to unknown
tokens. See results/figures/tokenization_audit.txt for the full log.

Usage:
    python scripts/verify_tokenization.py --results ./results
"""
import argparse
import json
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

CACHE_DIR = Path(".tokcheck_cache")


def fetch_tokenizer(hf_id: str) -> Path:
    dest = CACHE_DIR / hf_id.replace("/", "--") / "tokenizer.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{hf_id}/resolve/main/tokenizer.json"
        print(f"  fetching {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def find_unk_id(tok: Tokenizer):
    vocab = tok.get_vocab()
    for candidate in ("<unk>", "[UNK]", "<|unk|>", "unk"):
        if candidate in vocab:
            return vocab[candidate]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    results = Path(args.results)

    print("UNK-token audit: does each tokenizer actually tokenize every language,")
    print("or does some script collapse to <unk>?\n")

    all_ok = True
    for model_key, mcfg in config["models"].items():
        samples_path = results / model_key / "01_samples.json"
        if not samples_path.exists():
            print(f"[{model_key}] no samples file at {samples_path}; skipping")
            continue

        tok_path = fetch_tokenizer(mcfg["hf_id"])
        tok = Tokenizer.from_file(str(tok_path))
        unk_id = find_unk_id(tok)
        samples = json.loads(samples_path.read_text(encoding="utf-8"))

        print(f"=== {model_key} ({mcfg['hf_id']}) -- unk_token_id={unk_id} ===")
        for lang, d in samples.items():
            ids = tok.encode(d["text"]).ids
            n = len(ids)
            n_unk = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
            pct = 100.0 * n_unk / n if n else 0.0
            flag = "" if pct < 0.5 else "   <-- HIGH UNK, INVESTIGATE"
            if pct >= 0.5:
                all_ok = False
            print(f"  {lang:11s} {n:6d} tokens, {n_unk:5d} unk ({pct:.4f}%){flag}")
        print()

    print("=" * 60)
    if all_ok:
        print("PASS: no language shows meaningful UNK -- every script is genuinely")
        print("tokenized into subwords, so routing reflects real processing, not")
        print("the model choking on unknown tokens.")
    else:
        print("WARNING: at least one language shows >=0.5% UNK. Its routing signal")
        print("may be a tokenization artifact -- investigate before trusting it.")


if __name__ == "__main__":
    main()
