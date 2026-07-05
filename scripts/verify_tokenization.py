#!/usr/bin/env python
"""Sanity check: confirm each model's tokenizer actually TOKENIZES every
language into real subwords, rather than falling back to <unk>, control
tokens, or byte-fallback/replacement-character garbage.

Why this matters: if a tokenizer silently emitted <unk> (or leaked a control
token like <|endoftext|>, or byte-fallback'd whole runs of a script) for one
language, the "routing distribution" for that language would reflect the
model choking on junk input, not genuine language processing — and any
specialization result built on it would be a mirage.

This checks THREE independent things per language, against the exact text
each model was actually fed (results/<model>/01_samples.json):

  1. <unk> tokens         — the classic "tokenizer doesn't know this" signal.
  2. CONTROL tokens       — tokens the tokenizer's own metadata marks
                            special=True (e.g. <|endoftext|>, <|im_start|>,
                            BOS/EOS sentence markers). These should never
                            appear inside plain running text; if they do,
                            something upstream (data prep, concatenation) is
                            leaking control tokens into content.
  3. Unicode replacement characters (U+FFFD) in the round-tripped decode —
     the signature of a byte that couldn't be cleanly re-assembled into a
     printable codepoint (a real encoding failure, as opposed to legitimate
     byte-fallback tokens that decode back to a normal character).

Note on scope: a tokenizer's `added_tokens` list also contains ORDINARY
vocabulary entries that happen to be registered as "added" for merge reasons
(e.g. OLMoE has whitespace-run tokens like a literal "  " token, marked
special=False) — these are not anomalies and are deliberately excluded from
the control-token check, which only flags entries the tokenizer itself marks
special=True.

Uses only the lightweight `tokenizers` library (pure Rust binding, no torch,
no transformers, no GPU) — runs on a laptop in seconds.

Verified result (2026-07-05, against the actual FLORES-200 samples used in
this study, re-audited with the broadened checks): 0 <unk> tokens and 0
control-token leaks in ALL 11 languages across ALL 3 models (33/33 clean).
The only flag anywhere: DeepSeek's ENGLISH sample shows 3 replacement
characters out of 19,973 tokens (0.015%), traced to em-dash/en-dash
punctuation ("USOC have the same goal — making...", "US–China relations")
in the source FLORES-200 English text triggering byte-fallback on a
tokenizer without a dedicated merged token for that punctuation. Zero
occurrences in any of the 10 Indic languages this study actually measures
specialization on — English here is only the reference baseline. See
results/figures/tokenization_audit.txt for the full log.

Usage:
    python scripts/verify_tokenization.py --results ./results
"""
import argparse
import json
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

CACHE_DIR = Path(".tokcheck_cache")
REPLACEMENT_CHAR = "�"


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


def control_token_ids(tokenizer_json_path: Path) -> dict[int, str]:
    """Only entries the tokenizer itself marks special=True — e.g.
    <|endoftext|>, <|im_start|>/<|im_end|>, BOS/EOS sentence markers. This
    deliberately excludes ordinary vocabulary entries (like OLMoE's
    whitespace-run tokens) that are registered as 'added_tokens' for
    tokenization-merge reasons but marked special=False."""
    data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    return {t["id"]: t["content"] for t in data.get("added_tokens", []) if t.get("special")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    results = Path(args.results)

    print("Tokenization integrity audit: <unk> tokens, leaked control tokens,")
    print("and Unicode replacement characters, per language per model.\n")

    all_ok = True
    for model_key, mcfg in config["models"].items():
        samples_path = results / model_key / "01_samples.json"
        if not samples_path.exists():
            print(f"[{model_key}] no samples file at {samples_path}; skipping")
            continue

        tok_path = fetch_tokenizer(mcfg["hf_id"])
        tok = Tokenizer.from_file(str(tok_path))
        unk_id = find_unk_id(tok)
        ctrl_ids = control_token_ids(tok_path)
        samples = json.loads(samples_path.read_text(encoding="utf-8"))

        # ascii-escape token contents before printing: some control tokens
        # (e.g. DeepSeek's BOS/EOS use full-width pipe U+FF5C) aren't
        # representable in the Windows console's default codepage and would
        # crash a plain print() — this is a display-only concern, unrelated
        # to whether the token appears in real text (which is what we test).
        def safe(s):
            return s.encode("ascii", "backslashreplace").decode("ascii")

        print(f"=== {model_key} ({mcfg['hf_id']}) ===")
        print(f"    unk_token_id={unk_id}")
        print(f"    control tokens checked for: {[safe(c) for c in ctrl_ids.values()]}")
        for lang, d in samples.items():
            ids = tok.encode(d["text"]).ids
            n = len(ids)
            n_unk = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
            ctrl_hits = [ctrl_ids[i] for i in ids if i in ctrl_ids]
            decoded = tok.decode(ids)
            n_replacement = decoded.count(REPLACEMENT_CHAR)

            problems = []
            if n_unk > 0:
                problems.append(f"{n_unk} unk")
            if ctrl_hits:
                problems.append(f"{len(ctrl_hits)} control-token leaks {set(safe(c) for c in ctrl_hits)}")
            if n_replacement > 0:
                problems.append(f"{n_replacement} replacement-char")
            status = "OK" if not problems else "FLAG: " + "; ".join(problems)
            if problems:
                all_ok = False
            print(f"  {lang:11s} n={n:6d}  [{status}]")
        print()

    print("=" * 70)
    if all_ok:
        print("PASS: no <unk>, no leaked control tokens, no replacement characters")
        print("in any language, in any model. Routing reflects real tokenization,")
        print("not the model choking on unknown/garbage/control input.")
    else:
        print("Some FLAGs above -- inspect the specific tokens/positions before")
        print("trusting that language's routing result. (A handful of replacement")
        print("characters traced to ordinary punctuation triggering byte-fallback")
        print("is not itself disqualifying; a language showing many, or any")
        print("leaked control tokens, would be.)")


if __name__ == "__main__":
    main()
