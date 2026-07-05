#!/usr/bin/env python
"""Three additional robustness checks the earlier analysis scripts computed
the inputs for but never actually ran:

1. SOFT vs HARD metric agreement. The pipeline computes JSD on both hard
   top-k expert-selection counts (primary) and the full soft routing
   distribution (secondary, meant as a robustness check per config.yaml) --
   but no script ever compared them. If the headline family-separation and
   Hindi-Urdu findings only show up under one metric, that's a red flag for
   the metric choice rather than a real routing phenomenon. This checks
   whether hard and soft rankings agree (Spearman correlation across all
   pairwise JSDs) and whether the SAME headline numbers (family ratio,
   Hindi-Urdu ratio) point the same direction under both metrics.

2. FERTILITY-RESIDUAL CONFOUND CHECK. Token-capped sampling was designed to
   equalize routing decisions per language regardless of tokenizer fertility
   (tokens/char) -- but this was never directly tested against the actual
   JSD-vs-English values. If fertility itself correlates with JSD-vs-English
   even after token-capping, that means SOMETHING about high-fertility
   scripts is still driving apparent divergence (e.g. more fragmented
   subwords could plausibly land on more varied experts for reasons having
   nothing to do with linguistic family) and the confound isn't fully closed.
   Runs a Spearman correlation and reports it plainly, honest either way.

3. GROUP-LEVEL WITHIN-vs-CROSS PERMUTATION TEST. Earlier permutation testing
   (in the main pipeline) is per LANGUAGE PAIR. It never tests the specific
   claim "family-structured routing" makes at the GROUP level: is the gap
   between {within-family JSDs} and {cross-family JSDs}, as two sets, itself
   larger than chance? This shuffles family-group labels among the Indic
   languages and rebuilds a null for the within/cross JSD gap -- a genuinely
   different (stronger, more direct) test of the family-structure claim than
   "some pairs are significant."

Reads only results/ (no GPU). Writes robustness_checks.txt to --out.

Usage:
    python scripts/robustness_checks.py --results ./results --out ./results/figures
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MODELS = ["olmoe", "qwen_moe", "deepseek_moe"]
FAMILY = {
    "english": "Indo-European",
    "hindi": "Indo-Aryan", "marathi": "Indo-Aryan", "bengali": "Indo-Aryan",
    "gujarati": "Indo-Aryan", "punjabi": "Indo-Aryan", "urdu": "Indo-Aryan",
    "tamil": "Dravidian", "telugu": "Dravidian", "malayalam": "Dravidian", "kannada": "Dravidian",
}


def load_jsd_both(results: Path, model: str):
    data = json.loads((results / model / "03_analysis" / "jsd_by_layer.json").read_text(encoding="utf-8"))
    layers = sorted(data.keys(), key=lambda k: int(k))
    lang_order = data[layers[0]]["lang_order"]
    hard = np.array([data[l]["matrix_hard"] for l in layers]).mean(axis=0)
    soft = np.array([data[l]["matrix_soft"] for l in layers]).mean(axis=0)
    return lang_order, hard, soft


def pairwise_vector(mat, lang_order):
    """Upper-triangle values as a flat vector, with the (a,b) label list in
    the same order, so two matrices (e.g. hard vs soft) can be compared pair
    by pair."""
    n = len(lang_order)
    vals, labels = [], []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(mat[i, j])
            labels.append((lang_order[i], lang_order[j]))
    return np.array(vals), labels


def check_soft_vs_hard(results: Path, lines: list):
    lines.append("=" * 78)
    lines.append("CHECK 1: does the SOFT-routing metric agree with the HARD (primary) metric?")
    lines.append("=" * 78)
    lines.append("(Spearman rank correlation across all pairwise JSDs; then whether the two")
    lines.append(" headline ratios -- family separation, Hindi-Urdu -- point the same direction)\n")

    for model in MODELS:
        lang_order, hard, soft = load_jsd_both(results, model)
        hard_vals, labels = pairwise_vector(hard, lang_order)
        soft_vals, _ = pairwise_vector(soft, lang_order)
        rho, pval = spearmanr(hard_vals, soft_vals)

        def jsd(mat, a, b):
            return mat[lang_order.index(a), lang_order.index(b)]

        indic = [l for l in lang_order if l != "english"]
        def fam_ratio(mat):
            within, cross = [], []
            for i, a in enumerate(indic):
                for b in indic[i+1:]:
                    (within if FAMILY[a] == FAMILY[b] else cross).append(jsd(mat, a, b))
            return np.mean(within) / np.mean(cross)

        def hu_ratio(mat):
            hu = jsd(mat, "hindi", "urdu")
            other = np.mean([jsd(mat, "hindi", x) for x in ["marathi", "bengali", "gujarati", "punjabi"]])
            return hu / other

        hard_fam, soft_fam = fam_ratio(hard), fam_ratio(soft)
        hard_hu, soft_hu = hu_ratio(hard), hu_ratio(soft)

        lines.append(f"{model}:")
        lines.append(f"  Spearman rho (hard vs soft, all {len(hard_vals)} pairs): "
                     f"{rho:.3f} (p={pval:.2e})")
        lines.append(f"  family within/cross ratio:  hard={hard_fam:.3f}  soft={soft_fam:.3f}  "
                     f"{'AGREE (both <1)' if (hard_fam < 1) == (soft_fam < 1) else 'DISAGREE'}")
        lines.append(f"  Hindi-Urdu ratio:           hard={hard_hu:.3f}  soft={soft_hu:.3f}  "
                     f"{'AGREE (both same side of 1)' if (hard_hu < 1) == (soft_hu < 1) else 'DISAGREE'}")
        lines.append("")


def check_fertility_confound(results: Path, lines: list):
    lines.append("=" * 78)
    lines.append("CHECK 2: residual fertility confound -- does tokens/char still predict")
    lines.append("JSD-vs-English even after token-capped sampling?")
    lines.append("=" * 78)
    lines.append("(Spearman correlation across the 10 Indic languages; token-capping should")
    lines.append(" have equalized routing-decision COUNT, but not necessarily eliminate any")
    lines.append(" relationship between how fragmented a script is and how it routes)\n")

    for model in MODELS:
        samples = json.loads((results / model / "01_samples.json").read_text(encoding="utf-8"))
        lang_order, hard, soft = load_jsd_both(results, model)
        ei = lang_order.index("english")

        ferts, jsds, langs = [], [], []
        for lang in lang_order:
            if lang == "english":
                continue
            fert = samples[lang]["n_tokens"] / len(samples[lang]["text"])
            jsds.append(hard[ei, lang_order.index(lang)])
            ferts.append(fert)
            langs.append(lang)

        rho, pval = spearmanr(ferts, jsds)
        lines.append(f"{model}: fertility vs JSD-vs-English, n={len(langs)} languages")
        lines.append(f"  Spearman rho={rho:.3f}  p={pval:.3f}  "
                     f"{'SIGNIFICANT correlation -- interpret family/script findings with this in mind' if pval < 0.05 else 'not significant at p<0.05'}")
        order = np.argsort(ferts)
        lines.append(f"  (sorted by fertility) " +
                     ", ".join(f"{langs[i]}={ferts[i]:.2f}/{jsds[i]:.3f}" for i in order))
        lines.append("")


def check_group_level_permutation(results: Path, lines: list, n_permutations: int, seed: int):
    lines.append("=" * 78)
    lines.append("CHECK 3: GROUP-level permutation test -- is within-vs-cross-family JSD gap")
    lines.append("itself larger than chance (not just individual pair significance)?")
    lines.append("=" * 78)
    lines.append(f"(Shuffle Indo-Aryan/Dravidian family labels among the 10 Indic languages,")
    lines.append(f" {n_permutations} times, rebuild the null distribution of the within/cross gap)\n")

    rng = np.random.default_rng(seed)
    for model in MODELS:
        lang_order, hard, soft = load_jsd_both(results, model)
        indic = [l for l in lang_order if l != "english"]
        n_ia = sum(1 for l in indic if FAMILY[l] == "Indo-Aryan")  # 6
        n_dr = len(indic) - n_ia  # 4

        def group_gap(labels_map):
            within, cross = [], []
            for i, a in enumerate(indic):
                for b in indic[i+1:]:
                    v = hard[lang_order.index(a), lang_order.index(b)]
                    (within if labels_map[a] == labels_map[b] else cross).append(v)
            return np.mean(cross) - np.mean(within)  # positive = family-structured

        observed_gap = group_gap(FAMILY)

        null_gaps = np.empty(n_permutations)
        for k in range(n_permutations):
            perm = rng.permutation(indic)
            fake_labels = {lang: ("Indo-Aryan" if i < n_ia else "Dravidian")
                          for i, lang in enumerate(perm)}
            null_gaps[k] = group_gap(fake_labels)

        p_value = (1 + (null_gaps >= observed_gap).sum()) / (1 + n_permutations)
        lines.append(f"{model}: observed (cross-within) gap = {observed_gap:.4f}")
        lines.append(f"  null mean={null_gaps.mean():.4f} null std={null_gaps.std():.4f}")
        lines.append(f"  p-value = {p_value:.4f} "
                     f"{'(SIGNIFICANT: the true Indo-Aryan/Dravidian split explains the JSD gap better than a random relabeling would)' if p_value < 0.05 else '(not significant)'}")
        lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./results/figures")
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    check_soft_vs_hard(results, lines)
    check_fertility_confound(results, lines)
    check_group_level_permutation(results, lines, args.n_permutations, args.seed)

    text = "\n".join(lines)
    (out / "robustness_checks.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {(out / 'robustness_checks.txt').resolve()}")


if __name__ == "__main__":
    main()
