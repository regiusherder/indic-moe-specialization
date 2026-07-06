#!/usr/bin/env python
"""Three additional robustness checks the earlier analysis scripts computed
the inputs for but never actually ran. All three are now computed PER (model,
condition), aggregated across seeds where relevant, over the new
results/<condition>/seed<N>/<model>/... tree.

1. SOFT vs HARD metric agreement. The pipeline computes JSD on both hard
   top-k expert-selection counts (primary) and the full soft routing
   distribution (secondary, meant as a robustness check per config.yaml) --
   but no script ever compared them. If the headline family-separation and
   script-pair findings only show up under one metric, that's a red flag for
   the metric choice rather than a real routing phenomenon. This checks
   whether hard and soft rankings agree (Spearman correlation across all
   pairwise JSDs) and whether the headline ratios (family separation, each
   script-pair) point the same direction under both metrics.

2. FERTILITY-RESIDUAL CONFOUND CHECK. Token-capped sampling was designed to
   equalize routing decisions per language regardless of tokenizer fertility
   -- but this was never directly tested against the actual JSD-vs-English
   values. NOTE (fixed here): the original version of this check read a
   "text" field and a char-count-based tokens/char ratio from 01_samples.json
   that was never actually written there (01_samples.json only stores
   n_sentences + the pre-extraction n_tokens_est, which is null for the
   aligned condition) -- that would have crashed with a KeyError against real
   data. This version instead uses tokens-PER-SENTENCE, computed from the
   actual extracted routing records (LanguageRoutingRecord.n_tokens /
   n_sentences), which is available for BOTH sampling conditions and reflects
   real post-extraction token counts rather than a pre-extraction estimate.
   Runs a Spearman correlation and reports it plainly, honest either way.

3. GROUP-LEVEL WITHIN-vs-CROSS PERMUTATION TEST. Earlier permutation testing
   (in the main pipeline) is per LANGUAGE PAIR. It never tests the specific
   claim "family-structured routing" makes at the GROUP level: is the gap
   between {within-family JSDs} and {cross-family JSDs}, as two sets, itself
   larger than chance? This shuffles family-group labels among the Indic
   languages and rebuilds a null for the within/cross JSD gap.

Reads only results/ (no GPU). Writes robustness_checks.txt to --out.

Usage:
    python scripts/robustness_checks.py --results ./results --out ./results/figures
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.results_io import (cell_has_analysis, discover_matrix, find_cells,
                            language_metadata, load_config, load_jsd,
                            mean_jsd_across_layers, script_pairs)


def condition_seed_mean_jsd_both(results_root, model, condition, seeds):
    """Mean-across-layers-and-seeds (hard, soft) matrices for (model, condition)."""
    per_seed_hard, per_seed_soft = {}, {}
    lang_order = None
    for seed in seeds:
        for _, _, _, cell_dir in find_cells(results_root, model=model, condition=condition, seed=seed):
            if not cell_has_analysis(cell_dir):
                continue
            lo, hard, soft, _ = load_jsd(cell_dir)
            if lang_order is None:
                lang_order = lo
            elif lo != lang_order:
                raise ValueError(f"{cell_dir}: lang_order mismatch across seeds for {model}/{condition}")
            mh, ms = mean_jsd_across_layers(hard, soft)
            per_seed_hard[seed] = mh
            per_seed_soft[seed] = ms
    if not per_seed_hard:
        return None, None, None
    return lang_order, np.mean(list(per_seed_hard.values()), axis=0), np.mean(list(per_seed_soft.values()), axis=0)


def pairwise_vector(mat, lang_order):
    n = len(lang_order)
    vals, labels = [], []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(mat[i, j])
            labels.append((lang_order[i], lang_order[j]))
    return np.array(vals), labels


def check_soft_vs_hard(results_root, models, conditions, seeds, lang_meta, pairs, lines: list):
    lines.append("=" * 78)
    lines.append("CHECK 1: does the SOFT-routing metric agree with the HARD (primary) metric?")
    lines.append("=" * 78)
    lines.append("(Spearman rank correlation across all pairwise JSDs; then whether the")
    lines.append(" headline ratios -- family separation, each script-pair -- point the same")
    lines.append(" direction under both metrics)\n")

    for model in models:
        for condition in conditions:
            lang_order, hard, soft = condition_seed_mean_jsd_both(results_root, model, condition, seeds)
            if lang_order is None:
                continue
            hard_vals, labels = pairwise_vector(hard, lang_order)
            soft_vals, _ = pairwise_vector(soft, lang_order)
            if len(hard_vals) < 2:
                continue
            rho, pval = spearmanr(hard_vals, soft_vals)

            def jsd(mat, a, b):
                return mat[lang_order.index(a), lang_order.index(b)]

            indic = [l for l in lang_order if lang_meta[l]["family"] != "Indo-European"]
            def fam_ratio(mat):
                within, cross = [], []
                for i, a in enumerate(indic):
                    for b in indic[i+1:]:
                        (within if lang_meta[a]["family"] == lang_meta[b]["family"] else cross).append(jsd(mat, a, b))
                if not within or not cross:
                    return float("nan")
                return np.mean(within) / np.mean(cross)

            hard_fam, soft_fam = fam_ratio(hard), fam_ratio(soft)

            lines.append(f"{model} [{condition}]:")
            lines.append(f"  Spearman rho (hard vs soft, all {len(hard_vals)} pairs): "
                         f"{rho:.3f} (p={pval:.2e})")
            if np.isfinite(hard_fam) and np.isfinite(soft_fam):
                lines.append(f"  family within/cross ratio:  hard={hard_fam:.3f}  soft={soft_fam:.3f}  "
                             f"{'AGREE (both same side of 1)' if (hard_fam < 1) == (soft_fam < 1) else 'DISAGREE'}")

            for pair_id, plangs in pairs.items():
                if len(plangs) != 2 or plangs[0] not in lang_order or plangs[1] not in lang_order:
                    continue
                a, b = plangs
                fam = lang_meta[a]["family"]
                relatives = [l for l in lang_order if l not in (a, b) and lang_meta[l]["family"] == fam]
                if not relatives:
                    continue
                def pair_ratio(mat):
                    other = np.mean([jsd(mat, a, r) for r in relatives])
                    return jsd(mat, a, b) / other if other > 0 else float("nan")
                hard_r, soft_r = pair_ratio(hard), pair_ratio(soft)
                if np.isfinite(hard_r) and np.isfinite(soft_r):
                    lines.append(f"  {pair_id} ({a}-{b}) ratio:  hard={hard_r:.3f}  soft={soft_r:.3f}  "
                                 f"{'AGREE (both same side of 1)' if (hard_r < 1) == (soft_r < 1) else 'DISAGREE'}")
            lines.append("")


def check_fertility_confound(results_root, models, conditions, seeds, lines: list):
    lines.append("=" * 78)
    lines.append("CHECK 2: residual fertility-proxy confound -- does tokens-per-sentence")
    lines.append("still predict JSD-vs-English even after token-capped sampling?")
    lines.append("=" * 78)
    lines.append("(Spearman correlation across the Indic languages present. Uses ACTUAL")
    lines.append(" extracted tokens-per-sentence, from the routing .pkl files, as a fertility")
    lines.append(" proxy -- available for both sampling conditions. Token-capping equalizes")
    lines.append(" total routing-decision COUNT per language, not necessarily eliminate any")
    lines.append(" relationship between subword fragmentation and how text routes.)\n")

    for model in models:
        for condition in conditions:
            lang_order, hard, soft = condition_seed_mean_jsd_both(results_root, model, condition, seeds)
            if lang_order is None or "english" not in lang_order:
                continue
            ei = lang_order.index("english")

            # tokens-per-sentence from the actual extracted .pkl (any seed; the
            # sampling is deterministic given a condition, so seeds share the
            # same sentences/tokenization -- use the first seed with a cell).
            fert_by_lang = {}
            for seed in seeds:
                cells = {m: d for c, s, m, d in find_cells(results_root, model=model, condition=condition, seed=seed)}
                if model not in cells:
                    continue
                routing_dir = cells[model] / "02_routing_raw"
                if not routing_dir.is_dir():
                    continue
                for pkl in routing_dir.glob("*.pkl"):
                    lang = pkl.stem
                    if lang in fert_by_lang:
                        continue
                    with open(pkl, "rb") as f:
                        rec = pickle.load(f)
                    if rec.n_sentences > 0:
                        fert_by_lang[lang] = rec.n_tokens / rec.n_sentences
                break  # one seed's .pkl is enough; tokenization is deterministic per condition

            ferts, jsds, langs = [], [], []
            for lang in lang_order:
                if lang == "english" or lang not in fert_by_lang:
                    continue
                jsds.append(hard[ei, lang_order.index(lang)])
                ferts.append(fert_by_lang[lang])
                langs.append(lang)

            if len(langs) < 3:
                lines.append(f"{model} [{condition}]: too few languages with recoverable token "
                             f"counts (n={len(langs)}) — skipping.")
                lines.append("")
                continue

            rho, pval = spearmanr(ferts, jsds)
            lines.append(f"{model} [{condition}]: tokens/sentence vs JSD-vs-English, n={len(langs)} languages")
            lines.append(f"  Spearman rho={rho:.3f}  p={pval:.3f}  "
                         f"{'SIGNIFICANT correlation -- interpret family/script findings with this in mind' if pval < 0.05 else 'not significant at p<0.05'}")
            order_idx = np.argsort(ferts)
            lines.append(f"  (sorted by tokens/sentence) " +
                         ", ".join(f"{langs[i]}={ferts[i]:.2f}/{jsds[i]:.3f}" for i in order_idx))
            lines.append("")


def check_group_level_permutation(results_root, models, conditions, seeds, lang_meta, lines: list,
                                  n_permutations: int, seed_rng: int):
    lines.append("=" * 78)
    lines.append("CHECK 3: GROUP-level permutation test -- is within-vs-cross-family JSD gap")
    lines.append("itself larger than chance (not just individual pair significance)?")
    lines.append("=" * 78)
    lines.append(f"(Shuffle Indo-Aryan/Dravidian family labels among the Indic languages present,")
    lines.append(f" {n_permutations} times, rebuild the null distribution of the within/cross gap)\n")

    rng = np.random.default_rng(seed_rng)
    for model in models:
        for condition in conditions:
            lang_order, hard, soft = condition_seed_mean_jsd_both(results_root, model, condition, seeds)
            if lang_order is None:
                continue
            indic = [l for l in lang_order if lang_meta[l]["family"] in ("Indo-Aryan", "Dravidian")]
            n_ia = sum(1 for l in indic if lang_meta[l]["family"] == "Indo-Aryan")
            n_dr = len(indic) - n_ia
            if n_ia < 2 or n_dr < 2:
                lines.append(f"{model} [{condition}]: too few languages per family "
                             f"(Indo-Aryan={n_ia}, Dravidian={n_dr}) for a meaningful group permutation — skipping.")
                lines.append("")
                continue

            def group_gap(labels_map):
                within, cross = [], []
                for i, a in enumerate(indic):
                    for b in indic[i+1:]:
                        v = hard[lang_order.index(a), lang_order.index(b)]
                        (within if labels_map[a] == labels_map[b] else cross).append(v)
                return np.mean(cross) - np.mean(within)

            fam_labels = {l: lang_meta[l]["family"] for l in indic}
            observed_gap = group_gap(fam_labels)

            null_gaps = np.empty(n_permutations)
            for k in range(n_permutations):
                perm = rng.permutation(indic)
                fake_labels = {lang: ("Indo-Aryan" if i < n_ia else "Dravidian")
                              for i, lang in enumerate(perm)}
                null_gaps[k] = group_gap(fake_labels)

            p_value = (1 + (null_gaps >= observed_gap).sum()) / (1 + n_permutations)
            lines.append(f"{model} [{condition}]: observed (cross-within) gap = {observed_gap:.4f}")
            lines.append(f"  null mean={null_gaps.mean():.4f} null std={null_gaps.std():.4f}")
            lines.append(f"  p-value = {p_value:.4f} "
                         f"{'(SIGNIFICANT: the true Indo-Aryan/Dravidian split explains the JSD gap better than a random relabeling would)' if p_value < 0.05 else '(not significant)'}")
            lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./results/figures")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    lang_meta = language_metadata(config)
    pairs = script_pairs(config)

    conditions, seeds, models = discover_matrix(results)
    if not models:
        raise SystemExit(f"No (condition/seed/model) cells found under {results.resolve()}.")
    print(f"Discovered: conditions={conditions} seeds={seeds} models={models}")

    lines = []
    check_soft_vs_hard(results, models, conditions, seeds, lang_meta, pairs, lines)
    check_fertility_confound(results, models, conditions, seeds, lines)
    check_group_level_permutation(results, models, conditions, seeds, lang_meta, lines,
                                  args.n_permutations, args.seed)

    text = "\n".join(lines)
    (out / "robustness_checks.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {(out / 'robustness_checks.txt').resolve()}")


if __name__ == "__main__":
    main()
