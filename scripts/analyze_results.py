#!/usr/bin/env python
"""Analyze the study's results/ folder into figures + a text summary of the
key findings. Reads only the artifacts run_all.sh / analyse_all.sh produce --
no GPU, no models, runs on a laptop in seconds.

Usage:
    python scripts/analyze_results.py --results ./results-from-runpod --out ./figures

Tree layout: results/<condition>/seed<seed>/<model>/{03_analysis,04_ablation}/...
where condition in {token_capped, aligned} and there are 2 seeds per condition.
The whole point of running both conditions and both seeds is that the
headline findings should SURVIVE all four (condition, seed) combinations per
model -- so every figure/number here is reported per condition, aggregated
across seeds (mean + spread), rather than silently picking one cell.

Produces, per (model, condition):
  - jsd_heatmap_{model}_{condition}.png   mean-across-layers-and-seeds JSD matrix
  - dendrogram_{model}_{condition}.png    hierarchical clustering of languages
  - layerwise_{model}_{condition}.png     JSD vs English, per language, by depth
And across all models/conditions:
  - script_pair_controls.png   Hindi-Urdu AND Kashmiri-Deva/Arab, both conditions
  - ablation_{model}_{condition}_topn{N}.png   targeted vs random deltas by family
  - findings_summary.txt       the numbers, in plain text, for the write-up
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.results_io import (cell_has_ablation, cell_has_analysis, discover_matrix,
                            find_cells, language_metadata, load_ablation, load_config,
                            load_jsd, load_permutation_tests, mean_jsd_across_layers,
                            reorder_matrix, script_pairs, seed_aggregate)


def display_order(lang_meta):
    """English first, then Indo-Aryan (Devanagari, then other scripts), then
    Dravidian -- mirrors config.yaml's grouping for readable figures."""
    fam_rank = {"Indo-European": 0, "Indo-Aryan": 1, "Dravidian": 2}
    script_rank = {"Devanagari": 0}
    return sorted(lang_meta.keys(),
                 key=lambda l: (fam_rank.get(lang_meta[l]["family"], 9),
                                script_rank.get(lang_meta[l]["script"], 1), l))


def load_cell_mean_jsd(cell_dir):
    lang_order, hard, soft, layers = load_jsd(cell_dir)
    mean_hard, mean_soft = mean_jsd_across_layers(hard, soft)
    return lang_order, mean_hard, mean_soft, hard, layers


def aggregate_condition_jsd(results_root, model, condition, seeds):
    """Mean + per-seed spread of the (already layer-averaged) hard JSD matrix
    across the seeds available for (model, condition). Returns
    (lang_order, mean_mat, spread_mat, per_seed_layerwise_hard, layers) or None
    if no cell for this (model, condition) has valid analysis."""
    per_seed_mean, per_seed_hard, lang_order, layers = {}, {}, None, None
    for seed in seeds:
        cells = find_cells(results_root, model=model, condition=condition, seed=seed)
        for _, _, _, cell_dir in cells:
            if not cell_has_analysis(cell_dir):
                continue
            lo, mean_hard, mean_soft, hard, ly = load_cell_mean_jsd(cell_dir)
            if lang_order is None:
                lang_order, layers = lo, ly
            elif lo != lang_order:
                raise ValueError(f"{cell_dir}: lang_order differs from other seeds for "
                                 f"{model}/{condition} -- cannot aggregate across seeds.")
            per_seed_mean[seed] = mean_hard
            per_seed_hard[seed] = hard
    if not per_seed_mean:
        return None
    mean_mat, spread_mat = seed_aggregate(per_seed_mean)
    return lang_order, mean_mat, spread_mat, per_seed_hard, layers


def fig_heatmap(mean_jsd, lang_order, model, condition, order, out: Path):
    m, names = reorder_matrix(mean_jsd, lang_order, order)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(m, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i,j]:.3f}", ha="center", va="center",
                    color="white" if m[i, j] < m.max() * 0.6 else "black", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="mean JSD across layers + seeds")
    ax.set_title(f"{model} [{condition}]: routing-distribution JSD between languages\n"
                f"(hard top-k, averaged over layers and seeds)")
    fig.tight_layout()
    fig.savefig(out / f"jsd_heatmap_{model}_{condition}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_dendrogram(mean_jsd, lang_order, model, condition, order, out: Path):
    m, names = reorder_matrix(mean_jsd, lang_order, order)
    if len(names) < 3:
        return
    m = np.asarray(m, dtype=float)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    condensed = squareform(m, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, labels=names, ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_ylabel("JSD (average linkage)")
    ax.set_title(f"{model} [{condition}]: language clustering by routing similarity")
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    fig.savefig(out / f"dendrogram_{model}_{condition}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_layerwise(per_seed_hard, lang_order, layers, model, condition, lang_meta, order, out: Path):
    if "english" not in lang_order:
        return
    eng = lang_order.index("english")
    # average layerwise curves across seeds
    stacked = np.stack(list(per_seed_hard.values()), axis=0)  # (seed, layer, n, n)
    mean_by_layer = stacked.mean(axis=0)  # (layer, n, n)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fam_color = {"Indo-Aryan": "tab:blue", "Dravidian": "tab:red", "Indo-European": "tab:gray"}
    for lang in order:
        if lang not in lang_order or lang == "english":
            continue
        i = lang_order.index(lang)
        color = fam_color.get(lang_meta[lang]["family"], "gray")
        ax.plot(layers, mean_by_layer[:, eng, i], marker="o", ms=3, color=color, alpha=0.75, label=lang)
    ax.set_xlabel("layer"); ax.set_ylabel("JSD vs English (mean across seeds)")
    ax.set_title(f"{model} [{condition}]: layer-wise routing divergence from English\n"
                f"(blue=Indo-Aryan, red=Dravidian)")
    ax.legend(fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / f"layerwise_{model}_{condition}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def script_pair_analysis(results_root, models, conditions, seeds, lang_meta, pairs, summary: list):
    """Generalized script-vs-language-identity control for EVERY pair_id in
    config.yaml (currently hindustani=hindi/urdu, kashmiri=kashmiri_deva/arab),
    reported per (model, condition), aggregated across seeds. Low same-pair JSD
    relative to the language's other same-family relatives => routing tracks
    language identity, not script."""
    summary.append("=" * 78)
    summary.append("SCRIPT-vs-LANGUAGE-IDENTITY CONTROLS (all script-pairs in config.yaml)")
    summary.append("=" * 78)
    rows = []
    for model in models:
        for condition in conditions:
            agg = aggregate_condition_jsd(results_root, model, condition, seeds)
            if agg is None:
                continue
            lang_order, mean_mat, spread_mat, _, _ = agg
            for pair_id, (a, b) in [(pid, langs) for pid, langs in pairs.items() if len(langs) == 2]:
                if a not in lang_order or b not in lang_order:
                    continue
                pair_jsd = mean_mat[lang_order.index(a), lang_order.index(b)]
                fam = lang_meta[a]["family"]
                relatives = [l for l in lang_order if l not in (a, b)
                            and lang_meta[l]["family"] == fam and lang_meta[l].get("pair_id") != lang_meta[a].get("pair_id")]
                if not relatives:
                    continue
                other_vals = [mean_mat[lang_order.index(a), lang_order.index(r)] for r in relatives]
                ratio = pair_jsd / np.mean(other_vals) if np.mean(other_vals) > 0 else float("nan")
                rows.append({"model": model, "condition": condition, "pair_id": pair_id,
                            "lang_a": a, "lang_b": b, "pair_jsd": pair_jsd,
                            "other_mean": np.mean(other_vals), "ratio": ratio})
                summary.append(f"\n{model} [{condition}] {pair_id} ({a}-{b}, same language/diff script):")
                summary.append(f"  {a}-{b} JSD (same lang, diff script):      {pair_jsd:.4f}")
                summary.append(f"  {a} vs other {fam} relatives (mean, n={len(relatives)}): {np.mean(other_vals):.4f}")
                verdict = "language identity > script" if pair_jsd < np.mean(other_vals) else "script effects dominant"
                summary.append(f"  -> {verdict} (ratio {ratio:.2f})")
    return pd.DataFrame(rows)


def fig_script_pair_controls(df, out: Path):
    if df.empty:
        return
    pair_ids = sorted(df["pair_id"].unique())
    fig, axes = plt.subplots(1, len(pair_ids), figsize=(6.5 * len(pair_ids), 5), squeeze=False)
    axes = axes[0]
    for ax, pair_id in zip(axes, pair_ids):
        sub = df[df["pair_id"] == pair_id]
        labels = [f"{m}\n{c}" for m, c in zip(sub["model"], sub["condition"])]
        x = np.arange(len(sub))
        w = 0.38
        ax.bar(x - w/2, sub["pair_jsd"], w, label="same lang, diff script", color="tab:green")
        ax.bar(x + w/2, sub["other_mean"], w, label="vs other same-family langs", color="tab:gray")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("mean JSD (layers + seeds)")
        ax.set_title(f"{pair_id}: {sub.iloc[0]['lang_a']}-{sub.iloc[0]['lang_b']}")
        ax.legend(fontsize=7)
    fig.suptitle("Script-vs-language-identity controls across models/conditions\n"
                "(lower green bar => routing tracks language, not script)")
    fig.tight_layout()
    fig.savefig(out / "script_pair_controls.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def family_clustering_score(results_root, models, conditions, seeds, lang_meta, summary: list):
    summary.append("\n" + "=" * 78)
    summary.append("LANGUAGE-FAMILY SEPARATION (within vs cross-family routing JSD)")
    summary.append("=" * 78)
    for model in models:
        for condition in conditions:
            agg = aggregate_condition_jsd(results_root, model, condition, seeds)
            if agg is None:
                continue
            lang_order, mean_mat, spread_mat, _, _ = agg
            indic = [l for l in lang_order if lang_meta[l]["family"] != "Indo-European"]
            within, cross = [], []
            for i, a in enumerate(indic):
                for b in indic[i+1:]:
                    v = mean_mat[lang_order.index(a), lang_order.index(b)]
                    (within if lang_meta[a]["family"] == lang_meta[b]["family"] else cross).append(v)
            if not within or not cross:
                continue
            ratio = np.mean(within) / np.mean(cross)
            summary.append(f"\n{model} [{condition}]:")
            summary.append(f"  within-family mean JSD: {np.mean(within):.4f}")
            summary.append(f"  cross-family mean JSD:  {np.mean(cross):.4f}")
            summary.append(f"  ratio (within/cross):   {ratio:.3f}  "
                          f"({'family-structured' if ratio < 1 else 'no family structure'})")


def ablation_analysis(results_root, models, conditions, seeds, lang_meta, summary: list, out: Path):
    """Per (model, condition, top_n): targeted vs random-control ablation deltas
    by family, aggregated (mean) across seeds and across trials. `delta_mean` is
    ALREADY the per-sentence loss delta vs that language's own baseline (see
    src/results_io.py docstring) -- there is no separate baseline row/column to
    subtract."""
    summary.append("\n" + "=" * 78)
    summary.append("CAUSAL ABLATION (targeted family experts vs random controls)")
    summary.append("=" * 78)
    all_families = sorted({m["family"] for m in lang_meta.values()})
    for model in models:
        for condition in conditions:
            dfs = []
            for seed in seeds:
                for _, _, _, cell_dir in find_cells(results_root, model=model, condition=condition, seed=seed):
                    if cell_has_ablation(cell_dir):
                        d = load_ablation(cell_dir)
                        d["seed"] = seed
                        dfs.append(d)
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
            for top_n in sorted(df["top_n"].unique()):
                tdf = df[df["top_n"] == top_n]
                summary.append(f"\n{model} [{condition}] top_n={top_n} (n_seeds={tdf['seed'].nunique()}):")
                rand_df = tdf[tdf["condition"] == "random_control"]
                targeted = tdf[tdf["condition"] == "targeted"]
                groups = sorted(targeted["group"].dropna().unique())
                if not groups:
                    continue
                fams = [f for f in all_families if f in tdf["family"].unique()]
                fig, ax = plt.subplots(figsize=(9, 5))
                width = 0.8 / max(len(groups), 1)
                xpos = np.arange(len(fams))
                for gi, group in enumerate(groups):
                    deltas = [targeted[(targeted["group"] == group) & (targeted["family"] == fam)]["delta_mean"].mean()
                             if len(targeted[(targeted["group"] == group) & (targeted["family"] == fam)]) else 0.0
                             for fam in fams]
                    ax.bar(xpos + gi * width, deltas, width, label=f"ablate {group} experts")

                    own = targeted[(targeted["group"] == group) & (targeted["family"] == group)]["delta_mean"].mean()
                    rand_same_fam = rand_df[rand_df["family"] == group]["delta_mean"].mean()
                    if pd.notna(own) and pd.notna(rand_same_fam):
                        summary.append(f"  ablating {group}-preferring experts -> mean delta on {group} langs: "
                                      f"{own:.4f} (random baseline, {group} languages only: {rand_same_fam:.4f}; "
                                      f"{'SPECIFIC vs random' if own > rand_same_fam else 'NOT above random'})")
                    other_fams = [f for f in fams if f != group and f in ("Indo-Aryan", "Dravidian")]
                    for other_fam in other_fams:
                        on_other = targeted[(targeted["group"] == group) & (targeted["family"] == other_fam)]["delta_mean"].mean()
                        if pd.notna(own) and pd.notna(on_other):
                            diff = own - on_other
                            summary.append(f"      vs {other_fam}: {on_other:.4f}  differential={diff:+.4f} "
                                          f"({'family-specific' if diff > 0 else 'NOT family-specific'})")
                for fi, fam in enumerate(fams):
                    rand_fam = rand_df[rand_df["family"] == fam]["delta_mean"].mean()
                    if pd.notna(rand_fam):
                        ax.plot([fi - 0.4, fi + 0.4], [rand_fam, rand_fam], "k--", lw=1.2,
                               label="random-control (per family)" if fi == 0 else None)
                ax.set_xticks(xpos + width * (len(groups)-1) / 2)
                ax.set_xticklabels(fams)
                ax.set_ylabel("mean loss increase vs baseline")
                ax.set_title(f"{model} [{condition}] top_n={top_n}: ablation loss deltas by family\n(mean across {tdf['seed'].nunique()} seed(s))")
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(out / f"ablation_{model}_{condition}_topn{top_n}.png", dpi=600, bbox_inches="tight")
                plt.close(fig)


def significance_summary(results_root, models, conditions, seeds, summary: list):
    summary.append("\n" + "=" * 78)
    summary.append("PERMUTATION SIGNIFICANCE (sentence-level = primary)")
    summary.append("=" * 78)
    for model in models:
        for condition in conditions:
            dfs = []
            for seed in seeds:
                for _, _, _, cell_dir in find_cells(results_root, model=model, condition=condition, seed=seed):
                    if cell_has_analysis(cell_dir):
                        d = load_permutation_tests(cell_dir)
                        d["seed"] = seed
                        dfs.append(d)
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
            sent = df[df["unit"] == "sentence"]
            alpha = 0.05
            # aggregate per (lang_a, lang_b, seed) first (min/max p across layers),
            # then average the resulting fraction across seeds
            per_seed_fracs_any, per_seed_fracs_all = [], []
            for seed in sent["seed"].unique():
                s = sent[sent["seed"] == seed]
                pairs = s.groupby(["lang_a", "lang_b"])
                per_seed_fracs_any.append(pairs["p_value"].min().lt(alpha).mean())
                per_seed_fracs_all.append(pairs["p_value"].max().lt(alpha).mean())
            med_eff = sent["effect_size_sd"].median()
            summary.append(f"\n{model} [{condition}] (n_seeds={sent['seed'].nunique()}):")
            summary.append(f"  pairs significant (p<{alpha}) at >=1 layer: "
                          f"{np.mean(per_seed_fracs_any)*100:.0f}% (per-seed: "
                          f"{', '.join(f'{v*100:.0f}%' for v in per_seed_fracs_any)})")
            summary.append(f"  pairs significant (p<{alpha}) at ALL layers: {np.mean(per_seed_fracs_all)*100:.0f}%")
            summary.append(f"  median effect size (sentence-level): {med_eff:.1f} SD above null")


def condition_agreement_summary(results_root, models, conditions, seeds, lang_meta, summary: list):
    """Does the family-separation finding survive BOTH sampling conditions? This
    is the direct answer to the confound pair the two conditions exist to
    control (content-vs-precision, critiques #5/#6) -- if the ratio only shows
    up under one condition, that's the headline result of this whole rerun."""
    summary.append("\n" + "=" * 78)
    summary.append("CROSS-CONDITION AGREEMENT (does family structure survive BOTH conditions?)")
    summary.append("=" * 78)
    for model in models:
        ratios = {}
        for condition in conditions:
            agg = aggregate_condition_jsd(results_root, model, condition, seeds)
            if agg is None:
                continue
            lang_order, mean_mat, _, _, _ = agg
            indic = [l for l in lang_order if lang_meta[l]["family"] != "Indo-European"]
            within, cross = [], []
            for i, a in enumerate(indic):
                for b in indic[i+1:]:
                    v = mean_mat[lang_order.index(a), lang_order.index(b)]
                    (within if lang_meta[a]["family"] == lang_meta[b]["family"] else cross).append(v)
            if within and cross:
                ratios[condition] = np.mean(within) / np.mean(cross)
        if len(ratios) < 2:
            summary.append(f"\n{model}: only {len(ratios)}/{len(conditions)} condition(s) available — "
                          f"cannot assess cross-condition agreement yet.")
            continue
        vals = list(ratios.values())
        agree = all(v < 1 for v in vals) or all(v >= 1 for v in vals)
        summary.append(f"\n{model}: " + ", ".join(f"{c}={r:.3f}" for c, r in ratios.items())
                      + f"  {'AGREE (both same side of 1)' if agree else 'DISAGREE across conditions'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to the results/ folder pulled off the pod")
    ap.add_argument("--out", default="figures", help="where to write figures + summary")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if not any((results / d).is_dir() for d in ("token_capped", "aligned")) \
       and (results / "results").exists():
        results = results / "results"

    config = load_config(args.config)
    lang_meta = language_metadata(config)
    pairs = script_pairs(config)
    order = display_order(lang_meta)

    conditions, seeds, models = discover_matrix(results)
    if not models:
        raise SystemExit(f"No (condition/seed/model) cells with content found under {results.resolve()}. "
                         f"Expected results/<condition>/seed<N>/<model>/...")
    print(f"Discovered: conditions={conditions} seeds={seeds} models={models}")

    analyzed_cells = [(c, s, m, d) for c, s, m, d in find_cells(results)
                      if cell_has_analysis(d)]
    if not analyzed_cells:
        raise SystemExit(f"No cell under {results.resolve()} has valid analysis artifacts yet. "
                         f"Run analyse_all.sh first.")
    missing_analysis = [(c, s, m) for c, s, m, d in find_cells(results) if not cell_has_analysis(d)]
    if missing_analysis:
        print(f"NOTE: {len(missing_analysis)} cell(s) still lack analysis and will be skipped: "
             f"{missing_analysis[:10]}{' ...' if len(missing_analysis) > 10 else ''}")

    summary = [f"Discovered matrix: conditions={conditions} seeds={seeds} models={models}"]

    for model in models:
        for condition in conditions:
            agg = aggregate_condition_jsd(results, model, condition, seeds)
            if agg is None:
                continue
            lang_order, mean_mat, spread_mat, per_seed_hard, layers = agg
            fig_heatmap(mean_mat, lang_order, model, condition, order, out)
            fig_dendrogram(mean_mat, lang_order, model, condition, order, out)
            fig_layerwise(per_seed_hard, lang_order, layers, model, condition, lang_meta, order, out)
            max_spread = spread_mat.max()
            summary.append(f"\n{model} [{condition}]: max pairwise seed-spread in mean-JSD = {max_spread:.4f}"
                          + (f" (n_seeds={len(per_seed_hard)})" if len(per_seed_hard) > 1 else " (only 1 seed present)"))

    pair_df = script_pair_analysis(results, models, conditions, seeds, lang_meta, pairs, summary)
    fig_script_pair_controls(pair_df, out)
    family_clustering_score(results, models, conditions, seeds, lang_meta, summary)
    condition_agreement_summary(results, models, conditions, seeds, lang_meta, summary)
    significance_summary(results, models, conditions, seeds, summary)
    ablation_analysis(results, models, conditions, seeds, lang_meta, summary, out)

    text = "\n".join(summary)
    (out / "findings_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n\nFigures + findings_summary.txt written to {out.resolve()}")


if __name__ == "__main__":
    main()
