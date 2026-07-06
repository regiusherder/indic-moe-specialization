#!/usr/bin/env python
"""Deeper, per-language exploration of family structure and ablation, beyond
the family-level means in analyze_results.py / findings_summary.txt.

analyze_results.py answers "is routing family-structured, on average?" This
script answers the next-level questions a careful reader will ask:
  - WITHIN a family, are all language pairs equally tight, or does one
    language sit apart from its own family (an exception worth flagging)?
  - WITHIN the causal ablation, does the family-level effect come from all
    languages in that family being hurt roughly equally, or is it driven by
    one or two languages while the others are barely affected?
  - How wide are the bootstrap confidence intervals on the pairwise JSD
    estimates — are we confident to the precision the point estimates imply?

Every question is answered PER (model, condition), aggregated across seeds
(mean), reading the results/<condition>/seed<N>/<model>/... tree and the
17-language metadata from config.yaml (never a hardcoded per-script list).

Reads only results/ (no GPU). Produces, per (model, condition):
  - family_pairwise_breakdown.txt   every within/cross-family pair's JSD
  - family_outliers.txt             which language sits furthest from its
                                     own family's centroid
  - ablation_per_language_{model}_{condition}_topn{N}.png / .txt
                                     per-language loss deltas, targeted vs random
  - ci_width_summary.txt            bootstrap CI width for every pair

Usage:
    python scripts/explore_families.py --results ./results --out ./results/figures
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.results_io import (cell_has_ablation, cell_has_analysis, discover_matrix,
                            find_cells, language_metadata, load_ablation, load_config,
                            load_bootstrap_cis, load_jsd, mean_jsd_across_layers)


def indic_display_order(lang_meta):
    fam_rank = {"Indo-Aryan": 0, "Dravidian": 1}
    return sorted((l for l in lang_meta if lang_meta[l]["family"] in fam_rank),
                 key=lambda l: (fam_rank[lang_meta[l]["family"]], l))


def condition_seed_mean_jsd(results_root, model, condition, seeds):
    """Mean-across-layers-and-seeds hard JSD matrix for (model, condition), or
    None if no cell has valid analysis. Also returns lang_order and n_seeds_used."""
    per_seed = {}
    lang_order = None
    for seed in seeds:
        for _, _, _, cell_dir in find_cells(results_root, model=model, condition=condition, seed=seed):
            if not cell_has_analysis(cell_dir):
                continue
            lo, hard, soft, layers = load_jsd(cell_dir)
            if lang_order is None:
                lang_order = lo
            elif lo != lang_order:
                raise ValueError(f"{cell_dir}: lang_order mismatch across seeds for {model}/{condition}")
            mean_hard, _ = mean_jsd_across_layers(hard, soft)
            per_seed[seed] = mean_hard
    if not per_seed:
        return None, None, 0
    mean_mat = np.mean(list(per_seed.values()), axis=0)
    return lang_order, mean_mat, len(per_seed)


def pairwise_breakdown(results_root, models, conditions, seeds, lang_meta, order, out: Path):
    lines = []
    for model in models:
        for condition in conditions:
            lang_order, mean, n_seeds = condition_seed_mean_jsd(results_root, model, condition, seeds)
            if lang_order is None:
                continue
            lines.append("=" * 78)
            lines.append(f"{model} [{condition}] (n_seeds={n_seeds}): full pairwise JSD, "
                         f"Indic languages, grouped by family")
            lines.append("=" * 78)

            indic = [l for l in order if l in lang_order]
            by_family = {}
            for l in indic:
                by_family.setdefault(lang_meta[l]["family"], []).append(l)

            for fam, langs in by_family.items():
                lines.append(f"\n  -- within {fam} ({len(langs)} languages) --")
                vals = []
                for i, a in enumerate(langs):
                    for b in langs[i+1:]:
                        v = mean[lang_order.index(a), lang_order.index(b)]
                        vals.append((a, b, v))
                if not vals:
                    lines.append("    (fewer than 2 languages — nothing to compare)")
                    continue
                vals.sort(key=lambda x: x[2])
                for a, b, v in vals:
                    lines.append(f"    {a:14s} - {b:14s} {v:.4f}")
                arr = np.array([v for _, _, v in vals])
                lines.append(f"    [min={arr.min():.4f} max={arr.max():.4f} "
                             f"spread(max-min)={arr.max()-arr.min():.4f}]")

            fams_present = [f for f in by_family if f in ("Indo-Aryan", "Dravidian")]
            if len(fams_present) == 2:
                fam_a, fam_b = fams_present
                lines.append(f"\n  -- cross-family ({fam_a} <-> {fam_b}) --")
                cross_vals = []
                for a in by_family.get(fam_a, []):
                    for b in by_family.get(fam_b, []):
                        v = mean[lang_order.index(a), lang_order.index(b)]
                        cross_vals.append((a, b, v))
                if cross_vals:
                    cross_vals.sort(key=lambda x: x[2])
                    arr = np.array([v for _, _, v in cross_vals])
                    lines.append(f"    lowest 3: " + ", ".join(f"{a}-{b}={v:.4f}" for a, b, v in cross_vals[:3]))
                    lines.append(f"    highest 3: " + ", ".join(f"{a}-{b}={v:.4f}" for a, b, v in cross_vals[-3:]))
                    lines.append(f"    [min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}]")
            lines.append("")

    (out / "family_pairwise_breakdown.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def family_outliers(results_root, models, conditions, seeds, lang_meta, order, out: Path):
    lines = []
    for model in models:
        for condition in conditions:
            lang_order, mean, n_seeds = condition_seed_mean_jsd(results_root, model, condition, seeds)
            if lang_order is None:
                continue
            lines.append("=" * 78)
            lines.append(f"{model} [{condition}] (n_seeds={n_seeds}): per-language distance to own-family centroid")
            lines.append("=" * 78)
            indic = [l for l in order if l in lang_order]
            by_family = {}
            for l in indic:
                by_family.setdefault(lang_meta[l]["family"], []).append(l)

            for fam, langs in by_family.items():
                if len(langs) < 2:
                    continue
                own_family_means = {}
                for lang in langs:
                    others = [x for x in langs if x != lang]
                    own_family_means[lang] = np.mean(
                        [mean[lang_order.index(lang), lang_order.index(o)] for o in others]
                    )
                fam_avg = np.mean(list(own_family_means.values()))
                lines.append(f"\n  {fam}: mean-to-family-mates (family average = {fam_avg:.4f})")
                max_v = max(own_family_means.values())
                for lang, v in sorted(own_family_means.items(), key=lambda x: -x[1]):
                    flag = "  <-- FURTHEST from own family" if v == max_v else ""
                    lines.append(f"    {lang:14s} {v:.4f}{flag}")
            lines.append("")

    (out / "family_outliers.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def ablation_per_language(results_root, models, conditions, seeds, lang_meta, order, out: Path):
    lines = []
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
                rand_by_lang = tdf[tdf["condition"] == "random_control"].groupby("language")["delta_mean"].mean()
                targeted = tdf[tdf["condition"] == "targeted"]
                groups = [g for g in ("Indo-Aryan", "Dravidian") if g in targeted["group"].unique()]
                if not groups:
                    continue

                lines.append("=" * 78)
                lines.append(f"{model} [{condition}] top_n={top_n} (n_seeds={tdf['seed'].nunique()}): "
                             f"per-language ablation delta (targeted vs random-control baseline)")
                lines.append("=" * 78)

                fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5.5), sharey=True, squeeze=False)
                axes = axes[0]
                for ax, group in zip(axes, groups):
                    sub = targeted[targeted["group"] == group]
                    langs_in_group = [l for l in order if l in sub["language"].unique()]
                    own_deltas = [sub[sub["language"] == l]["delta_mean"].mean() for l in langs_in_group]
                    rand_deltas = [rand_by_lang.get(l, np.nan) for l in langs_in_group]

                    lines.append(f"\n  ablating {group}-preferring experts, per language:")
                    for lang, d, r in zip(langs_in_group, own_deltas, rand_deltas):
                        tag = lang_meta[lang]["family"]
                        marker = "(own family)" if tag == group else "(other family)"
                        rtxt = f"{r:.4f}" if pd.notna(r) else "n/a"
                        cmp_txt = ("ABOVE random" if pd.notna(r) and d > r
                                  else "at/below random" if pd.notna(r) else "no random baseline")
                        lines.append(f"    {lang:14s} {tag:13s} {marker:16s} delta={d:.4f} "
                                     f"(random baseline for this lang={rtxt}) {cmp_txt}")

                    x = np.arange(len(langs_in_group))
                    colors = ["tab:blue" if lang_meta[l]["family"] == "Indo-Aryan" else "tab:red"
                             for l in langs_in_group]
                    ax.bar(x, own_deltas, color=colors, alpha=0.85)
                    for xi, r in zip(x, rand_deltas):
                        if pd.notna(r):
                            ax.plot([xi-0.4, xi+0.4], [r, r], "k--", lw=1)
                    ax.set_xticks(x); ax.set_xticklabels(langs_in_group, rotation=45, ha="right")
                    ax.set_title(f"ablate {group} experts")
                    ax.set_ylabel("loss delta vs baseline")
                fig.suptitle(f"{model} [{condition}] top_n={top_n}: per-language ablation vulnerability\n"
                            f"(dashed = that language's own random-control baseline; "
                            f"mean across {tdf['seed'].nunique()} seed(s))")
                fig.tight_layout()
                fig.savefig(out / f"ablation_per_language_{model}_{condition}_topn{top_n}.png",
                           dpi=600, bbox_inches="tight")
                plt.close(fig)
                lines.append("")

    (out / "ablation_per_language.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def ci_width_summary(results_root, models, conditions, seeds, out: Path):
    lines = []
    for model in models:
        for condition in conditions:
            dfs = []
            for seed in seeds:
                for _, _, _, cell_dir in find_cells(results_root, model=model, condition=condition, seed=seed):
                    if cell_has_analysis(cell_dir):
                        d = load_bootstrap_cis(cell_dir)
                        d["seed"] = seed
                        dfs.append(d)
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
            df["ci_width"] = df["ci_high"] - df["ci_low"]
            df["rel_width"] = df["ci_width"] / df["point_estimate"].replace(0, np.nan)
            per_pair = df.groupby(["lang_a", "lang_b"]).agg(
                mean_point=("point_estimate", "mean"),
                mean_ci_width=("ci_width", "mean"),
                mean_rel_width=("rel_width", "mean"),
            ).reset_index()
            wide = per_pair.sort_values("mean_rel_width", ascending=False).head(5)
            tight = per_pair.sort_values("mean_rel_width", ascending=True).head(5)

            lines.append("=" * 78)
            lines.append(f"{model} [{condition}] (n_seeds={df['seed'].nunique()}): "
                         f"bootstrap CI width sanity check (mean across layers + seeds)")
            lines.append("=" * 78)
            lines.append(f"  overall mean relative CI width (width / point estimate): "
                         f"{per_pair['mean_rel_width'].mean():.3f}")
            lines.append(f"\n  5 WIDEST (least precise) pairs:")
            for _, row in wide.iterrows():
                lines.append(f"    {row['lang_a']:14s}-{row['lang_b']:14s} "
                             f"point={row['mean_point']:.4f} ci_width={row['mean_ci_width']:.4f} "
                             f"rel_width={row['mean_rel_width']:.2f}")
            lines.append(f"\n  5 TIGHTEST (most precise) pairs:")
            for _, row in tight.iterrows():
                lines.append(f"    {row['lang_a']:14s}-{row['lang_b']:14s} "
                             f"point={row['mean_point']:.4f} ci_width={row['mean_ci_width']:.4f} "
                             f"rel_width={row['mean_rel_width']:.2f}")
            lines.append("")

    (out / "ci_width_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./results/figures")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    lang_meta = language_metadata(config)
    order = indic_display_order(lang_meta)

    conditions, seeds, models = discover_matrix(results)
    if not models:
        raise SystemExit(f"No (condition/seed/model) cells found under {results.resolve()}.")
    print(f"Discovered: conditions={conditions} seeds={seeds} models={models}")

    pairwise_breakdown(results, models, conditions, seeds, lang_meta, order, out)
    family_outliers(results, models, conditions, seeds, lang_meta, order, out)
    ablation_per_language(results, models, conditions, seeds, lang_meta, order, out)
    ci_width_summary(results, models, conditions, seeds, out)

    print(f"\nAll deep-dive outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
