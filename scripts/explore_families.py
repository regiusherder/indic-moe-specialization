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

Reads only results/ (no GPU). Produces:
  - family_pairwise_breakdown.txt   every within/cross-family pair's JSD, per model
  - family_outliers.txt             which language (if any) sits furthest from
                                     its own family's centroid, per model
  - ablation_per_language.png / .txt   per-language (not just per-family) loss
                                     deltas under targeted vs random ablation
  - ci_width_summary.txt           bootstrap CI width for every pair, flagging
                                     any estimate too imprecise to trust

Usage:
    python scripts/explore_families.py --results ./results --out ./results/figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["olmoe", "qwen_moe", "deepseek_moe"]
FAMILY = {
    "english": "Indo-European",
    "hindi": "Indo-Aryan", "marathi": "Indo-Aryan", "bengali": "Indo-Aryan",
    "gujarati": "Indo-Aryan", "punjabi": "Indo-Aryan", "urdu": "Indo-Aryan",
    "tamil": "Dravidian", "telugu": "Dravidian", "malayalam": "Dravidian", "kannada": "Dravidian",
}
ORDER = ["hindi", "marathi", "bengali", "gujarati", "punjabi", "urdu",
         "tamil", "telugu", "malayalam", "kannada"]


def load_jsd(results: Path, model: str):
    data = json.loads((results / model / "03_analysis" / "jsd_by_layer.json").read_text(encoding="utf-8"))
    layers = sorted(data.keys(), key=lambda k: int(k))
    lang_order = data[layers[0]]["lang_order"]
    mats = np.array([data[l]["matrix_hard"] for l in layers])
    return lang_order, mats.mean(axis=0)


def pairwise_breakdown(results: Path, out: Path):
    """Every within-family and cross-family pair's JSD, not just the family
    mean. This is where a reader sees whether "family-structured" is true
    uniformly or is an average hiding a messier picture."""
    lines = []
    for model in MODELS:
        lang_order, mean = load_jsd(results, model)
        lines.append("=" * 78)
        lines.append(f"{model}: full pairwise JSD, Indic languages only, grouped by family")
        lines.append("=" * 78)

        indic = [l for l in ORDER if l in lang_order]
        by_family = {}
        for l in indic:
            by_family.setdefault(FAMILY[l], []).append(l)

        for fam, langs in by_family.items():
            lines.append(f"\n  -- within {fam} ({len(langs)} languages) --")
            vals = []
            for i, a in enumerate(langs):
                for b in langs[i+1:]:
                    v = mean[lang_order.index(a), lang_order.index(b)]
                    vals.append((a, b, v))
            vals.sort(key=lambda x: x[2])
            for a, b, v in vals:
                lines.append(f"    {a:11s} - {b:11s} {v:.4f}")
            arr = np.array([v for _, _, v in vals])
            lines.append(f"    [min={arr.min():.4f} max={arr.max():.4f} "
                         f"spread(max-min)={arr.max()-arr.min():.4f}]")

        lines.append(f"\n  -- cross-family (Indo-Aryan <-> Dravidian) --")
        cross_vals = []
        for a in by_family.get("Indo-Aryan", []):
            for b in by_family.get("Dravidian", []):
                v = mean[lang_order.index(a), lang_order.index(b)]
                cross_vals.append((a, b, v))
        cross_vals.sort(key=lambda x: x[2])
        arr = np.array([v for _, _, v in cross_vals])
        lines.append(f"    lowest 3: " + ", ".join(f"{a}-{b}={v:.4f}" for a, b, v in cross_vals[:3]))
        lines.append(f"    highest 3: " + ", ".join(f"{a}-{b}={v:.4f}" for a, b, v in cross_vals[-3:]))
        lines.append(f"    [min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}]")
        lines.append("")

    (out / "family_pairwise_breakdown.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def family_outliers(results: Path, out: Path):
    """For each language, its mean JSD to same-family relatives vs the
    family's internal cohesion. Flags any language that sits unusually far
    from its own family (an exception a reviewer would ask about)."""
    lines = []
    for model in MODELS:
        lang_order, mean = load_jsd(results, model)
        lines.append("=" * 78)
        lines.append(f"{model}: per-language distance to own-family centroid")
        lines.append("=" * 78)
        indic = [l for l in ORDER if l in lang_order]
        by_family = {}
        for l in indic:
            by_family.setdefault(FAMILY[l], []).append(l)

        for fam, langs in by_family.items():
            own_family_means = {}
            for lang in langs:
                others = [x for x in langs if x != lang]
                own_family_means[lang] = np.mean(
                    [mean[lang_order.index(lang), lang_order.index(o)] for o in others]
                )
            fam_avg = np.mean(list(own_family_means.values()))
            lines.append(f"\n  {fam}: mean-to-family-mates (family average = {fam_avg:.4f})")
            for lang, v in sorted(own_family_means.items(), key=lambda x: -x[1]):
                flag = "  <-- FURTHEST from own family" if v == max(own_family_means.values()) else ""
                lines.append(f"    {lang:11s} {v:.4f}{flag}")
        lines.append("")

    (out / "family_outliers.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def ablation_per_language(results: Path, out: Path):
    """Per-LANGUAGE (not per-family-mean) ablation deltas: does the family-
    level causal effect come from all member languages being hurt roughly
    equally, or is it driven by one or two outlier languages?"""
    lines = []
    for model in MODELS:
        df = pd.read_csv(results / model / "04_ablation" / "ablation_results.csv")
        rand_by_lang = df[df["condition"] == "random_control"].groupby("language")["delta_vs_baseline"].mean()

        lines.append("=" * 78)
        lines.append(f"{model}: per-language ablation delta (targeted vs random-control baseline)")
        lines.append("=" * 78)

        targeted = df[df["condition"] == "targeted"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        for ax, group in zip(axes, ["Indo-Aryan", "Dravidian"]):
            sub = targeted[targeted["group"] == group]
            langs_in_group = [l for l in ORDER if FAMILY.get(l) is not None and l in sub["language"].unique()]
            own_deltas = [sub[sub["language"] == l]["delta_vs_baseline"].mean() for l in langs_in_group]
            rand_deltas = [rand_by_lang.get(l, np.nan) for l in langs_in_group]

            lines.append(f"\n  ablating {group}-preferring experts, per language:")
            for lang, d, r in zip(langs_in_group, own_deltas, rand_deltas):
                tag = FAMILY[lang]
                marker = "(own family)" if tag == group else "(other family)"
                lines.append(f"    {lang:11s} {tag:13s} {marker:16s} delta={d:.4f} "
                             f"(random baseline for this lang={r:.4f}) "
                             f"{'ABOVE random' if d > r else 'at/below random'}")

            x = np.arange(len(langs_in_group))
            colors = ["tab:blue" if FAMILY[l] == "Indo-Aryan" else "tab:red" for l in langs_in_group]
            ax.bar(x, own_deltas, color=colors, alpha=0.85)
            for xi, r in zip(x, rand_deltas):
                ax.plot([xi-0.4, xi+0.4], [r, r], "k--", lw=1)
            ax.set_xticks(x); ax.set_xticklabels(langs_in_group, rotation=45, ha="right")
            ax.set_title(f"ablate {group} experts")
            ax.set_ylabel("loss delta vs baseline")
        fig.suptitle(f"{model}: per-language ablation vulnerability (dashed line = that language's own random-control baseline)")
        fig.tight_layout()
        fig.savefig(out / f"ablation_per_language_{model}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        lines.append("")

    (out / "ablation_per_language.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def ci_width_summary(results: Path, out: Path):
    """How wide are the bootstrap CIs on the pairwise JSD point estimates?
    A narrow CI relative to the point estimate means the family-structure
    finding is precise, not just directionally suggestive."""
    lines = []
    for model in MODELS:
        df = pd.read_csv(results / model / "03_analysis" / "bootstrap_cis.csv")
        # one row per (layer, lang_a, lang_b) -- average the CI width across layers per pair
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
        lines.append(f"{model}: bootstrap CI width sanity check (mean across layers)")
        lines.append("=" * 78)
        lines.append(f"  overall mean relative CI width (width / point estimate): "
                     f"{per_pair['mean_rel_width'].mean():.3f}")
        lines.append(f"\n  5 WIDEST (least precise) pairs:")
        for _, row in wide.iterrows():
            lines.append(f"    {row['lang_a']:11s}-{row['lang_b']:11s} "
                         f"point={row['mean_point']:.4f} ci_width={row['mean_ci_width']:.4f} "
                         f"rel_width={row['mean_rel_width']:.2f}")
        lines.append(f"\n  5 TIGHTEST (most precise) pairs:")
        for _, row in tight.iterrows():
            lines.append(f"    {row['lang_a']:11s}-{row['lang_b']:11s} "
                         f"point={row['mean_point']:.4f} ci_width={row['mean_ci_width']:.4f} "
                         f"rel_width={row['mean_rel_width']:.2f}")
        lines.append("")

    (out / "ci_width_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./results/figures")
    args = ap.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairwise_breakdown(results, out)
    family_outliers(results, out)
    ablation_per_language(results, out)
    ci_width_summary(results, out)

    print(f"\nAll deep-dive outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
