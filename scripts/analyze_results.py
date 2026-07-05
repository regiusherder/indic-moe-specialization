#!/usr/bin/env python
"""Analyze the study's results/ folder into figures + a text summary of the
key findings. Reads only the artifacts run_all.sh produces — no GPU, no
models, runs on a laptop in seconds.

Usage:
    python scripts/analyze_results.py --results ./results-from-runpod --out ./figures

Produces, per model (olmoe / qwen_moe / deepseek_moe):
  - jsd_heatmap_{model}.png      mean-across-layers JSD matrix, family-ordered
  - dendrogram_{model}.png       hierarchical clustering of languages by routing
  - layerwise_{model}.png        JSD vs English, per language, across depth
And across all models:
  - hindi_urdu_control.png       the key script-vs-family test
  - ablation_{model}.png         targeted vs random-control loss deltas by family
  - findings_summary.txt         the numbers, in plain text, for the write-up
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# headless backend so this runs anywhere
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

MODELS = ["olmoe", "qwen_moe", "deepseek_moe"]

# linguistic metadata (mirrors config.yaml) for grouping/interpretation
FAMILY = {
    "english": "Indo-European",
    "hindi": "Indo-Aryan", "marathi": "Indo-Aryan", "bengali": "Indo-Aryan",
    "gujarati": "Indo-Aryan", "punjabi": "Indo-Aryan", "urdu": "Indo-Aryan",
    "tamil": "Dravidian", "telugu": "Dravidian", "malayalam": "Dravidian", "kannada": "Dravidian",
}
SCRIPT = {
    "english": "Latin", "hindi": "Devanagari", "marathi": "Devanagari",
    "bengali": "Bengali", "gujarati": "Gujarati", "punjabi": "Gurmukhi",
    "urdu": "Perso-Arabic", "tamil": "Tamil", "telugu": "Telugu",
    "malayalam": "Malayalam", "kannada": "Kannada",
}
# family-then-name display order for readable heatmaps
ORDER = ["english",
         "hindi", "marathi", "bengali", "gujarati", "punjabi", "urdu",
         "tamil", "telugu", "malayalam", "kannada"]


def load_jsd(results: Path, model: str):
    data = json.loads((results / model / "03_analysis" / "jsd_by_layer.json").read_text())
    # keys are stringified layer indices
    layers = sorted(data.keys(), key=lambda k: int(k))
    lang_order = data[layers[0]]["lang_order"]
    n_layers = len(layers)
    n = len(lang_order)
    hard = np.zeros((n_layers, n, n))
    soft = np.zeros((n_layers, n, n))
    for li, lk in enumerate(layers):
        hard[li] = np.array(data[lk]["matrix_hard"])
        soft[li] = np.array(data[lk]["matrix_soft"])
    return lang_order, hard, soft, [int(x) for x in layers]


def reorder(mat, lang_order, target=ORDER):
    target = [l for l in target if l in lang_order]
    idx = [lang_order.index(l) for l in target]
    return mat[np.ix_(idx, idx)], target


def fig_heatmap(mean_jsd, lang_order, model, out: Path):
    m, names = reorder(mean_jsd, lang_order)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(m, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i,j]:.3f}", ha="center", va="center",
                    color="white" if m[i, j] < m.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean JSD across layers")
    ax.set_title(f"{model}: routing-distribution JSD between languages\n(hard top-k, averaged over layers)")
    fig.tight_layout()
    fig.savefig(out / f"jsd_heatmap_{model}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_dendrogram(mean_jsd, lang_order, model, out: Path):
    m, names = reorder(mean_jsd, lang_order)
    # force exact symmetry + zero diagonal, then convert the square (n x n)
    # distance matrix to condensed form. squareform infers direction from
    # input shape: a 2-D square array -> condensed vector, which is what we
    # want. Tiny float asymmetries otherwise trip checks; we pass checks=False
    # after symmetrizing ourselves.
    m = np.asarray(m, dtype=float)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    condensed = squareform(m, checks=False)  # (n,n) square -> length n*(n-1)/2 vector
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=names, ax=ax, leaf_rotation=45, leaf_font_size=11)
    ax.set_ylabel("JSD (average linkage)")
    ax.set_title(f"{model}: language clustering by routing similarity\n"
                 f"(Dravidian vs Indo-Aryan; does Urdu sit with Hindi or apart?)")
    fig.subplots_adjust(bottom=0.18)  # room for rotated labels
    fig.tight_layout()
    fig.savefig(out / f"dendrogram_{model}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_layerwise(hard, lang_order, layers, model, out: Path):
    if "english" not in lang_order:
        return
    eng = lang_order.index("english")
    fig, ax = plt.subplots(figsize=(12, 6))
    for lang in ORDER:
        if lang not in lang_order or lang == "english":
            continue
        i = lang_order.index(lang)
        color = {"Indo-Aryan": "tab:blue", "Dravidian": "tab:red"}.get(FAMILY[lang], "gray")
        ax.plot(layers, hard[:, eng, i], marker="o", ms=3, color=color, alpha=0.75, label=lang)
    ax.set_xlabel("layer"); ax.set_ylabel("JSD vs English")
    ax.set_title(f"{model}: layer-wise routing divergence from English\n(blue=Indo-Aryan, red=Dravidian)")
    # legend outside the plot area so it never overlaps the lines
    ax.legend(fontsize=9, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / f"layerwise_{model}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def hindi_urdu_analysis(results: Path, summary: list):
    """The headline control: Hindi-Urdu (same language, different script) vs
    Hindi to its other Indo-Aryan relatives (different script AND some
    linguistic distance). Low Hindi-Urdu relative to the others => routing
    tracks language identity, not script."""
    summary.append("=" * 70)
    summary.append("HINDI-URDU CONTROL (script vs. language-identity)")
    summary.append("=" * 70)
    rows = []
    for model in MODELS:
        lang_order, hard, soft, layers = load_jsd(results, model)
        mean = hard.mean(axis=0)
        def jsd_between(a, b):
            return mean[lang_order.index(a), lang_order.index(b)]
        hu = jsd_between("hindi", "urdu")
        others = [("hindi", x) for x in ["marathi", "bengali", "gujarati", "punjabi"] if x in lang_order]
        other_vals = [jsd_between(a, b) for a, b in others]
        rows.append({"model": model, "hindi_urdu": hu,
                     "hindi_other_IA_mean": np.mean(other_vals),
                     "ratio": hu / np.mean(other_vals)})
        summary.append(f"\n{model}:")
        summary.append(f"  Hindi-Urdu JSD (same lang, diff script): {hu:.4f}")
        summary.append(f"  Hindi vs other Indo-Aryan (mean):        {np.mean(other_vals):.4f}")
        for (a, b), v in zip(others, other_vals):
            summary.append(f"    {a}-{b}: {v:.4f}")
        verdict = ("language identity > script" if hu < np.mean(other_vals)
                   else "script effects dominant")
        summary.append(f"  -> {verdict} (ratio {hu/np.mean(other_vals):.2f})")
    return pd.DataFrame(rows)


def fig_hindi_urdu(df, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["hindi_urdu"], w, label="Hindi-Urdu (same lang, diff script)", color="tab:green")
    ax.bar(x + w/2, df["hindi_other_IA_mean"], w, label="Hindi vs other Indo-Aryan (mean)", color="tab:gray")
    ax.set_xticks(x); ax.set_xticklabels(df["model"])
    ax.set_ylabel("mean JSD across layers")
    ax.set_title("Hindi-Urdu control across architectures\n"
                 "(Hindi-Urdu lower => routing tracks language, not script)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hindi_urdu_control.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def family_clustering_score(results: Path, summary: list):
    """Quantify how well routing separates Dravidian from Indo-Aryan: mean
    within-family JSD vs mean cross-family JSD. Ratio << 1 => strong
    family-structured routing."""
    summary.append("\n" + "=" * 70)
    summary.append("LANGUAGE-FAMILY SEPARATION (within vs cross-family routing JSD)")
    summary.append("=" * 70)
    for model in MODELS:
        lang_order, hard, soft, layers = load_jsd(results, model)
        mean = hard.mean(axis=0)
        indic = [l for l in lang_order if l != "english"]
        within, cross = [], []
        for i, a in enumerate(indic):
            for b in indic[i+1:]:
                v = mean[lang_order.index(a), lang_order.index(b)]
                (within if FAMILY[a] == FAMILY[b] else cross).append(v)
        summary.append(f"\n{model}:")
        summary.append(f"  within-family mean JSD: {np.mean(within):.4f}")
        summary.append(f"  cross-family mean JSD:  {np.mean(cross):.4f}")
        summary.append(f"  ratio (within/cross):   {np.mean(within)/np.mean(cross):.3f}"
                       f"  ({'family-structured' if np.mean(within) < np.mean(cross) else 'no family structure'})")


def ablation_analysis(results: Path, summary: list, out: Path):
    """Targeted (family-preferring experts ablated) vs random-control ablation.
    The causal claim holds if ablating a family's experts hurts THAT family's
    languages more than random experts do, and more than it hurts other families."""
    summary.append("\n" + "=" * 70)
    summary.append("CAUSAL ABLATION (targeted family experts vs random controls)")
    summary.append("=" * 70)
    for model in MODELS:
        df = pd.read_csv(results / model / "04_ablation" / "ablation_results.csv")
        summary.append(f"\n{model}:")
        # random-control baseline: mean loss increase from ablating random experts
        rand = df[df["condition"] == "random_control"]["delta_vs_baseline"].mean()
        summary.append(f"  random-control mean delta (random experts, same count): {rand:.4f}")
        targeted = df[df["condition"] == "targeted"]
        groups = sorted(targeted["group"].dropna().unique())
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / max(len(groups), 1)
        fams = ["Indo-Aryan", "Dravidian", "Indo-European"]
        xpos = np.arange(len(fams))
        for gi, group in enumerate(groups):
            deltas = []
            for fam in fams:
                sub = targeted[(targeted["group"] == group) & (targeted["family"] == fam)]
                deltas.append(sub["delta_vs_baseline"].mean() if len(sub) else 0.0)
            ax.bar(xpos + gi * width, deltas, width, label=f"ablate {group} experts")
            # summary line: does ablating this group's experts hit its own family hardest?
            own = targeted[(targeted["group"] == group) & (targeted["family"] == group)]["delta_vs_baseline"].mean()
            summary.append(f"  ablating {group}-preferring experts -> mean delta on {group} langs: "
                           f"{own:.4f} (random baseline {rand:.4f}; "
                           f"{'SPECIFIC' if own > rand else 'not above random'})")
        ax.set_xticks(xpos + width * (len(groups)-1) / 2)
        ax.set_xticklabels(fams)
        ax.axhline(rand, ls="--", color="k", lw=1, label="random-control mean")
        ax.set_ylabel("mean loss increase vs baseline")
        ax.set_title(f"{model}: expert-ablation loss deltas by language family")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"ablation_{model}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def significance_summary(results: Path, summary: list):
    """Fraction of language pairs significant under the PRIMARY (sentence-level)
    permutation test, per model."""
    summary.append("\n" + "=" * 70)
    summary.append("PERMUTATION SIGNIFICANCE (sentence-level = primary)")
    summary.append("=" * 70)
    for model in MODELS:
        df = pd.read_csv(results / model / "03_analysis" / "permutation_tests.csv")
        sent = df[df["unit"] == "sentence"]
        # collapse to one row per language pair: significant if significant at ANY layer
        # (and also report the stricter all-layers view)
        alpha = 0.05
        pairs = sent.groupby(["lang_a", "lang_b"])
        any_sig = pairs["p_value"].min().lt(alpha).mean()
        med_eff = sent["effect_size_sd"].median()
        summary.append(f"\n{model}:")
        summary.append(f"  pairs significant (p<{alpha}) at >=1 layer: {any_sig*100:.0f}%")
        summary.append(f"  median effect size (sentence-level): {med_eff:.1f} SD above null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to the results/ folder pulled off the pod")
    ap.add_argument("--out", default="figures", help="where to write figures + summary")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # tolerate being pointed either at results/ or its parent
    if not (results / "olmoe").exists() and (results / "results" / "olmoe").exists():
        results = results / "results"

    missing = [m for m in MODELS if not (results / m / "03_analysis" / "jsd_by_layer.json").exists()]
    if missing:
        raise SystemExit(f"Missing analysis outputs for: {missing}\n"
                         f"Looked under {results.resolve()}. Point --results at the folder "
                         f"containing olmoe/ qwen_moe/ deepseek_moe/.")

    summary = []
    for model in MODELS:
        lang_order, hard, soft, layers = load_jsd(results, model)
        mean = hard.mean(axis=0)
        fig_heatmap(mean, lang_order, model, out)
        fig_dendrogram(mean, lang_order, model, out)
        fig_layerwise(hard, lang_order, layers, model, out)

    hu_df = hindi_urdu_analysis(results, summary)
    fig_hindi_urdu(hu_df, out)
    family_clustering_score(results, summary)
    significance_summary(results, summary)
    ablation_analysis(results, summary, out)

    text = "\n".join(summary)
    (out / "findings_summary.txt").write_text(text)
    print(text)
    print(f"\n\nFigures + findings_summary.txt written to {out.resolve()}")


if __name__ == "__main__":
    main()
