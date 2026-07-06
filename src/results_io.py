"""Shared loader for the downstream (laptop-side, no-GPU) analysis scripts:
scripts/analyze_results.py, explore_families.py, robustness_checks.py.

Centralizes three things that changed together and must stay consistent:

  1. TREE LAYOUT. Results now live at
     results/<condition>/seed<seed>/<model>/03_analysis/... and
     .../04_ablation/ablation_results.csv -- not the old results/<model>/...
     single-condition layout. Every downstream script walks this the same way.

  2. LANGUAGE METADATA. family/script/pair_id come from config.yaml's
     `languages:` block (17 languages), never a hardcoded dict in a script --
     the old scripts' hand-copied FAMILY/SCRIPT dicts had only 11 of the 17
     languages (missing nepali, assamese, odia, sindhh, kashmiri_deva/arab),
     which would KeyError or silently drop languages from every aggregate.

  3. SEED AGGREGATION. The study runs 2 seeds per (model, condition) specifically
     to measure how much the stochastic analyses (permutation shuffles,
     bootstrap, random-expert ablation) move run-to-run. Every loader here
     returns BOTH the per-seed values and a seed-mean + seed-spread, so
     downstream scripts can report agreement rather than silently picking one
     seed or silently averaging away the disagreement signal.

Design choice: ablation CSV rows are ALREADY per-sentence-mean deltas vs that
language's own baseline (see src/ablation.py's `deltas_vs_base` -- baseline is
subtracted before the CSV is ever written). There is no "baseline" row and no
`delta_vs_baseline` column; the correct column is `delta_mean`. (The pre-split
analysis scripts referenced `delta_vs_baseline`, which never existed in the
ablation CSV schema -- that was a latent bug, never caught because those
scripts were never run against a real ablation CSV. Fixed here.)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path="config.yaml") -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def language_metadata(config: dict) -> dict:
    """{lang_name: {"family":..., "script":..., "pair_id":...}} from config.yaml.
    This is the ONLY source of truth for language metadata in the downstream
    scripts -- never hardcode a language list; config.yaml may grow languages."""
    return {name: {"family": m["family"], "script": m["script"], "pair_id": m.get("pair_id")}
            for name, m in config["languages"].items()}


def script_pairs(config: dict) -> dict:
    """{pair_id: [lang_names]} for languages sharing a pair_id -- the
    same-language-different-script controls (currently hindustani: hindi/urdu;
    kashmiri: kashmiri_deva/kashmiri_arab). Generalizes beyond a hardcoded
    Hindi-Urdu special case so a THIRD pair added later needs no script changes."""
    pairs: dict[str, list] = {}
    for name, m in config["languages"].items():
        pid = m.get("pair_id")
        if pid:
            pairs.setdefault(pid, []).append(name)
    return {pid: langs for pid, langs in pairs.items() if len(langs) >= 2}


def find_cells(results_root, model=None, condition=None, seed=None):
    """Discover (condition, seed, model, cell_dir) tuples under results_root,
    optionally filtered. A cell is valid if it has analysis artifacts (the
    downstream scripts are analysis consumers, not producers)."""
    results_root = Path(results_root)
    if not results_root.is_dir():
        # Missing dir -> no cells (not an exception): callers already check for
        # an empty result and raise a clean, actionable SystemExit; a raw
        # traceback here would be a worse user experience for the same fact.
        return []
    cells = []
    for cond_dir in sorted(results_root.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name.startswith("_"):
            continue
        if condition is not None and cond_dir.name != condition:
            continue
        for seed_dir in sorted(cond_dir.glob("seed*")):
            if not seed_dir.is_dir():
                continue
            try:
                s = int(seed_dir.name[4:])
            except ValueError:
                continue
            if seed is not None and s != seed:
                continue
            for model_dir in sorted(seed_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if model is not None and model_dir.name != model:
                    continue
                cells.append((cond_dir.name, s, model_dir.name, model_dir))
    return cells


def discover_matrix(results_root):
    """Returns (conditions, seeds, models) actually present under results_root,
    sorted, so scripts can report on whatever subset of the matrix has landed
    (e.g. only one condition synced down so far) instead of assuming all of it."""
    cells = find_cells(results_root)
    conditions = sorted({c for c, s, m, d in cells})
    seeds = sorted({s for c, s, m, d in cells})
    models = sorted({m for c, s, m, d in cells})
    return conditions, seeds, models


def cell_has_analysis(cell_dir) -> bool:
    return (cell_dir / "03_analysis" / "jsd_by_layer.json").exists() \
        and (cell_dir / "03_analysis" / "permutation_tests.csv").exists() \
        and (cell_dir / "03_analysis" / "bootstrap_cis.csv").exists()


def cell_has_ablation(cell_dir) -> bool:
    return (cell_dir / "04_ablation" / "ablation_results.csv").exists()


def load_jsd(cell_dir):
    """Returns (lang_order, hard[n_layers,n,n], soft[n_layers,n,n], layers:list[int])
    for one cell. Tolerates both the new schema-versioned JSON (with a
    "_schema_version" key to skip) and, defensively, an old unversioned one."""
    data = json.loads((cell_dir / "03_analysis" / "jsd_by_layer.json").read_text(encoding="utf-8"))
    layer_keys = sorted((k for k in data if not k.startswith("_")), key=lambda k: int(k))
    if not layer_keys:
        raise ValueError(f"{cell_dir}/03_analysis/jsd_by_layer.json has no layer entries.")
    lang_order = data[layer_keys[0]]["lang_order"]
    n = len(lang_order)
    hard = np.zeros((len(layer_keys), n, n))
    soft = np.zeros((len(layer_keys), n, n))
    for li, lk in enumerate(layer_keys):
        entry = data[lk]
        if entry["lang_order"] != lang_order:
            raise ValueError(
                f"{cell_dir}: layer {lk} has a different lang_order than layer {layer_keys[0]} "
                f"-- JSD matrices from different layers are not comparable across a mismatched "
                f"language axis. Refusing to average.")
        hard[li] = np.array(entry["matrix_hard"])
        soft[li] = np.array(entry["matrix_soft"])
    return lang_order, hard, soft, [int(k) for k in layer_keys]


def load_permutation_tests(cell_dir) -> pd.DataFrame:
    return pd.read_csv(cell_dir / "03_analysis" / "permutation_tests.csv")


def load_bootstrap_cis(cell_dir) -> pd.DataFrame:
    return pd.read_csv(cell_dir / "03_analysis" / "bootstrap_cis.csv")


def load_ablation(cell_dir) -> pd.DataFrame:
    """Ablation rows are ALREADY per-sentence-mean deltas vs baseline
    (`delta_mean`/`delta_std`/`n_sentences`), one row per
    (language, condition in {targeted,random_control}, group, top_n, trial).
    There is no separate baseline row/condition to subtract -- see module
    docstring."""
    df = pd.read_csv(cell_dir / "04_ablation" / "ablation_results.csv")
    required = {"language", "family", "condition", "group", "top_n", "trial",
                "delta_mean", "delta_std", "n_sentences"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cell_dir}/04_ablation/ablation_results.csv missing columns {missing}; "
                         f"got {list(df.columns)}. Ablation CSV schema may have changed.")
    return df


def load_samples_meta(results_root, condition) -> dict:
    """01_samples.json is written ONCE per condition (identical across seeds/
    models -- same tokenizer-independent sentence sampling), at
    results/<condition>/01_samples.json."""
    p = Path(results_root) / condition / "01_samples.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def mean_jsd_across_layers(hard, soft):
    """Collapse the layer axis: (n_layers,n,n) -> (n,n), for both metrics."""
    return hard.mean(axis=0), soft.mean(axis=0)


def seed_aggregate(per_seed_values: dict):
    """per_seed_values: {seed: scalar_or_array}. Returns (mean, spread) where
    spread is max-min across seeds (0 if only one seed) -- the simplest,
    least assumption-laden agreement measure for exactly 2 seeds; generalizes
    to N seeds without implying a distributional shape."""
    vals = list(per_seed_values.values())
    stacked = np.stack(vals, axis=0)
    mean = stacked.mean(axis=0)
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    return mean, spread


def reorder_matrix(mat, lang_order, target_order):
    """Reorder a square (n,n) matrix + its label list to `target_order`,
    silently dropping any target-order languages absent from lang_order
    (so a partial/aligned-only run doesn't crash formatting code)."""
    target = [l for l in target_order if l in lang_order]
    idx = [lang_order.index(l) for l in target]
    return mat[np.ix_(idx, idx)], target
