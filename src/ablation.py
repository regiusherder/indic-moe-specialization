"""Causal ablation experiment with a random-expert-ablation control.

The pilot's original ablation (see olmoe-pilot.ipynb, cell-14) compared
"language-preferring experts ablated" only against an unablated baseline.
That cannot distinguish "these specific experts matter for this language"
from "ablating ANY 8 experts increases loss for everyone" — the null
hypothesis a reviewer will raise first. This module closes that gap by also
running N_RANDOM_CONTROLS random-expert-ablation trials (same expert COUNT,
different random experts, fresh seed per trial) per layer-group, so the
targeted-ablation effect can be compared against the random-ablation
distribution rather than only against zero.
"""
import numpy as np
import torch
from tqdm import tqdm


def compute_perplexity_loss(model, tokenizer, text: str, max_tokens: int) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(outputs.loss.item())


def top_experts_for_group(
    per_lang_distributions: dict[str, dict[int, np.ndarray]],
    target_langs: list[str],
    num_layers: int,
    n_experts: int,
    top_n: int,
) -> dict[int, list[int]]:
    """For each layer, find experts most disproportionately used by target_langs
    vs. all other languages in per_lang_distributions (ratio of mean target usage
    to mean non-target usage). Returns {layer_idx: [expert_ids]}."""
    other_langs = [l for l in per_lang_distributions if l not in target_langs]
    top_experts = {}
    for layer in range(num_layers):
        if layer not in next(iter(per_lang_distributions.values())):
            continue  # dense/non-MoE layer (e.g. DeepSeek-V2-Lite layer 0)
        target_mean = np.mean([per_lang_distributions[l][layer] for l in target_langs], axis=0)
        other_mean = np.mean([per_lang_distributions[l][layer] for l in other_langs], axis=0)
        ratio = (target_mean + 1e-8) / (other_mean + 1e-8)
        top_experts[layer] = np.argsort(-ratio)[:top_n].tolist()
    return top_experts


def random_experts_for_group(
    num_layers: int,
    n_experts: int,
    top_n: int,
    rng: np.random.Generator,
    layers_with_moe: list[int],
) -> dict[int, list[int]]:
    return {layer: rng.choice(n_experts, size=top_n, replace=False).tolist() for layer in layers_with_moe}


def run_ablation_study(
    adapter,
    test_texts: dict[str, str],
    language_metadata: dict[str, dict],
    targeted_experts_by_group: dict[str, dict[int, list[int]]],
    n_random_controls: int,
    top_n_experts: int,
    perplexity_max_tokens: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Runs baseline + targeted ablation (per language-family group) + N random
    controls, for every language. Returns one row per (language, condition) —
    long format, so downstream analysis doesn't need to know the group
    structure in advance.

    The random-control expert sets are generated ONCE and reused across all
    languages, so "trial k" is the same condition (same ablated experts) for
    every language. Regenerating them per language would make each trial an
    incomparable condition and inflate the variance of exactly the
    cross-language contrast the control exists to support. The control sets
    are also independent of group (random experts don't have a family), so
    they're shared across groups too — each language pays baseline + targeted
    per group + n_random_controls forward passes, not n_random_controls per group.
    """
    results = []
    layers_with_moe = list(next(iter(targeted_experts_by_group.values())).keys())

    random_control_sets = [
        random_experts_for_group(
            num_layers=adapter.num_layers,
            n_experts=adapter.num_routed_experts,
            top_n=top_n_experts,
            rng=rng,
            layers_with_moe=layers_with_moe,
        )
        for _ in range(n_random_controls)
    ]

    for lang, text in tqdm(test_texts.items(), desc="Ablation study"):
        meta = language_metadata[lang]
        baseline_loss = compute_perplexity_loss(adapter.model, adapter.tokenizer, text, perplexity_max_tokens)
        results.append({
            "language": lang, "family": meta["family"], "condition": "baseline",
            "group": None, "trial": None, "loss": baseline_loss,
            "delta_vs_baseline": 0.0,
        })

        for group_name, experts_by_layer in targeted_experts_by_group.items():
            with adapter.ablate_experts(experts_by_layer):
                loss = compute_perplexity_loss(adapter.model, adapter.tokenizer, text, perplexity_max_tokens)
            results.append({
                "language": lang, "family": meta["family"], "condition": "targeted",
                "group": group_name, "trial": None, "loss": loss,
                "delta_vs_baseline": loss - baseline_loss,
            })

        for trial, random_experts in enumerate(random_control_sets):
            with adapter.ablate_experts(random_experts):
                loss = compute_perplexity_loss(adapter.model, adapter.tokenizer, text, perplexity_max_tokens)
            results.append({
                "language": lang, "family": meta["family"], "condition": "random_control",
                "group": None, "trial": trial, "loss": loss,
                "delta_vs_baseline": loss - baseline_loss,
            })

    return results
