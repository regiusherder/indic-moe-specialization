"""Causal ablation experiment with a random-expert-ablation control.

Design (with the critique-driven fixes baked in):

  - TARGETING (critique #16): "family-preferring" experts are chosen by the
    ratio of mean target-family usage to mean other-language usage -- but only
    among experts whose OVERALL usage clears a floor (min_usage_floor_frac x
    uniform). Without that floor, the +eps smoothing lets near-zero-traffic
    experts dominate the ratio and get ablated for essentially no effect,
    which is what made the earlier ablation weak and noisy.

  - KNOCKOUT MECHANISM (critique #18): controlled by the adapter's
    knockout mode. "renorm" zeros the ablated experts' routing weight and
    renormalizes the survivors (which UPWEIGHTS the neighbors -- conflates
    removal with redistribution). "drop" removes them from top-k WITHOUT
    renormalizing, isolating the ablated experts' own contribution. We use
    "drop" as the cleaner causal test; both are available.

  - PER-SENTENCE LOSS + ERROR BARS (critique #15): loss is measured PER
    SENTENCE, not on one concatenated blob, so every ablation delta comes with
    a distribution (mean + std over sentences), not a single point estimate.

  - TOP-N SWEEP (critique #17): the whole study is run at several ablation
    sizes (e.g. 4/8/16 experts) so no conclusion hinges on one unmotivated k.

  - RANDOM CONTROL: N random-expert-ablation trials (same expert COUNT,
    different random experts), generated once and reused across languages so
    "trial k" is a consistent condition -- distinguishes "these specific
    experts matter" from "ablating any k experts hurts".
"""
import numpy as np
from tqdm import tqdm

# torch is imported lazily inside the GPU-only function so the pure targeting
# logic (top_experts_for_group, random_experts_for_group) can be imported and
# unit-tested on a laptop without torch installed.


def per_sentence_losses(model, tokenizer, sentences: list[str], max_tokens: int) -> np.ndarray:
    """Mean per-token NLL for each sentence, as an array (critique #15 -- gives
    a distribution over sentences, not a single blob number)."""
    import torch
    out = []
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
        if inputs["input_ids"].shape[1] < 2:
            continue  # can't compute LM loss on a single token
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        out.append(float(loss.item()))
    return np.array(out)


def top_experts_for_group(
    per_lang_distributions: dict[str, dict[int, np.ndarray]],
    target_langs: list[str],
    num_layers: int,
    n_experts: int,
    top_n: int,
    min_usage_floor_frac: float = 0.5,
) -> dict[int, list[int]]:
    """For each layer, the experts most disproportionately used by target_langs
    vs other languages -- restricted to experts whose OVERALL usage clears a
    floor so near-zero-traffic experts can't win the ratio (critique #16).

    min_usage_floor_frac is a fraction of uniform usage (1/n_experts): an expert
    must be used at least that much (averaged over all languages) to be eligible.
    """
    other_langs = [l for l in per_lang_distributions if l not in target_langs]
    uniform = 1.0 / n_experts
    floor = min_usage_floor_frac * uniform
    top_experts = {}
    any_layer = next(iter(per_lang_distributions.values()))
    for layer in range(num_layers):
        if layer not in any_layer:
            continue  # dense/non-MoE layer (e.g. DeepSeek layer 0)
        target_mean = np.mean([per_lang_distributions[l][layer] for l in target_langs], axis=0)
        other_mean = np.mean([per_lang_distributions[l][layer] for l in other_langs], axis=0)
        overall_mean = np.mean([per_lang_distributions[l][layer] for l in per_lang_distributions], axis=0)

        ratio = (target_mean + 1e-8) / (other_mean + 1e-8)
        # disqualify experts below the usage floor by pushing their ratio to -inf
        ratio = np.where(overall_mean >= floor, ratio, -np.inf)
        eligible = int(np.isfinite(ratio).sum())
        k = min(top_n, eligible)
        top_experts[layer] = np.argsort(-ratio)[:k].tolist()
    return top_experts


def random_experts_for_group(
    n_experts: int,
    top_n: int,
    rng: np.random.Generator,
    layers_with_moe: list[int],
) -> dict[int, list[int]]:
    return {layer: rng.choice(n_experts, size=top_n, replace=False).tolist() for layer in layers_with_moe}


def run_ablation_study(
    adapter,
    test_sentences: dict[str, list[str]],
    language_metadata: dict[str, dict],
    per_lang_layer_dist: dict[str, dict[int, np.ndarray]],
    families_to_ablate: dict[str, list[str]],
    top_n_sweep: list[int],
    n_random_controls: int,
    perplexity_max_tokens: int,
    min_usage_floor_frac: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Runs, for each top_n in the sweep: baseline (once) + targeted ablation
    per family group + N random controls, measuring PER-SENTENCE loss for every
    condition. Returns long-format rows with mean/std/n over sentences.

    families_to_ablate: {family_name: [languages in it]} -- the groups whose
    preferred experts we ablate.
    """
    results = []
    layers_with_moe = sorted(next(iter(per_lang_layer_dist.values())).keys())

    # baseline per-sentence losses (no ablation), once per language
    baseline = {}
    for lang, sents in test_sentences.items():
        baseline[lang] = per_sentence_losses(adapter.model, adapter.tokenizer, sents, perplexity_max_tokens)

    def record(lang, condition, group, top_n, trial, deltas):
        meta = language_metadata[lang]
        results.append({
            "language": lang, "family": meta["family"], "condition": condition,
            "group": group, "top_n": top_n, "trial": trial,
            "delta_mean": float(np.mean(deltas)), "delta_std": float(np.std(deltas)),
            "n_sentences": int(len(deltas)),
        })

    for top_n in top_n_sweep:
        # targeted expert sets per family, at this sweep size
        targeted_by_group = {
            fam: top_experts_for_group(
                per_lang_layer_dist, langs, adapter.num_layers,
                adapter.num_routed_experts, top_n, min_usage_floor_frac,
            )
            for fam, langs in families_to_ablate.items()
        }
        # random controls at this sweep size, generated once, reused across langs
        random_sets = [
            random_experts_for_group(adapter.num_routed_experts, top_n, rng, layers_with_moe)
            for _ in range(n_random_controls)
        ]

        for lang, sents in tqdm(test_sentences.items(), desc=f"Ablation top_n={top_n}"):
            base = baseline[lang]
            # align per-sentence: per_sentence_losses skips <2-token sentences
            # identically for every condition (same sentence list), so index i
            # corresponds across conditions.
            for group_name, experts_by_layer in targeted_by_group.items():
                with adapter.ablate_experts(experts_by_layer):
                    abl = per_sentence_losses(adapter.model, adapter.tokenizer, sents, perplexity_max_tokens)
                record(lang, "targeted", group_name, top_n, None, abl - base)

            for trial, random_experts in enumerate(random_sets):
                with adapter.ablate_experts(random_experts):
                    abl = per_sentence_losses(adapter.model, adapter.tokenizer, sents, perplexity_max_tokens)
                record(lang, "random_control", None, top_n, trial, abl - base)

    return results
