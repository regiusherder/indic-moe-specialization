"""Routing distribution extraction, JSD divergence, per-layer permutation
testing, and bootstrap confidence intervals.

Design decisions made explicit here (see Conversation Log for the full
reasoning trail this project is built on):
  - PRIMARY metric: JSD on hard top-k EXPERT SELECTION COUNTS (bincount over
    selected_experts) — the interpretable quantity ("which expert did this
    token actually get routed to"). SECONDARY metric: JSD on the mean soft
    routing distribution (full softmax over routed experts), showing the
    finding is robust to metric choice. Both are reported side by side.
  - Permutation testing runs at EVERY layer, not one "representative" layer
    (the pilot's shortcut). Specialization strength is a live hypothesis
    about layer depth; only testing layer 8 would beg exactly the question
    the layer-wise experiment is supposed to answer.
  - PRIMARY permutation unit: SENTENCE labels. Tokens within a sentence are
    correlated (topic, subword neighborhood), so a token-level null is
    anticonservative — it's part of why the pilot showed ~90 SD effect
    sizes. Token-level results are still computed as a supplementary
    comparison (unit column in the output), but the sentence-level numbers
    are the ones to lead with.
  - Bootstrap CIs resample at the SENTENCE level for the same independence
    reason.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import jensenshannon


@dataclass
class LanguageRoutingRecord:
    """Everything captured for one (language, model) pair, sentence-resolved
    so bootstrap resampling can operate at the sentence level."""
    language: str
    lang_code: str
    n_sentences: int
    n_tokens: int
    # per_sentence_selected[layer_idx] = list of np.ndarray, one per sentence,
    # each array shape (tokens_in_sentence, top_k) of routed-expert indices
    per_sentence_selected: dict[int, list[np.ndarray]] = field(default_factory=dict)
    # per_sentence_prob_sums[layer_idx] = list of np.ndarray, one per sentence,
    # each shape (n_routed_experts,): the SUM over the sentence's tokens of the
    # full softmax routing distribution. Summing (not averaging) per sentence
    # means language-level soft distributions weight every token equally
    # (sum of sums / total tokens) instead of every sentence equally.
    per_sentence_prob_sums: dict[int, list[np.ndarray]] = field(default_factory=dict)
    # per_sentence_token_counts = list of ints, one per sentence (same across layers)
    per_sentence_token_counts: list[int] = field(default_factory=list)


def expert_distribution(selected_indices: np.ndarray, n_experts: int) -> np.ndarray:
    """selected_indices: (n_tokens, top_k) -> normalized bincount over n_experts."""
    flat = np.asarray(selected_indices).flatten()
    if flat.size == 0:
        raise ValueError("No routing decisions recorded — check hook wiring before trusting any output.")
    # bounds guard: an out-of-range index would make bincount return an array
    # of a DIFFERENT length than n_experts, which then misaligns / crashes the
    # cross-language JSD. This must never happen (top-k indices are in
    # [0, n_experts)); fail loud if it does rather than compute silently-wrong JSD.
    lo, hi = int(flat.min()), int(flat.max())
    if lo < 0 or hi >= n_experts:
        raise ValueError(
            f"selected expert index out of range: got [{lo}, {hi}] but n_experts={n_experts}. "
            f"The adapter's selected_experts are not plain routed-expert indices -- "
            f"routing distributions would be misaligned across languages.")
    counts = np.bincount(flat, minlength=n_experts).astype(float)
    # minlength guarantees len>=n_experts; the bounds check guarantees len==n_experts
    counts = counts[:n_experts]
    total = counts.sum()
    if total == 0:
        raise ValueError("No routing decisions recorded — check hook wiring before trusting any output.")
    return counts / total


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Squared Jensen-Shannon divergence in bits (base 2). scipy's
    jensenshannon returns the DISTANCE (sqrt of divergence) — squaring here
    is required or every downstream number is silently wrong by a sqrt."""
    if len(p) != len(q):
        raise ValueError(f"JSD on mismatched-length distributions ({len(p)} vs {len(q)}) -- "
                         f"expert-index spaces differ across languages, results would be meaningless.")
    d = float(jensenshannon(p, q, base=2) ** 2)
    if not np.isfinite(d):
        # scipy returns nan for degenerate inputs; our upstream guards should
        # prevent this, so surface it rather than silently poisoning permutation
        # p-values (which compare >= against a nan observed value).
        raise ValueError("JSD is non-finite -- degenerate routing distribution reached the metric.")
    return d


def soft_distribution(record: LanguageRoutingRecord, layer_idx: int) -> np.ndarray:
    """Token-weighted mean softmax routing distribution for a language at a layer."""
    total = np.sum(record.per_sentence_prob_sums[layer_idx], axis=0)
    n_tokens = sum(record.per_sentence_token_counts)
    if n_tokens == 0:
        raise ValueError("No tokens recorded — check hook wiring before trusting any output.")
    dist = total / n_tokens
    s = dist.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(
            f"soft routing distribution has non-positive/non-finite sum ({s}) at layer "
            f"{layer_idx} for '{record.language}' -- corrupt prob sums, refusing to normalize.")
    # each token's softmax sums to 1, so dist should too (up to float error)
    return dist / s


def pairwise_jsd_matrix(
    records: dict[str, LanguageRoutingRecord],
    layer_idx: int,
    n_experts: int,
    metric: str = "hard",
) -> tuple[np.ndarray, list[str]]:
    """metric='hard' uses top-k selection counts (primary); metric='soft' uses
    the token-weighted mean softmax routing distribution (secondary)."""
    langs = list(records.keys())
    n = len(langs)
    matrix = np.zeros((n, n))
    dists = {}
    for lang in langs:
        if metric == "hard":
            all_sel = np.concatenate(records[lang].per_sentence_selected[layer_idx], axis=0)
            dists[lang] = expert_distribution(all_sel, n_experts)
        elif metric == "soft":
            dists[lang] = soft_distribution(records[lang], layer_idx)
        else:
            raise ValueError(f"Unknown metric '{metric}' — 'hard' or 'soft'.")
    for i, a in enumerate(langs):
        for j, b in enumerate(langs):
            if i < j:
                d = jsd(dists[a], dists[b])
                matrix[i, j] = d
                matrix[j, i] = d
    return matrix, langs


def permutation_test_sentences(
    sents_a: list[np.ndarray],
    sents_b: list[np.ndarray],
    n_experts: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """PRIMARY significance test: shuffle SENTENCE labels between the two
    languages and rebuild the null JSD distribution. Sentences (not tokens)
    are the independent units here — token-level shuffling destroys
    within-sentence correlation and produces an anticonservative null.
    """
    all_sents = sents_a + sents_b
    n_a = len(sents_a)

    obs = jsd(
        expert_distribution(np.concatenate(sents_a, axis=0), n_experts),
        expert_distribution(np.concatenate(sents_b, axis=0), n_experts),
    )

    null = np.empty(n_permutations)
    indices = np.arange(len(all_sents))
    for k in range(n_permutations):
        perm = rng.permutation(indices)
        group_a = np.concatenate([all_sents[i] for i in perm[:n_a]], axis=0)
        group_b = np.concatenate([all_sents[i] for i in perm[n_a:]], axis=0)
        null[k] = jsd(
            expert_distribution(group_a, n_experts),
            expert_distribution(group_b, n_experts),
        )

    # add-one (Phipson & Smyth) estimator: a permutation p-value can never
    # legitimately be 0 with finite permutations
    p_value = float((1 + (null >= obs).sum()) / (1 + n_permutations))
    null_mean, null_std = float(null.mean()), float(null.std())
    effect_size = (obs - null_mean) / null_std if null_std > 0 else float("inf")
    return {
        "observed_jsd": obs,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "effect_size_sd": effect_size,
    }


def permutation_test(
    sel_a: np.ndarray,
    sel_b: np.ndarray,
    n_experts: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """SUPPLEMENTARY token-level permutation test: shuffle which language each
    token 'belongs' to. Kept for comparability with the pilot, but its null is
    anticonservative (tokens within a sentence are correlated) — report
    permutation_test_sentences as the primary result.

    Permutes token ROWS (each row = one token's full top-k expert set), not
    flattened expert slots. Slot-level shuffling (what the pilot did) splits a
    single token's k picks across both null groups, which is not a null any
    real data process could generate.
    """
    combined = np.concatenate([sel_a, sel_b], axis=0)  # (n_tokens_total, top_k)
    n_a = sel_a.shape[0]

    obs = jsd(
        expert_distribution(sel_a, n_experts),
        expert_distribution(sel_b, n_experts),
    )

    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        p = expert_distribution(combined[perm[:n_a]], n_experts)
        q = expert_distribution(combined[perm[n_a:]], n_experts)
        null[k] = jsd(p, q)

    # add-one (Phipson & Smyth) estimator: a permutation p-value can never
    # legitimately be 0 with finite permutations
    p_value = float((1 + (null >= obs).sum()) / (1 + n_permutations))
    null_mean, null_std = float(null.mean()), float(null.std())
    effect_size = (obs - null_mean) / null_std if null_std > 0 else float("inf")
    return {
        "observed_jsd": obs,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "effect_size_sd": effect_size,
    }


def bootstrap_jsd_ci(
    record_a: LanguageRoutingRecord,
    record_b: LanguageRoutingRecord,
    layer_idx: int,
    n_experts: int,
    n_resamples: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> dict:
    """Sentence-level bootstrap: resample sentences with replacement within
    each language independently, recompute JSD each time. Returns point
    estimate + percentile CI."""
    sents_a = record_a.per_sentence_selected[layer_idx]
    sents_b = record_b.per_sentence_selected[layer_idx]

    point_a = np.concatenate(sents_a, axis=0)
    point_b = np.concatenate(sents_b, axis=0)
    point_estimate = jsd(
        expert_distribution(point_a, n_experts),
        expert_distribution(point_b, n_experts),
    )

    boot = np.empty(n_resamples)
    for k in range(n_resamples):
        idx_a = rng.integers(0, len(sents_a), size=len(sents_a))
        idx_b = rng.integers(0, len(sents_b), size=len(sents_b))
        resampled_a = np.concatenate([sents_a[i] for i in idx_a], axis=0)
        resampled_b = np.concatenate([sents_b[i] for i in idx_b], axis=0)
        boot[k] = jsd(
            expert_distribution(resampled_a, n_experts),
            expert_distribution(resampled_b, n_experts),
        )

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return {
        "point_estimate": point_estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "boot_std": float(boot.std()),
    }
