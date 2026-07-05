"""Unit tests for the statistical core (src/routing.py) — runs on CPU, no
model needed. These exist because a silent math bug here produces plausible-
looking wrong numbers in every downstream CSV.

Run: python -m pytest tests/ -q   (or python tests/test_routing.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing import (
    LanguageRoutingRecord,
    bootstrap_jsd_ci,
    expert_distribution,
    jsd,
    pairwise_jsd_matrix,
    permutation_test,
    permutation_test_sentences,
    soft_distribution,
)

N_EXPERTS = 8
TOP_K = 2
RNG = np.random.default_rng(0)


def _fake_sentences(n_sentences, tokens_per_sentence, expert_bias=None, rng=RNG):
    """Generate per-sentence (tokens, top_k) selections. expert_bias: array of
    per-expert selection probabilities (defaults to uniform)."""
    p = expert_bias if expert_bias is not None else np.ones(N_EXPERTS) / N_EXPERTS
    out = []
    for _ in range(n_sentences):
        sel = np.stack([
            rng.choice(N_EXPERTS, size=TOP_K, replace=False, p=p)
            for _ in range(tokens_per_sentence)
        ])
        out.append(sel)
    return out


def _record(name, sents):
    n_tokens = sum(s.shape[0] for s in sents)
    prob_sums = [np.bincount(s.flatten(), minlength=N_EXPERTS).astype(float) / TOP_K for s in sents]
    return LanguageRoutingRecord(
        language=name, lang_code=name, n_sentences=len(sents), n_tokens=n_tokens,
        per_sentence_selected={0: sents},
        per_sentence_prob_sums={0: prob_sums},
        per_sentence_token_counts=[s.shape[0] for s in sents],
    )


def test_jsd_identical_is_zero():
    p = np.array([0.5, 0.25, 0.25, 0, 0, 0, 0, 0])
    assert abs(jsd(p, p)) < 1e-12


def test_jsd_disjoint_is_one():
    p = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    q = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    assert abs(jsd(p, q) - 1.0) < 1e-9  # base-2 JSD of disjoint distributions = 1


def test_jsd_symmetric():
    p = np.array([0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0])
    q = np.array([0.1, 0.7, 0.1, 0.1, 0, 0, 0, 0])
    assert abs(jsd(p, q) - jsd(q, p)) < 1e-12


def test_expert_distribution_sums_to_one():
    sel = np.array([[0, 1], [2, 3], [0, 1]])
    d = expert_distribution(sel, N_EXPERTS)
    assert abs(d.sum() - 1.0) < 1e-12
    assert d[0] == d[1] == 2 / 6


def test_expert_distribution_rejects_out_of_range_index():
    # an index >= n_experts must fail loud, not silently make a longer vector
    bad = np.array([[0, 1], [2, N_EXPERTS]])  # N_EXPERTS is out of range (valid: 0..N_EXPERTS-1)
    try:
        expert_distribution(bad, N_EXPERTS)
        assert False, "expected out-of-range index to raise"
    except ValueError:
        pass
    # length is always exactly n_experts even when high indices are absent
    d = expert_distribution(np.array([[0, 1]]), N_EXPERTS)
    assert len(d) == N_EXPERTS


def test_jsd_rejects_mismatched_lengths():
    try:
        jsd(np.ones(4) / 4, np.ones(5) / 5)
        assert False, "expected mismatched-length JSD to raise"
    except ValueError:
        pass


def test_permutation_null_no_difference():
    """Same generating process for both 'languages' -> p should NOT be small."""
    rng = np.random.default_rng(1)
    sents_a = _fake_sentences(30, 20, rng=rng)
    sents_b = _fake_sentences(30, 20, rng=rng)
    res = permutation_test_sentences(sents_a, sents_b, N_EXPERTS, 500, rng)
    assert res["p_value"] > 0.01, f"false positive under null: p={res['p_value']}"


def test_permutation_detects_real_difference():
    rng = np.random.default_rng(2)
    bias_a = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    bias_b = np.array([0.03, 0.03, 0.04, 0.05, 0.05, 0.1, 0.3, 0.4])
    sents_a = _fake_sentences(30, 20, expert_bias=bias_a, rng=rng)
    sents_b = _fake_sentences(30, 20, expert_bias=bias_b, rng=rng)
    res = permutation_test_sentences(sents_a, sents_b, N_EXPERTS, 500, rng)
    assert res["p_value"] < 0.01, f"missed a large real difference: p={res['p_value']}"
    tok = permutation_test(
        np.concatenate(sents_a, axis=0), np.concatenate(sents_b, axis=0),
        N_EXPERTS, 500, rng)
    assert tok["p_value"] < 0.01


def test_pvalue_never_zero():
    rng = np.random.default_rng(3)
    bias_a = np.zeros(N_EXPERTS); bias_a[:2] = 0.5
    bias_b = np.zeros(N_EXPERTS); bias_b[-2:] = 0.5
    sents_a = _fake_sentences(20, 10, expert_bias=bias_a, rng=rng)
    sents_b = _fake_sentences(20, 10, expert_bias=bias_b, rng=rng)
    res = permutation_test_sentences(sents_a, sents_b, N_EXPERTS, 100, rng)
    assert res["p_value"] >= 1 / 101  # add-one estimator floor


def test_soft_distribution_sums_to_one_and_matches_hard_here():
    """prob_sums built from the same counts as selections -> soft == hard."""
    rng = np.random.default_rng(4)
    sents = _fake_sentences(10, 15, rng=rng)
    rec = _record("x", sents)
    soft = soft_distribution(rec, 0)
    hard = expert_distribution(np.concatenate(sents, axis=0), N_EXPERTS)
    assert abs(soft.sum() - 1.0) < 1e-9
    assert np.allclose(soft, hard, atol=1e-9)


def test_pairwise_matrix_symmetric_zero_diagonal():
    rng = np.random.default_rng(5)
    recs = {n: _record(n, _fake_sentences(10, 15, rng=rng)) for n in ["a", "b", "c"]}
    for metric in ["hard", "soft"]:
        m, order = pairwise_jsd_matrix(recs, 0, N_EXPERTS, metric=metric)
        assert np.allclose(m, m.T)
        assert np.allclose(np.diag(m), 0)
        assert order == ["a", "b", "c"]


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(6)
    ra = _record("a", _fake_sentences(30, 15, rng=rng))
    rb = _record("b", _fake_sentences(30, 15, rng=rng))
    res = bootstrap_jsd_ci(ra, rb, 0, N_EXPERTS, 100, rng)
    assert res["ci_low"] <= res["ci_high"]
    # point estimate should be near the bootstrap distribution (loose check)
    assert res["ci_low"] - 0.05 <= res["point_estimate"] <= res["ci_high"] + 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
