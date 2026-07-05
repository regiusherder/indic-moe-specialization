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


def _reference_single_loss(model, tokenizer, sentence: str, max_tokens: int) -> float:
    """The OLD one-sentence-at-a-time loss (model's built-in .loss on a single
    unpadded sentence). Used only to verify the batched path agrees -- if
    batching had a masking/shift bug, per-sentence numbers would silently be
    wrong, so we check rather than trust."""
    import torch
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    if inputs["input_ids"].shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        return float(model(**inputs, labels=inputs["input_ids"]).loss.item())


def verify_batched_loss(model, tokenizer, sentences: list[str], max_tokens: int, batch_size: int, n_check: int = 8):
    """Fail loud if the batched per-sentence loss disagrees with the unbatched
    reference. Compares ONLY sentences that survive BOTH paths' token-count
    filter (a word-count filter would misalign, since batched and reference
    both drop <2-TOKEN sentences, not <2-word ones -- verifying element by
    element by re-tokenizing each candidate so we never compare a dropped
    sentence against a NaN)."""
    import torch
    candidates = sentences[:n_check * 3]  # over-sample; some may be too short
    pairs = []  # (batched_loss, reference_loss) for sentences valid in both
    for s in candidates:
        # reference on this single sentence (NaN if <2 tokens)
        ref = _reference_single_loss(model, tokenizer, s, max_tokens)
        if not np.isfinite(ref):
            continue
        # batched path on this single sentence
        b = per_sentence_losses(model, tokenizer, [s], max_tokens, batch_size=batch_size)
        if len(b) != 1:
            continue  # batched also dropped it -> not comparable, skip
        pairs.append((float(b[0]), ref))
        if len(pairs) >= n_check:
            break
    if not pairs:
        print("    [warn] no sentences long enough to verify batched loss; skipping check")
        return
    batched = np.array([p[0] for p in pairs])
    ref = np.array([p[1] for p in pairs])
    # Tolerance: the batched and reference paths run the SAME model but at
    # different tensor shapes/precision, and under 4-bit quantization the matmul
    # kernels round slightly differently -- so expect ~0.1% numerical noise
    # (observed max ~0.003 on losses of 3-6, i.e. ~0.05% relative). A REAL
    # shift/mask/padding bug would be off by whole nats (>10% relative), so a
    # 2% relative + 0.05 absolute tolerance cleanly separates noise from bugs.
    RTOL, ATOL = 0.02, 0.05
    rel = np.max(np.abs(batched - ref) / np.maximum(np.abs(ref), 1e-6))
    if not np.allclose(batched, ref, rtol=RTOL, atol=ATOL):
        raise RuntimeError(
            f"Batched per-sentence loss (single) disagrees with unbatched reference beyond "
            f"quantization noise (max relative diff {rel*100:.2f}%, tol {RTOL*100:.0f}%). "
            f"batched={batched}, ref={ref} -- likely a real shift/mask bug, refusing to proceed."
        )

    # Critically: also verify the ACTUAL padded multi-sentence batch path (a
    # batch of 1 has no padding, so the single-sentence check above doesn't
    # exercise the masking that's the real risk). Run all valid candidates as
    # one real batch and confirm each still matches its own reference.
    valid_sents = [s for s in candidates if np.isfinite(_reference_single_loss(model, tokenizer, s, max_tokens))][:len(pairs)]
    if len(valid_sents) >= 2:
        batch_out = per_sentence_losses(model, tokenizer, valid_sents, max_tokens, batch_size=len(valid_sents))
        ref2 = np.array([_reference_single_loss(model, tokenizer, s, max_tokens) for s in valid_sents])
        if len(batch_out) != len(ref2):
            raise RuntimeError(
                f"padded-batch path returned {len(batch_out)} losses for {len(ref2)} "
                f"valid sentences -- the padding/skip filter is inconsistent, refusing to proceed.")
        brel = np.max(np.abs(batch_out - ref2) / np.maximum(np.abs(ref2), 1e-6))
        if not np.allclose(batch_out, ref2, rtol=RTOL, atol=ATOL):
            raise RuntimeError(
                f"PADDED multi-sentence batch disagrees with per-sentence reference beyond "
                f"quantization noise (max relative diff {brel*100:.2f}%, tol {RTOL*100:.0f}%). "
                f"A padding/masking bug would corrupt every ablation delta. "
                f"batch={batch_out}, ref={ref2}. Refusing to proceed."
            )
        print(f"    [ok] batched per-sentence loss matches reference within quantization noise: "
              f"single (max {rel*100:.2f}%) AND padded batch of {len(valid_sents)} (max {brel*100:.2f}%)")
    else:
        print(f"    [ok] batched loss matches reference (single only, max {rel*100:.2f}%; "
              f"too few valid sentences for a padded-batch check)")


def per_sentence_losses(model, tokenizer, sentences: list[str], max_tokens: int,
                        batch_size: int = 16) -> np.ndarray:
    """Mean per-token NLL for EACH sentence, as an array (critique #15 -- a
    distribution over sentences, not one blob number).

    Batched for speed: single-sentence forward passes barely use a modern GPU
    (~7% util observed), so this pads and runs BATCH_SIZE sentences per forward
    pass. Crucially it computes the loss PER SENTENCE by hand (masking pad and
    the shifted positions) rather than using the model's built-in .loss, which
    would average over all tokens in the whole batch and destroy the
    per-sentence granularity the error bars need.
    """
    import torch
    import torch.nn.functional as F

    # need a pad token to batch (many causal LMs lack one) and RIGHT padding:
    # with a causal mask, real tokens only attend to earlier real tokens, and
    # right-side pads are masked out of the loss below -- so right-padding does
    # not perturb any real token's logits. (Left-padding would shift positions.)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    def _one_batch(sub_idxs):
        """Compute per-sentence loss for one (possibly sub-)batch. Returns
        {orig_index: loss}. Raised OOMs are handled by the caller."""
        enc = tokenizer([sentences[i] for i in sub_idxs], return_tensors="pt",
                        truncation=True, max_length=max_tokens, padding=True).to(model.device)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if attn is None:
            attn = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn)
        # most HF causal LMs return an object with .logits; some
        # trust_remote_code models return a tuple/dict. Handle both, or fail
        # loud rather than AttributeError deep in the stack.
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            raise RuntimeError(
                f"model forward returned {type(outputs)} with no .logits -- cannot "
                f"compute per-sentence loss. Inspect this model's forward return type.")
        shift_logits = logits[:, :-1, :]          # predict t+1 from t
        shift_labels = input_ids[:, 1:]
        shift_mask = attn[:, 1:].bool()
        if shift_labels.shape[1] == 0:
            # whole batch is single-token sentences -> no predictions possible;
            # all get skipped by the <1-real-token filter below anyway.
            return {}
        tok_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="none",
        ).view(shift_labels.shape)
        tok_nll = tok_nll.masked_fill(~shift_mask, 0.0)
        counts = shift_mask.sum(dim=1).clamp(min=1)
        row_loss = (tok_nll.sum(dim=1) / counts).float().cpu().numpy()
        result = {}
        for local_i, orig_i in enumerate(sub_idxs):
            if int(shift_mask[local_i].sum()) >= 1:  # skip <2-real-token sentences
                result[orig_i] = float(row_loss[local_i])
        return result

    # sort by length so batches pad minimally, but remember original order
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    losses_by_orig = {}

    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        # OOM fallback: on CUDA OOM, clear cache and halve the batch, down to
        # batch-of-1, rather than crashing the whole (multi-hour) run. Results
        # are identical -- batch size only affects memory/speed, not the math.
        bs = len(idxs)
        while True:
            try:
                for s2 in range(0, len(idxs), bs):
                    losses_by_orig.update(_one_batch(idxs[s2:s2 + bs]))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise RuntimeError(
                        "CUDA OOM even at batch size 1 during loss computation -- "
                        "a single sentence exceeds available memory. Lower "
                        "ablation.perplexity_max_tokens in config.yaml.")
                bs = max(1, bs // 2)
                print(f"      [oom] halving loss batch to {bs} and retrying")

    tokenizer.padding_side = prev_side  # restore; don't leak a global side-effect
    # return in original sentence order, skipping any that were too short
    return np.array([losses_by_orig[i] for i in range(len(sentences)) if i in losses_by_orig])


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
    loss_batch_size: int = 16,
) -> list[dict]:
    """Runs, for each top_n in the sweep: baseline (once) + targeted ablation
    per family group + N random controls, measuring PER-SENTENCE loss for every
    condition. Returns long-format rows with mean/std/n over sentences.

    families_to_ablate: {family_name: [languages in it]} -- the groups whose
    preferred experts we ablate.
    """
    import time
    results = []
    layers_with_moe = sorted(next(iter(per_lang_layer_dist.values())).keys())
    n_groups = len(families_to_ablate)

    # ---- work accounting, so the log shows exactly how much is left ----
    # per top_n, per language: 1 baseline (done once, up front) + n_groups
    # targeted + n_random_controls random. Each is one per_sentence_losses call.
    langs = list(test_sentences.keys())
    calls_per_topn = len(langs) * (n_groups + n_random_controls)
    total_ablation_calls = calls_per_topn * len(top_n_sweep)
    print(f"    ablation plan: {len(top_n_sweep)} sweep sizes {top_n_sweep} x "
          f"{len(langs)} langs x ({n_groups} targeted + {n_random_controls} random) "
          f"= {total_ablation_calls} ablation forward-batches, plus {len(langs)} baselines.")

    # one-time correctness check: the batched per-sentence loss must match the
    # unbatched reference, or every ablation delta would be silently wrong.
    _some_sents = next(iter(test_sentences.values()))
    verify_batched_loss(adapter.model, adapter.tokenizer, _some_sents, perplexity_max_tokens, batch_size=loss_batch_size)

    # one-time EFFECTIVENESS check: confirm the ablation hooks actually change
    # the loss. If a hook silently didn't fire (wrong module path, etc.), every
    # delta would be ~0 and the whole causal experiment would be meaningless
    # noise -- worse than crashing, because it looks like a real null result.
    # Ablate a big chunk of experts on every MoE layer and require the loss to
    # move appreciably.
    _check_sents = [s for s in _some_sents if len(s.split()) >= 3][:8] or _some_sents[:8]
    _base_check = per_sentence_losses(adapter.model, adapter.tokenizer, _check_sents, perplexity_max_tokens, batch_size=loss_batch_size)
    # ablate each layer's MOST-USED experts (by overall usage), not arbitrary
    # low-numbered ones -- ablating rarely-used experts might not move the loss
    # even when the hooks work, which would make this check unreliable.
    _probe_lang = next(iter(per_lang_layer_dist))
    _n_probe = min(adapter.num_routed_experts, max(top_n_sweep) * 2)
    _big = {layer: np.argsort(-per_lang_layer_dist[_probe_lang][layer])[:_n_probe].tolist()
            for layer in layers_with_moe}
    with adapter.ablate_experts(_big):
        _abl_check = per_sentence_losses(adapter.model, adapter.tokenizer, _check_sents, perplexity_max_tokens, batch_size=loss_batch_size)
    _mean_shift = float(np.mean(np.abs(_abl_check - _base_check))) if len(_abl_check) == len(_base_check) else -1
    if _mean_shift < 1e-4:
        raise RuntimeError(
            f"Ablating {len(_big[layers_with_moe[0]])} experts/layer barely changed the loss "
            f"(mean |delta|={_mean_shift:.2e}). The ablation hooks are likely NOT firing / not "
            f"suppressing experts -- every ablation number would be a false null. Refusing to "
            f"proceed. Check the adapter's ablate_experts hook path against the live model.")
    print(f"    [ok] ablation is effective (ablating a large expert set shifts loss by "
          f"mean |delta|={_mean_shift:.3f} on a probe) -- hooks are firing.")

    # baseline per-sentence losses (no ablation), once per language
    print(f"    computing baselines for {len(langs)} languages ...")
    baseline = {}
    t0 = time.time()
    for lang, sents in tqdm(test_sentences.items(), desc="    baseline", unit="lang"):
        baseline[lang] = per_sentence_losses(adapter.model, adapter.tokenizer, sents, perplexity_max_tokens, batch_size=loss_batch_size)
    print(f"    baselines done in {time.time()-t0:.1f}s")

    def record(lang, condition, group, top_n, trial, deltas):
        meta = language_metadata[lang]
        results.append({
            "language": lang, "family": meta["family"], "condition": condition,
            "group": group, "top_n": top_n, "trial": trial,
            "delta_mean": float(np.mean(deltas)) if len(deltas) else float("nan"),
            "delta_std": float(np.std(deltas)) if len(deltas) else float("nan"),
            "n_sentences": int(len(deltas)),
        })

    def deltas_vs_base(lang, abl):
        base = baseline[lang]
        if len(abl) != len(base):
            # per_sentence_losses drops <2-token sentences deterministically by
            # sentence content, and ablation doesn't change tokenization, so this
            # must never happen -- if it does, alignment is broken, fail loud.
            raise RuntimeError(
                f"ablation loss length {len(abl)} != baseline length {len(base)} for "
                f"'{lang}' -- per-sentence alignment broken, refusing to record garbage deltas.")
        return abl - base

    ablation_start = time.time()
    done_calls = 0
    for ti, top_n in enumerate(top_n_sweep):
        targeted_by_group = {
            fam: top_experts_for_group(per_lang_layer_dist, glangs, adapter.num_layers,
                                       adapter.num_routed_experts, top_n, min_usage_floor_frac)
            for fam, glangs in families_to_ablate.items()
        }
        random_sets = [random_experts_for_group(adapter.num_routed_experts, top_n, rng, layers_with_moe)
                       for _ in range(n_random_controls)]

        bar = tqdm(test_sentences.items(), desc=f"    ablation top_n={top_n} [{ti+1}/{len(top_n_sweep)}]", unit="lang")
        for lang, sents in bar:
            for group_name, experts_by_layer in targeted_by_group.items():
                with adapter.ablate_experts(experts_by_layer):
                    abl = per_sentence_losses(adapter.model, adapter.tokenizer, sents, perplexity_max_tokens, batch_size=loss_batch_size)
                record(lang, "targeted", group_name, top_n, None, deltas_vs_base(lang, abl))
                done_calls += 1
            for trial, random_experts in enumerate(random_sets):
                with adapter.ablate_experts(random_experts):
                    abl = per_sentence_losses(adapter.model, adapter.tokenizer, sents, perplexity_max_tokens, batch_size=loss_batch_size)
                record(lang, "random_control", None, top_n, trial, deltas_vs_base(lang, abl))
                done_calls += 1
            # live ETA for the whole ablation stage of this cell
            elapsed = time.time() - ablation_start
            rate = done_calls / elapsed if elapsed > 0 else 0
            remaining = (total_ablation_calls - done_calls) / rate if rate > 0 else 0
            bar.set_postfix_str(f"{done_calls}/{total_ablation_calls} calls, ~{remaining/60:.1f}min left in ablation")

    print(f"    ablation stage done in {(time.time()-ablation_start)/60:.1f}min "
          f"({total_ablation_calls} forward-batches).")
    return results
