"""Orchestrates one model end-to-end: load -> sample data -> extract routing
-> JSD/permutation/bootstrap -> ablation (targeted + random control) -> save.

Checkpointing contract: after EVERY language's routing extraction and after
the full ablation study, an intermediate artifact is written to disk before
moving on. If the process crashes or the pod is killed mid-run, re-running
`run_model.py --model X` picks up from the last completed checkpoint instead
of redoing finished work — this matters on rented spot/community-cloud GPUs
that can be preempted.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import routing as routing_mod
from .ablation import run_ablation_study, top_experts_for_group
from .data import build_token_capped_sample, download_flores, load_language_sentences
from .manifest import write_manifest


def _adapter_for(name: str):
    if name == "olmoe":
        from .adapters.olmoe import OLMoEAdapter
        return OLMoEAdapter()
    if name == "qwen_moe":
        from .adapters.qwen_moe import QwenMoEAdapter
        return QwenMoEAdapter()
    if name == "deepseek_moe":
        from .adapters.deepseek_moe import DeepSeekMoEAdapter
        return DeepSeekMoEAdapter()
    raise ValueError(f"Unknown adapter '{name}' — add it to src/adapters/ and register it here.")


def run_model_pipeline(model_key: str, config: dict, config_path: Path, results_root: Path):
    model_cfg = config["models"][model_key]
    out_dir = results_root / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"stage": "start"}

    print(f"\n{'='*70}\n{model_key}: starting from checkpoint stage '{checkpoint['stage']}'\n{'='*70}")

    if checkpoint.get("stage") == "complete" and (out_dir / "04_ablation" / "ablation_results.csv").exists():
        print(f"{model_key}: already complete (checkpoint says so and ablation_results.csv exists) — "
              f"skipping entirely, no model load needed. Delete {checkpoint_path} to force a re-run.")
        return out_dir

    # ---- Stage 0: data ----
    cache_dir = results_root / "_flores_cache"
    flores_dir, flores_sha256 = download_flores(cache_dir, config["data"]["flores_url"], config["data"]["flores_sha256"])
    write_manifest(out_dir, config, config_path, extra={"model_key": model_key, "flores_sha256": flores_sha256})

    # ---- Stage 1: load model ----
    adapter = _adapter_for(model_cfg["adapter"])
    print(f"Loading {model_cfg['hf_id']} ...")
    adapter.load(model_cfg["hf_id"], model_cfg.get("revision"), model_cfg["quantization"])
    print(f"Loaded. layers={adapter.num_layers} routed_experts={adapter.num_routed_experts} "
          f"top_k={adapter.top_k} shared_experts={adapter.n_shared_experts} "
          f"resolved_revision={getattr(adapter, 'resolved_revision', 'n/a')}")

    hooks = adapter.register_hooks()

    # ---- Stage 2: build token-capped samples per language ----
    samples_path = out_dir / "01_samples.json"
    if samples_path.exists():
        samples = json.loads(samples_path.read_text())
        # Guard against config drift on resume: if the language set changed
        # since samples were built, silently reusing them would produce a
        # run whose manifest config doesn't match its actual data.
        if set(samples.keys()) != set(config["languages"].keys()):
            raise RuntimeError(
                f"Resume mismatch: {samples_path} covers languages "
                f"{sorted(samples.keys())} but config specifies "
                f"{sorted(config['languages'].keys())}. The config changed after "
                f"this run started. Delete {out_dir} to start fresh, or restore the config."
            )
    else:
        samples = {}
        for lang_name, lang_meta in config["languages"].items():
            sentences = load_language_sentences(
                flores_dir, lang_meta["code"], config["data"]["split"], config["data"]["max_sentences_pool"]
            )
            text, n_sent, n_tok = build_token_capped_sample(
                sentences, adapter.tokenizer, config["data"]["max_tokens_per_language"]
            )
            samples[lang_name] = {"text": text, "n_sentences": n_sent, "n_tokens": n_tok, **lang_meta}
            print(f"  {lang_name:<12s} sampled: {n_sent} sentences, {n_tok} tokens (budget {config['data']['max_tokens_per_language']})")
        samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    print(f"Saved intermediate artifact: {samples_path}")

    # ---- Stage 3: routing extraction (per-sentence, checkpointed per language) ----
    extraction_dir = out_dir / "02_routing_raw"
    extraction_dir.mkdir(exist_ok=True)
    records: dict[str, routing_mod.LanguageRoutingRecord] = {}

    for lang_name, sample in samples.items():
        lang_ckpt = extraction_dir / f"{lang_name}.pkl"
        if lang_ckpt.exists():
            with open(lang_ckpt, "rb") as f:
                loaded = pickle.load(f)
            # Old-format checkpoints (pre soft-routing capture) lack prob sums;
            # loading one would crash stage 4 deep inside soft_distribution with
            # an unhelpful error. Detect and re-extract instead.
            if getattr(loaded, "per_sentence_prob_sums", None):
                records[lang_name] = loaded
                print(f"  [resume] {lang_name}: routing already extracted, loaded from checkpoint")
                continue
            print(f"  [resume] {lang_name}: checkpoint is old-format (no soft-routing data) — re-extracting")
            lang_ckpt.unlink()

        print(f"  Extracting routing for {lang_name} ...")
        per_sentence_selected = {}
        per_sentence_prob_sums = {}
        per_sentence_token_counts = []
        # Re-load original FLORES sentences (not the concatenated blob from
        # data.py) so routing is captured per-sentence — bootstrap resampling
        # in routing.py needs sentence-level granularity, and concatenation
        # only exists to build the token-capped sample for the fertility check.
        sentences = load_language_sentences(
            flores_dir, sample["code"], config["data"]["split"], config["data"]["max_sentences_pool"]
        )[: sample["n_sentences"]]

        n_tokens_actual = 0
        for sentence in sentences:
            adapter.clear_captures()
            inputs = adapter.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(adapter.model.device)
            n_sentence_tokens = inputs["input_ids"].shape[1]
            n_tokens_actual += n_sentence_tokens
            per_sentence_token_counts.append(n_sentence_tokens)
            with torch.no_grad():
                _ = adapter.model(**inputs)
            captures = adapter.get_captures()
            for layer_idx, cap in captures.items():
                per_sentence_selected.setdefault(layer_idx, []).append(cap.selected_experts.numpy())
                # sum (not mean) over tokens — see LanguageRoutingRecord docstring
                per_sentence_prob_sums.setdefault(layer_idx, []).append(
                    cap.routing_probs.sum(dim=0).numpy()
                )

        # n_tokens is the count actually forwarded per-sentence, which can
        # differ slightly from the sample-building estimate (special tokens,
        # concatenation boundaries) — record what really happened.
        record = routing_mod.LanguageRoutingRecord(
            language=lang_name, lang_code=sample["code"],
            n_sentences=sample["n_sentences"], n_tokens=n_tokens_actual,
            per_sentence_selected=per_sentence_selected,
            per_sentence_prob_sums=per_sentence_prob_sums,
            per_sentence_token_counts=per_sentence_token_counts,
        )
        records[lang_name] = record
        with open(lang_ckpt, "wb") as f:
            pickle.dump(record, f)
        print(f"    saved checkpoint: {lang_ckpt}")

    # Routing extraction is done — remove the capture hooks now. Leaving them
    # attached through the analysis/ablation stages would copy router tensors
    # to CPU on every one of the hundreds of ablation forward passes for
    # nothing, and stack capture hooks on the same modules the ablation
    # hooks target.
    for h in hooks:
        h.remove()
    hooks = []

    # ---- Stage 4: JSD matrices, permutation tests, bootstrap CIs — per layer ----
    analysis_dir = out_dir / "03_analysis"
    analysis_dir.mkdir(exist_ok=True)
    layers_present = sorted(next(iter(records.values())).per_sentence_selected.keys())

    rng = np.random.default_rng(config["seed"])
    analysis_artifacts = [analysis_dir / "jsd_by_layer.json",
                          analysis_dir / "permutation_tests.csv",
                          analysis_dir / "bootstrap_cis.csv"]
    run_stage_4 = True
    if all(p.exists() for p in analysis_artifacts):
        # only trust existing artifacts if they're the current schema —
        # old-format ones (single hard-only "matrix" key) must be recomputed
        existing = json.loads((analysis_dir / "jsd_by_layer.json").read_text())
        first_layer = next(iter(existing.values()), {})
        if "matrix_hard" in first_layer:
            print("  [resume] analysis artifacts already exist (current schema), skipping stage 4")
            run_stage_4 = False
        else:
            print("  [resume] analysis artifacts are old-schema — recomputing stage 4")

    jsd_by_layer = {}
    permtest_rows = []
    bootstrap_rows = []

    for layer_idx in layers_present if run_stage_4 else []:
        matrix_hard, lang_order = routing_mod.pairwise_jsd_matrix(
            records, layer_idx, adapter.num_routed_experts, metric="hard")
        matrix_soft, _ = routing_mod.pairwise_jsd_matrix(
            records, layer_idx, adapter.num_routed_experts, metric="soft")
        jsd_by_layer[layer_idx] = {
            "matrix_hard": matrix_hard.tolist(),
            "matrix_soft": matrix_soft.tolist(),
            "lang_order": lang_order,
        }

        n_perms = config["routing"]["permutation_test"]["n_permutations"]
        for i, a in enumerate(lang_order):
            for j, b in enumerate(lang_order):
                if i >= j:
                    continue
                sents_a = records[a].per_sentence_selected[layer_idx]
                sents_b = records[b].per_sentence_selected[layer_idx]

                # primary: sentence-level null
                pt_sent = routing_mod.permutation_test_sentences(
                    sents_a, sents_b, adapter.num_routed_experts, n_perms, rng)
                permtest_rows.append({"layer": layer_idx, "lang_a": a, "lang_b": b,
                                      "unit": "sentence", **pt_sent})

                # supplementary: token-level null (pilot-comparable, anticonservative)
                sel_a = np.concatenate(sents_a, axis=0)
                sel_b = np.concatenate(sents_b, axis=0)
                pt_tok = routing_mod.permutation_test(
                    sel_a, sel_b, adapter.num_routed_experts, n_perms, rng)
                permtest_rows.append({"layer": layer_idx, "lang_a": a, "lang_b": b,
                                      "unit": "token", **pt_tok})

                bt = routing_mod.bootstrap_jsd_ci(
                    records[a], records[b], layer_idx, adapter.num_routed_experts,
                    config["routing"]["bootstrap"]["n_resamples"], rng,
                )
                bootstrap_rows.append({"layer": layer_idx, "lang_a": a, "lang_b": b, **bt})

        print(f"  layer {layer_idx}: hard+soft JSD matrices + {len(lang_order)*(len(lang_order)-1)//2} "
              f"pairwise permutation tests (sentence+token) + bootstrap CIs done")

    if run_stage_4:
        (analysis_dir / "jsd_by_layer.json").write_text(json.dumps(jsd_by_layer, indent=2))
        pd.DataFrame(permtest_rows).to_csv(analysis_dir / "permutation_tests.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(analysis_dir / "bootstrap_cis.csv", index=False)
        print(f"Saved intermediate artifacts to {analysis_dir}")

    # ---- Stage 5: ablation study (targeted + random control) ----
    ablation_dir = out_dir / "04_ablation"
    ablation_dir.mkdir(exist_ok=True)
    ablation_csv = ablation_dir / "ablation_results.csv"

    if ablation_csv.exists():
        print(f"  [resume] ablation study already complete, skipping")
    else:
        per_lang_layer_dist = {
            lang: {
                layer: routing_mod.expert_distribution(
                    np.concatenate(rec.per_sentence_selected[layer], axis=0), adapter.num_routed_experts
                )
                for layer in layers_present
            }
            for lang, rec in records.items()
        }

        families = {meta["family"] for meta in config["languages"].values()}
        targeted_by_group = {}
        for family in families:
            group_langs = [l for l, m in config["languages"].items() if m["family"] == family and l in records]
            if not group_langs:
                print(f"  WARNING: family '{family}' has no languages with extracted routing records — "
                      f"skipping it in the ablation study. This means fewer than expected family "
                      f"groups will appear in ablation_results.csv; check for an earlier extraction failure.")
                continue
            targeted_by_group[family] = top_experts_for_group(
                per_lang_layer_dist, group_langs, adapter.num_layers,
                adapter.num_routed_experts, config["ablation"]["top_n_experts"],
            )

        test_texts = {lang: s["text"] for lang, s in samples.items()}
        # Dedicated generator: the shared stage-4 rng's state depends on whether
        # stage 4 ran or was resumed-past, which would make the random control
        # expert sets non-reproducible across resume paths.
        ablation_rng = np.random.default_rng(config["seed"] + 1)
        ablation_rows = run_ablation_study(
            adapter, test_texts, config["languages"], targeted_by_group,
            config["ablation"]["n_random_controls"], config["ablation"]["top_n_experts"],
            config["ablation"]["perplexity_max_tokens"], ablation_rng,
        )
        pd.DataFrame(ablation_rows).to_csv(ablation_csv, index=False)
        print(f"Saved ablation results: {ablation_csv}")

    checkpoint_path.write_text(json.dumps({
        "stage": "complete",
        # methods-relevant: which router return format the adapter saw, which
        # determines the exact ablation mechanism used (renormalize surviving
        # top-k weights for the tuple format vs. -inf logits forcing top-k
        # re-selection for the tensor format)
        "gate_output_format": getattr(adapter, "gate_output_format", "n/a"),
    }))
    print(f"\n{model_key}: pipeline complete. All artifacts under {out_dir}")
    return out_dir
