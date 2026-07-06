"""Orchestrates one model across the full run matrix: for a given model, loop
over (sampling condition) x (seed), and for each such CELL run:
  sample data -> extract routing -> JSD/permutation/bootstrap -> ablation.

The model is loaded ONCE (the expensive step) and reused across all cells; only
the data sampling and the stochastic analyses differ per cell. Results for each
cell go to results/<condition>/<seed>/<model>/ so the matrix never collides.

Checkpointing is per cell and per language: a crash re-runs only the unfinished
cell, and within it only the unfinished languages.

Sampling conditions (see src/data.py):
  token_capped -- equal tokens/language (equal precision, different content)
  aligned      -- same aligned FLORES sentences/language (same content,
                  different token counts)
Running both and requiring the finding to survive each answers the content-vs-
precision confound pair (critiques #5, #6).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
# torch is imported lazily inside _extract_routing (the only place it's used)
# so the pipeline can be imported and mock-tested without torch installed.

from . import routing as routing_mod
from .analysis import analysis_artifacts_valid, load_records, run_analysis
from .ablation import run_ablation_study
from .data import (build_aligned_sample, build_token_capped_sample,
                   download_flores, load_language_sentences)
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


def _routing_checkpoint_is_valid(loaded, n_routed_experts: int) -> bool:
    prob_sums = getattr(loaded, "per_sentence_prob_sums", None)
    if not prob_sums:
        return False
    for _, per_sentence in prob_sums.items():
        if not per_sentence:
            return False
        for vec in per_sentence:
            if getattr(vec, "shape", None) != (n_routed_experts,):
                return False
    return True


def _build_samples(config, adapter, flores_dir, condition_name, condition_cfg):
    """Returns {lang: {"code","family","script","pair_id","sentences":[...],
    "n_sentences","n_tokens_est"}} for one sampling condition."""
    split = config["data"]["split"]
    samples = {}
    for lang_name, lang_meta in config["languages"].items():
        if condition_name == "token_capped":
            pool = load_language_sentences(flores_dir, lang_meta["code"], split,
                                           condition_cfg["max_sentences_pool"])
            sents, n_tok = build_token_capped_sample(pool, adapter.tokenizer,
                                                     condition_cfg["max_tokens_per_language"])
        elif condition_name == "aligned":
            pool = load_language_sentences(flores_dir, lang_meta["code"], split,
                                           condition_cfg["n_aligned_sentences"])
            sents, _ = build_aligned_sample(pool, condition_cfg["n_aligned_sentences"])
            n_tok = None  # varies by fertility; measured for real during extraction
        else:
            raise ValueError(f"Unknown sampling condition '{condition_name}'")
        samples[lang_name] = {**lang_meta, "sentences": sents,
                              "n_sentences": len(sents), "n_tokens_est": n_tok}
        print(f"  {lang_name:<14s} [{condition_name}] {len(sents)} sentences"
              + (f", ~{n_tok} tokens" if n_tok else " (aligned; tokens vary)"))

    # For the aligned condition, every language MUST have the same number of
    # sentences -- that's the entire point (identical content, index-aligned).
    # If a FLORES file were short, they'd silently misalign; fail loud instead.
    if condition_name == "aligned":
        counts = {l: s["n_sentences"] for l, s in samples.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(
                f"aligned condition requires identical sentence counts across "
                f"languages, but got {counts}. A FLORES file may be shorter than "
                f"n_aligned_sentences; lower it or drop the short language.")
    return samples


def _extract_routing(adapter, samples, extraction_dir):
    """Per-language, per-sentence routing extraction with per-language checkpoints.

    Not batched on purpose: extraction is <1% of a cell's compute (the ablation
    stage dominates), and batching the router hooks would require splitting a
    (batch*seq, experts) capture back per-sentence while excluding pad rows --
    an error-prone reshape on the CORE measurement. Single-sentence extraction
    keeps the routing data trivially correct; the speed is spent where it
    matters (batched ablation loss). Verbose so progress is always visible.
    """
    import time
    import torch
    from tqdm import tqdm
    records = {}
    n_langs = len(samples)
    stage_t0 = time.time()
    for li, (lang_name, sample) in enumerate(samples.items(), 1):
        lang_ckpt = extraction_dir / f"{lang_name}.pkl"
        if lang_ckpt.exists():
            with open(lang_ckpt, "rb") as f:
                loaded = pickle.load(f)
            if _routing_checkpoint_is_valid(loaded, adapter.num_routed_experts):
                records[lang_name] = loaded
                print(f"    [{li}/{n_langs}] {lang_name}: reused checkpoint ({loaded.n_sentences} sentences)")
                continue
            lang_ckpt.unlink(missing_ok=True)

        per_sentence_selected, per_sentence_prob_sums, per_sentence_token_counts = {}, {}, []
        n_tokens_actual = 0
        bar = tqdm(sample["sentences"], desc=f"    [{li}/{n_langs}] extract {lang_name}",
                   unit="sent", leave=False)
        for sentence in bar:
            adapter.clear_captures()
            inputs = adapter.tokenizer(sentence, return_tensors="pt", truncation=True,
                                       max_length=512).to(adapter.model.device)
            n_tok = inputs["input_ids"].shape[1]
            n_tokens_actual += n_tok
            per_sentence_token_counts.append(n_tok)
            with torch.no_grad():
                _ = adapter.model(**inputs)
            captures = adapter.get_captures()
            if not captures:
                raise RuntimeError(
                    f"No routing captured for a sentence in '{lang_name}'. The forward "
                    f"hooks didn't fire -- the model's router module path likely changed. "
                    f"Refusing to save empty routing data.")
            for layer_idx, cap in captures.items():
                per_sentence_selected.setdefault(layer_idx, []).append(cap.selected_experts.numpy())
                per_sentence_prob_sums.setdefault(layer_idx, []).append(cap.routing_probs.sum(dim=0).numpy())

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
        elapsed = time.time() - stage_t0
        done_new = li  # rough; includes reused, fine for a coarse ETA
        eta = (elapsed / done_new) * (n_langs - done_new) if done_new else 0
        print(f"    [{li}/{n_langs}] {lang_name}: {sample['n_sentences']} sentences, "
              f"{n_tokens_actual} tokens extracted (~{eta/60:.1f}min left in this stage)")
    return records


def _run_analysis(routing_dir, adapter, config, analysis_dir, seed, n_workers=None):
    """Stage 4: per-layer hard+soft JSD, sentence+token permutation tests,
    bootstrap CIs. Delegates to the shared, layer-parallel, torch-free core in
    src/analysis.py so the identical computation can also run standalone on a
    laptop (scripts/run_analysis.py / analyse_all.sh) from the extracted .pkl
    files. Passes the ROUTING DIR (not in-memory records) so the parallel path's
    workers reload from disk instead of each holding a full pickled copy.

    Parallelizing across the independent layers is the single biggest speedup in
    the pipeline: this stage was ~half of every cell's wall-clock and was
    single-threaded.
    """
    return run_analysis(
        routing_dir,
        n_experts=adapter.num_routed_experts,
        n_perms=config["routing"]["permutation_test"]["n_permutations"],
        n_boot=config["routing"]["bootstrap"]["n_resamples"],
        analysis_dir=analysis_dir,
        base_seed=seed,
        n_workers=n_workers,
    )


def _run_ablation(records, samples, adapter, config, layers_present, ablation_dir, rng):
    ablation_csv = ablation_dir / "ablation_results.csv"
    if ablation_csv.exists():
        return
    per_lang_layer_dist = {
        lang: {layer: routing_mod.expert_distribution(
            np.concatenate(rec.per_sentence_selected[layer], 0), adapter.num_routed_experts)
            for layer in layers_present}
        for lang, rec in records.items()
    }
    families = {m["family"] for m in config["languages"].values()}
    families_to_ablate = {}
    for fam in families:
        langs = [l for l, m in config["languages"].items() if m["family"] == fam and l in records]
        if langs:
            families_to_ablate[fam] = langs
    test_sentences = {lang: s["sentences"] for lang, s in samples.items()}
    rows = run_ablation_study(
        adapter, test_sentences, config["languages"], per_lang_layer_dist, families_to_ablate,
        config["ablation"]["top_n_experts_sweep"], config["ablation"]["n_random_controls"],
        config["ablation"]["perplexity_max_tokens"], config["ablation"]["min_usage_floor_frac"], rng,
        loss_batch_size=config["ablation"].get("loss_batch_size", 16),
    )
    pd.DataFrame(rows).to_csv(ablation_csv, index=False)


# ---- checkpoint stages, in order ----
# The pipeline is split into GPU phases so the CPU-bound analysis can run
# elsewhere (a laptop) between them. A cell's _checkpoint.json "stage" records
# how far it has progressed:
#   "extracted" -> 02_routing_raw/*.pkl written; analysis + ablation still owed.
#   "complete"  -> ablation done too (04_ablation/ablation_results.csv present).
# Analysis (03_analysis/) is tracked by its own artifacts, NOT the checkpoint,
# because it may be produced on a different machine than the one that wrote the
# checkpoint. `phase` selects which GPU work this invocation performs:
#   "extract" -> only extraction (stop before analysis/ablation)
#   "ablate"  -> only ablation; REQUIRES extraction done AND analysis present
#   "full"    -> extract, analysis (inline), ablation — the all-on-one-box path
STAGE_EXTRACTED = "extracted"
STAGE_COMPLETE = "complete"


def _read_stage(ck_path: Path):
    if not ck_path.exists():
        return None
    try:
        return json.loads(ck_path.read_text()).get("stage")
    except (json.JSONDecodeError, OSError):
        return None


def run_model_pipeline(model_key: str, config: dict, config_path: Path,
                       results_root: Path, phase: str = "full"):
    """Run one model across the (condition x seed) matrix for the given GPU phase.

    phase="extract": extraction only (GPU). Writes .pkl, checkpoints "extracted".
    phase="ablate" : ablation only (GPU). Requires each cell to be "extracted"
                     AND to have valid analysis artifacts (produced on the laptop
                     via analyse_all.sh). Checkpoints "complete".
    phase="full"   : extract -> analysis (inline) -> ablation on one box (old
                     behavior; use when you have a single GPU machine and don't
                     want to offload analysis).
    """
    if phase not in ("extract", "ablate", "full"):
        raise ValueError(f"phase must be extract|ablate|full, got {phase!r}")
    model_cfg = config["models"][model_key]

    conditions = config["data"]["sampling_conditions"]
    seeds = config["seeds"]
    cells = [(c, s) for c in conditions for s in seeds]

    def cell_dir(cond, seed):
        return results_root / cond / f"seed{seed}" / model_key

    # ---- fast exits that avoid loading the model at all ----
    # Determine, per cell, what THIS phase still needs to do; if nothing, skip
    # the (expensive) model load entirely.
    def cell_needs_work(cond, seed):
        d = cell_dir(cond, seed)
        stage = _read_stage(d / "_checkpoint.json")
        if phase == "extract":
            return stage not in (STAGE_EXTRACTED, STAGE_COMPLETE)
        if phase == "full":
            return stage != STAGE_COMPLETE
        if phase == "ablate":
            if stage == STAGE_COMPLETE:
                return False
            # ablate needs to run here; whether it CAN is checked later (needs
            # extraction + analysis). Returning True means "load model and try".
            return True
        return True

    if not any(cell_needs_work(c, s) for c, s in cells):
        print(f"{model_key}: nothing to do for phase='{phase}' — all cells satisfied. No model load.")
        return

    # ---- data (once) ----
    cache_dir = results_root / "_flores_cache"
    flores_dir, flores_sha256 = download_flores(
        cache_dir, config["data"]["flores_url"], config["data"]["flores_sha256"])

    # ---- model (once) ----
    adapter = _adapter_for(model_cfg["adapter"])
    adapter.knockout = config["ablation"]["knockout"]
    print(f"Loading {model_cfg['hf_id']} (phase={phase}) ...")
    adapter.load(model_cfg["hf_id"], model_cfg.get("revision"), model_cfg["quantization"])
    print(f"Loaded. layers={adapter.num_layers} routed_experts={adapter.num_routed_experts} "
          f"top_k={adapter.top_k} shared_experts={adapter.n_shared_experts} "
          f"knockout={adapter.knockout} resolved_revision={getattr(adapter, 'resolved_revision', 'n/a')}")

    for cond_name, cond_cfg in conditions.items():
        samples = None  # built lazily, and only if some cell in this condition needs GPU work

        def ensure_samples(_samples):
            if _samples is not None:
                return _samples
            print("  building samples ...")
            s = _build_samples(config, adapter, flores_dir, cond_name, cond_cfg)
            cond_samples_path = results_root / cond_name / "01_samples.json"
            cond_samples_path.parent.mkdir(parents=True, exist_ok=True)
            cond_samples_path.write_text(json.dumps(
                {l: {k: v for k, v in si.items() if k != "sentences"} | {"n_sentences": si["n_sentences"]}
                 for l, si in s.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
            return s

        for seed in seeds:
            out_dir = cell_dir(cond_name, seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            ck = out_dir / "_checkpoint.json"
            stage = _read_stage(ck)
            tag = f"[{cond_name}/seed{seed}/{model_key}]"

            extraction_dir = out_dir / "02_routing_raw"
            analysis_dir = out_dir / "03_analysis"
            ablation_dir = out_dir / "04_ablation"

            if stage == STAGE_COMPLETE and (ablation_dir / "ablation_results.csv").exists():
                print(f"{tag} complete — skipping.")
                continue

            print(f"\n{'='*70}\n{model_key} | condition={cond_name} | seed={seed} | phase={phase}\n{'='*70}")

            # ============ EXTRACTION ============
            # Run if this cell isn't at least 'extracted', for extract/full phases.
            need_extract = stage not in (STAGE_EXTRACTED, STAGE_COMPLETE)
            if phase in ("extract", "full") and need_extract:
                write_manifest(out_dir, config, config_path, extra={
                    "model_key": model_key, "sampling_condition": cond_name, "seed": seed,
                    "flores_sha256": flores_sha256, "knockout": adapter.knockout,
                    "resolved_revision": getattr(adapter, "resolved_revision", None),
                    "phase_extract": True,
                })
                samples = ensure_samples(samples)
                hooks = adapter.register_hooks()
                extraction_dir.mkdir(exist_ok=True)
                _extract_routing(adapter, samples, extraction_dir)
                for h in hooks:
                    h.remove()
                ck.write_text(json.dumps({
                    "stage": STAGE_EXTRACTED,
                    "gate_output_format": getattr(adapter, "gate_output_format", "n/a"),
                    "knockout": adapter.knockout,
                }))
                stage = STAGE_EXTRACTED
                print(f"{tag} extraction done.")

            if phase == "extract":
                # GPU-only extract phase stops here; analysis+ablation happen later.
                continue

            # From here we need the records in memory (for full's inline analysis
            # and for ablation's expert targeting). They must exist on disk.
            if not (extraction_dir.exists() and any(extraction_dir.glob("*.pkl"))):
                if phase == "ablate":
                    print(f"{tag} SKIP: no extracted routing (.pkl) found — run the extract "
                          f"phase (run_all.sh) first.")
                    continue
                raise RuntimeError(f"{tag}: extraction produced no .pkl files.")

            # ============ ANALYSIS ============
            if phase == "full":
                samples = ensure_samples(samples)
                analysis_dir.mkdir(exist_ok=True)
                _run_analysis(
                    extraction_dir, adapter, config, analysis_dir, seed,
                    n_workers=config.get("analysis", {}).get("n_workers"))
            else:  # phase == "ablate": analysis MUST already be present (laptop-produced)
                if not analysis_artifacts_valid(analysis_dir):
                    print(f"{tag} SKIP ablation: analysis artifacts missing/stale in "
                          f"{analysis_dir}. Run analyse_all.sh on the laptop and sync results "
                          f"back before the ablate phase.")
                    continue
                samples = ensure_samples(samples)

            # ============ ABLATION ============
            # load records from disk (works for both full and ablate; the ablate
            # phase may run on a fresh box that never held them in memory).
            records = load_records(extraction_dir)
            layers_present = sorted(next(iter(records.values())).per_sentence_selected.keys())
            ablation_dir.mkdir(exist_ok=True)
            _run_ablation(records, samples, adapter, config, layers_present, ablation_dir,
                          np.random.default_rng(seed + 1))

            ck.write_text(json.dumps({
                "stage": STAGE_COMPLETE,
                "gate_output_format": getattr(adapter, "gate_output_format", "n/a"),
                "knockout": adapter.knockout,
            }))
            print(f"{tag} complete.")

    print(f"\n{model_key}: phase='{phase}' finished.")
