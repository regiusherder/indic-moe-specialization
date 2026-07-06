"""Stage-4 analysis core: per-layer hard+soft JSD matrices, sentence- and
token-level permutation tests, and sentence-level bootstrap CIs.

This module is deliberately TORCH-FREE and MODEL-FREE. Everything it needs is
the routing `records` (the `.pkl` files produced by extraction) plus a couple of
ints and a seed. That has two payoffs:

  1. PARALLELISM. The layers are statistically independent -- layer i's JSD /
     permutation / bootstrap never touch layer j's -- so the stage fans out
     across CPU cores, one task per layer. `analyze_one_layer` is a module-level
     function (not a closure) so ProcessPoolExecutor can pickle and ship it.

  2. PORTABILITY. No GPU / model / torch is involved, so this stage runs ANYWHERE
     the `.pkl` files land: on the rented GPU pod (wasteful -- paying for a GPU
     to do single-threaded CPU stats) or on a laptop for free. The standalone
     entry point is scripts/run_analysis.py, driven by analyse_all.sh.

MEMORY MODEL (why workers reload records from disk):
  With 17 languages x hundreds of sentences x numpy arrays, one full copy of
  `records` is large. Windows/`spawn` pickles every argument to every worker, so
  passing `records` directly would put one multi-GB copy PER WORKER in RAM and
  can exhaust a 16GB laptop (observed: BrokenProcessPool / "paging file too
  small"). Instead workers receive the ROUTING DIR PATH and each loads only what
  it needs. A per-process cache (`_WORKER_RECORDS`) means each worker loads the
  records at most once no matter how many layers it handles.

DETERMINISM: parallelizing changes NO numbers. Each layer gets its own RNG from
np.random.default_rng([base_seed, layer_idx]), so results are identical whether
layers run serially or across N workers and identical across machines. (This is
a change from the original single-shared-RNG stream, hence the schema bump.)
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import routing as routing_mod

# Bumped whenever the analysis output schema or the RNG scheme changes, so a
# stale 03_analysis/ from an older code version is detected and recomputed
# rather than silently mixed with new-scheme results.
ANALYSIS_SCHEMA_VERSION = 2

# per-worker record cache: {routing_dir_str: {lang: LanguageRoutingRecord}}.
# Populated lazily in the worker process; never crosses the process boundary.
_WORKER_RECORDS: dict = {}


def load_records(routing_dir) -> dict:
    """Load {lang_name: LanguageRoutingRecord} from a 02_routing_raw dir."""
    routing_dir = Path(routing_dir)
    records = {}
    for pkl in sorted(routing_dir.glob("*.pkl")):
        with open(pkl, "rb") as f:
            records[pkl.stem] = pickle.load(f)
    return records


def _records_for(routing_dir) -> dict:
    """Return records for routing_dir, loading (and caching) them in THIS process
    on first use. Keeps each worker to a single load regardless of layer count."""
    key = str(Path(routing_dir).resolve())
    cached = _WORKER_RECORDS.get(key)
    if cached is None:
        cached = load_records(routing_dir)
        _WORKER_RECORDS[key] = cached
    return cached


def analyze_one_layer(layer_idx, routing_dir, n_experts, n_perms, n_boot, base_seed):
    """All pairwise stats for a SINGLE layer. Pure function of its arguments
    (module-level -> picklable). Receives the ROUTING DIR (not the records) so
    workers don't each carry a multi-GB pickled copy; the records are loaded once
    per process and cached. Returns this layer's jsd matrices + per-pair rows.

    RNG is seeded from (base_seed, layer_idx), independent of worker count and
    execution order -> fully reproducible.
    """
    records = _records_for(routing_dir)
    rng = np.random.default_rng([base_seed, layer_idx])

    mh, lang_order = routing_mod.pairwise_jsd_matrix(records, layer_idx, n_experts, metric="hard")
    ms, _ = routing_mod.pairwise_jsd_matrix(records, layer_idx, n_experts, metric="soft")

    permtest_rows, bootstrap_rows = [], []
    for i, a in enumerate(lang_order):
        for j, b in enumerate(lang_order):
            if i >= j:
                continue
            sa = records[a].per_sentence_selected[layer_idx]
            sb = records[b].per_sentence_selected[layer_idx]
            pt_sent = routing_mod.permutation_test_sentences(sa, sb, n_experts, n_perms, rng)
            permtest_rows.append({"layer": layer_idx, "lang_a": a, "lang_b": b, "unit": "sentence", **pt_sent})
            pt_tok = routing_mod.permutation_test(
                np.concatenate(sa, 0), np.concatenate(sb, 0), n_experts, n_perms, rng)
            permtest_rows.append({"layer": layer_idx, "lang_a": a, "lang_b": b, "unit": "token", **pt_tok})
            bt = routing_mod.bootstrap_jsd_ci(records[a], records[b], layer_idx, n_experts, n_boot, rng)
            bootstrap_rows.append({"layer": layer_idx, "lang_a": a, "lang_b": b, **bt})

    return {
        "layer_idx": layer_idx,
        "jsd": {"matrix_hard": mh.tolist(), "matrix_soft": ms.tolist(), "lang_order": lang_order},
        "permtest_rows": permtest_rows,
        "bootstrap_rows": bootstrap_rows,
    }


def analysis_artifacts_valid(analysis_dir) -> bool:
    """True iff a complete, current-schema analysis already exists in this dir.
    Used both to skip re-analysis and (by run_all.sh's gate) to decide whether
    ablation may proceed."""
    analysis_dir = Path(analysis_dir)
    needed = [analysis_dir / "jsd_by_layer.json",
              analysis_dir / "permutation_tests.csv",
              analysis_dir / "bootstrap_cis.csv"]
    if not all(p.exists() for p in needed):
        return False
    try:
        existing = json.loads((analysis_dir / "jsd_by_layer.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if existing.get("_schema_version") != ANALYSIS_SCHEMA_VERSION:
        return False
    layers = {k: v for k, v in existing.items() if not k.startswith("_")}
    return bool(layers) and all("matrix_hard" in v for v in layers.values())


def _safe_worker_default() -> int:
    """A conservative default worker count for a laptop. Each worker holds a full
    copy of the records + scipy/numpy, so we do NOT spawn one-per-layer by
    default (that OOM'd a 16GB box). Cap at 4 and at cpu_count-1 (leave a core
    for the OS). The user can override with --workers."""
    import os
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu - 1))


def run_analysis(records_or_dir, n_experts, n_perms, n_boot, analysis_dir, base_seed,
                 n_workers=None, force=False, verbose=True):
    """Run the full stage-4 analysis, parallelized across layers.

    records_or_dir : either a routing dir (Path/str containing *.pkl) OR an
                     already-loaded {lang: record} dict. A dir is preferred for
                     the parallel path (workers reload from it, low memory); a
                     dict is accepted for the inline/pod path and for tests, and
                     is transparently backed by the dir if one is discoverable.
    n_experts      : adapter.num_routed_experts
    n_perms/n_boot : permutation count / bootstrap resamples
    analysis_dir   : where to write jsd_by_layer.json + the two CSVs
    base_seed      : per-layer RNGs are derived from (base_seed, layer_idx)
    n_workers      : max worker processes. None -> _safe_worker_default().
                     1 -> serial, in-process (no pool; no pickling; debuggable).
    force          : recompute even if valid artifacts already exist.

    Returns the sorted list of layer indices present. Idempotent.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs: we need BOTH the in-memory records (for the serial path and
    # to read layer keys) and, for the parallel path, a routing dir the workers
    # can reload from.
    if isinstance(records_or_dir, (str, Path)):
        routing_dir = Path(records_or_dir)
        records = load_records(routing_dir)
    else:
        records = records_or_dir
        routing_dir = None  # in-memory only; parallel path will fall back to serial-with-copies

    if not records:
        raise ValueError(f"No routing records for analysis at {records_or_dir!r}.")

    layers_present = sorted(next(iter(records.values())).per_sentence_selected.keys())

    if not force and analysis_artifacts_valid(analysis_dir):
        if verbose:
            print(f"    analysis: valid artifacts present ({len(layers_present)} layers) — skipping.")
        return layers_present

    n_pairs = len(layers_present) * len(records) * (len(records) - 1) // 2
    if n_workers is None:
        n_workers = _safe_worker_default()
    n_workers = max(1, int(n_workers))
    # Never spawn more workers than layers.
    n_workers = min(n_workers, len(layers_present))

    # Parallel path needs a routing dir for the low-memory worker reload. If we
    # were handed an in-memory dict with no dir, run serially rather than pickle
    # a multi-GB copy to every worker (the exact thing that OOM'd before).
    use_parallel = n_workers > 1 and routing_dir is not None

    if verbose:
        mode = f"{n_workers} workers" if use_parallel else "serial"
        note = "" if (routing_dir is not None or n_workers == 1) else \
               " (in-memory records, no routing dir -> serial to avoid per-worker copies)"
        print(f"    analysis: {len(layers_present)} layers x {len(records)*(len(records)-1)//2} "
              f"language pairs = {n_pairs} pairwise tests "
              f"({n_perms} perms + {n_boot} bootstrap each). CPU-bound, {mode}{note}.")
    t0 = time.time()

    results_by_layer = {}
    if not use_parallel:
        # cache the in-memory records for analyze_one_layer's _records_for()
        if routing_dir is not None:
            _WORKER_RECORDS[str(routing_dir.resolve())] = records
            arg_dir = routing_dir
        else:
            # stash under a sentinel key and pass it through
            arg_dir = "__inmemory__"
            _WORKER_RECORDS[str(Path(arg_dir).resolve())] = records
        for done, layer_idx in enumerate(layers_present, 1):
            res = analyze_one_layer(layer_idx, arg_dir, n_experts, n_perms, n_boot, base_seed)
            results_by_layer[layer_idx] = res
            if verbose:
                el = time.time() - t0
                eta = el / done * (len(layers_present) - done)
                print(f"    analysis: layer {layer_idx} done "
                      f"[{done}/{len(layers_present)}] ~{eta/60:.1f}min left")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(analyze_one_layer, layer_idx, str(routing_dir),
                          n_experts, n_perms, n_boot, base_seed): layer_idx
                for layer_idx in layers_present
            }
            done = 0
            for fut in as_completed(futures):
                res = fut.result()  # re-raises worker exceptions here (fail loud)
                results_by_layer[res["layer_idx"]] = res
                done += 1
                if verbose:
                    el = time.time() - t0
                    eta = el / done * (len(layers_present) - done)
                    print(f"    analysis: layer {res['layer_idx']} done "
                          f"[{done}/{len(layers_present)}] ~{eta/60:.1f}min left")

    # merge in a FIXED (sorted-layer) order so the CSVs/JSON are byte-identical
    # regardless of which worker finished first.
    jsd_by_layer = {"_schema_version": ANALYSIS_SCHEMA_VERSION}
    permtest_rows, bootstrap_rows = [], []
    for layer_idx in layers_present:
        res = results_by_layer[layer_idx]
        jsd_by_layer[str(layer_idx)] = res["jsd"]
        permtest_rows.extend(res["permtest_rows"])
        bootstrap_rows.extend(res["bootstrap_rows"])

    (analysis_dir / "jsd_by_layer.json").write_text(
        json.dumps(jsd_by_layer, indent=2), encoding="utf-8")
    pd.DataFrame(permtest_rows).to_csv(analysis_dir / "permutation_tests.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(analysis_dir / "bootstrap_cis.csv", index=False)

    if verbose:
        print(f"    analysis done in {(time.time()-t0)/60:.1f}min")
    return layers_present
