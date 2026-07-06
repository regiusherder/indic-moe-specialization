"""Run (or re-run) the stage-4 analysis for one or more result cells FROM THE
EXTRACTED .pkl FILES ONLY -- no model, no GPU, no torch required.

Why this exists
---------------
The analysis stage (per-layer JSD + permutation + bootstrap) is pure CPU stats
over the routing records. It was ~half of every cell's wall-clock AND single-
threaded -- i.e. paying for a rented GPU to do slow serial CPU work. This script
runs that stage on a laptop across cores, for free, from the
`02_routing_raw/*.pkl` files extraction already produced. Same code path as the
pipeline (src/analysis.run_analysis), so the numbers are identical.

analyse_all.sh is the thin wrapper you actually run; it calls this.

Workflow
--------
  # pod (instance A):  bash run_all.sh   # extraction only, writes 02_routing_raw/
  # pull results/ down to the laptop
  # laptop:            bash analyse_all.sh          # -> fills 03_analysis/
  # push results/ back up
  # pod  (instance B): bash run_all.sh   # sees analysis present -> runs ablation

  # direct use:
  python scripts/run_analysis.py results/                       # all cells needing analysis
  python scripts/run_analysis.py results/aligned/seed42/olmoe   # one specific cell
  python scripts/run_analysis.py results/ --force               # recompute even if present
  python scripts/run_analysis.py results/ --workers 6           # cap worker processes

A "cell" is any directory containing a 02_routing_raw/ folder with per-language
.pkl files. n_routed_experts is recovered from the records themselves (the soft
prob-sum vectors are length n_experts), so NO model / config lookup is needed --
the script is self-contained.
"""
import argparse
import json
import sys
from pathlib import Path

# make `import src...` work when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import (analysis_artifacts_valid, load_records,  # noqa: E402
                          run_analysis)


def _infer_n_experts(records):
    """n_routed_experts = width of the per-sentence soft prob-sum vectors.
    Cross-checked against the hard selection indices (max index must be <
    width) so a corrupt/mismatched pkl is caught rather than producing wrong
    JSD on misaligned expert spaces."""
    rec = next(iter(records.values()))
    any_layer = next(iter(rec.per_sentence_prob_sums))
    ne = int(rec.per_sentence_prob_sums[any_layer][0].shape[0])
    max_idx = -1
    for r in records.values():
        for layer_sents in r.per_sentence_selected.values():
            for sel in layer_sents:
                if sel.size:
                    m = int(sel.max())
                    if m > max_idx:
                        max_idx = m
    if max_idx >= ne:
        raise ValueError(
            f"selected expert index {max_idx} >= inferred n_experts {ne}: the "
            f".pkl files are inconsistent (prob-sum width vs selection indices). "
            f"Refusing to compute JSD on misaligned expert spaces.")
    return ne


def _find_cells(root):
    """A cell is any dir with a 02_routing_raw/ subdir holding >=1 .pkl."""
    root = Path(root)
    if (root / "02_routing_raw").is_dir():
        return [root]
    return sorted({p.parent.parent for p in root.glob("**/02_routing_raw/*.pkl")})


def _config_for_cell(cell_dir):
    """Recover n_perms / n_boot / seed from the cell's manifest.json so the
    standalone run matches exactly what the pipeline would use. Falls back to
    study defaults / the path-encoded seed if a manifest is somehow absent."""
    manifest_path = cell_dir / "manifest.json"
    n_perms, n_boot, seed = 1000, 200, 42
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            cfg = m.get("config_snapshot", {})
            n_perms = cfg.get("routing", {}).get("permutation_test", {}).get("n_permutations", n_perms)
            n_boot = cfg.get("routing", {}).get("bootstrap", {}).get("n_resamples", n_boot)
            seed = m.get("seed", seed)
        except (json.JSONDecodeError, OSError):
            pass
    else:
        for part in cell_dir.parts:  # .../seed<N>/<model>
            if part.startswith("seed") and part[4:].isdigit():
                seed = int(part[4:])
    return int(n_perms), int(n_boot), int(seed)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="results/ root, or a single cell dir (…/seed42/olmoe)")
    ap.add_argument("--workers", type=int, default=None,
                    help="max worker processes (default: safe laptop cap = min(4, cpu-1))")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if valid current-schema artifacts already exist")
    args = ap.parse_args()

    cells = _find_cells(args.path)
    if not cells:
        print(f"No cells (dirs with 02_routing_raw/*.pkl) found under {args.path}")
        sys.exit(1)

    todo, already = [], []
    for c in cells:
        (already if (not args.force and analysis_artifacts_valid(c / "03_analysis")) else todo).append(c)

    print(f"Found {len(cells)} cell(s): {len(todo)} to analyze, "
          f"{len(already)} already have valid analysis"
          + (" (use --force to redo)" if already and not args.force else "") + ".")
    for c in todo:
        print(f"  TODO   {c}")
    for c in already:
        print(f"  skip   {c}")
    if not todo:
        print("Nothing to do.")
        return

    failures = []
    for i, cell_dir in enumerate(todo, 1):
        routing_dir = cell_dir / "02_routing_raw"
        try:
            records = load_records(routing_dir)
            if len(records) < 2:
                print(f"[skip] {cell_dir}: only {len(records)} language(s) — need >=2 for pairwise JSD.")
                continue
            ne = _infer_n_experts(records)
            n_perms, n_boot, seed = _config_for_cell(cell_dir)
            print(f"\n=== [{i}/{len(todo)}] {cell_dir} ===")
            print(f"    {len(records)} languages, n_experts={ne}, "
                  f"n_perms={n_perms}, n_boot={n_boot}, seed={seed}")
            # pass the DIR (not the loaded dict) so workers reload low-memory
            run_analysis(
                routing_dir, n_experts=ne, n_perms=n_perms, n_boot=n_boot,
                analysis_dir=cell_dir / "03_analysis", base_seed=seed,
                n_workers=args.workers, force=args.force)
        except Exception as e:  # one bad cell shouldn't abort the rest
            import traceback
            print(f"*** {cell_dir} FAILED: {e}")
            traceback.print_exc()
            failures.append(cell_dir)

    print(f"\nAnalyzed {len(todo) - len(failures)}/{len(todo)} cells.")
    if failures:
        print("Failed cells (re-run to retry — completed cells will be skipped):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
