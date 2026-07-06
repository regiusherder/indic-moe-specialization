"""Gate helper for run_all.sh: decide whether a GPU phase can proceed, by
inspecting the results tree. No torch, fast, exit-code driven.

Subcommands (exit 0 = ready/true, 1 = not ready/false, 2 = usage/error):

  extracted-all   results/   -> 0 iff EVERY (model x condition x seed) cell has
                  a 02_routing_raw/ with the expected number of language .pkl
                  files (i.e. extraction is complete for the whole matrix).

  analysis-all    results/   -> 0 iff EVERY cell that has extraction ALSO has
                  valid, current-schema analysis artifacts. This is the gate the
                  pod uses before starting the ablation phase: don't ablate until
                  the laptop has produced analysis for everything extracted.

  status          results/   -> always exit 0; prints a per-cell table of
                  extracted / analyzed / ablated so the human can see where the
                  split pipeline is. Purely informational.

The matrix (models x conditions x seeds) and the language list come from
config.yaml, so the gate knows the FULL expected set, not just what happens to
exist on disk.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.analysis import analysis_artifacts_valid  # noqa: E402


def _matrix(config):
    models = list(config["models"].keys())
    conditions = list(config["data"]["sampling_conditions"].keys())
    seeds = config["seeds"]
    return models, conditions, seeds


def _cell_dir(results_root, cond, seed, model):
    return results_root / cond / f"seed{seed}" / model


def _n_langs(config):
    return len(config["languages"])


def _is_extracted(cell_dir, n_langs):
    routing = cell_dir / "02_routing_raw"
    if not routing.is_dir():
        return False
    return len(list(routing.glob("*.pkl"))) >= n_langs


def _is_ablated(cell_dir):
    return (cell_dir / "04_ablation" / "ablation_results.csv").exists()


def _iter_cells(results_root, config):
    models, conditions, seeds = _matrix(config)
    for model in models:
        for cond in conditions:
            for seed in seeds:
                yield model, cond, seed, _cell_dir(results_root, cond, seed, model)


def cmd_extracted_all(results_root, config):
    n_langs = _n_langs(config)
    missing = [f"{m}/{c}/seed{s}" for m, c, s, d in _iter_cells(results_root, config)
               if not _is_extracted(d, n_langs)]
    if missing:
        print(f"NOT all extracted — {len(missing)} cell(s) missing extraction:")
        for x in missing[:20]:
            print(f"  {x}")
        return 1
    print("All cells extracted.")
    return 0


def cmd_analysis_all(results_root, config):
    n_langs = _n_langs(config)
    extracted = [(m, c, s, d) for m, c, s, d in _iter_cells(results_root, config)
                 if _is_extracted(d, n_langs)]
    if not extracted:
        print("No extracted cells yet — nothing to gate on. (Run extraction first.)")
        return 1
    missing = [f"{m}/{c}/seed{s}" for m, c, s, d in extracted
               if not analysis_artifacts_valid(d / "03_analysis")]
    if missing:
        print(f"NOT ready to ablate — {len(missing)} extracted cell(s) lack valid analysis:")
        for x in missing[:20]:
            print(f"  {x}")
        print("Run analyse_all.sh on the laptop and sync results/ back before ablating.")
        return 1
    print(f"All {len(extracted)} extracted cells have valid analysis — ready to ablate.")
    return 0


def cmd_status(results_root, config):
    n_langs = _n_langs(config)
    print(f"{'cell':<40s} {'extracted':>9s} {'analyzed':>8s} {'ablated':>7s}")
    print("-" * 68)
    n_ext = n_an = n_abl = n_tot = 0
    for m, c, s, d in _iter_cells(results_root, config):
        n_tot += 1
        ext = _is_extracted(d, n_langs)
        an = analysis_artifacts_valid(d / "03_analysis")
        abl = _is_ablated(d)
        n_ext += ext; n_an += an; n_abl += abl
        print(f"{m+'/'+c+'/seed'+str(s):<40s} {('yes' if ext else '-'):>9s} "
              f"{('yes' if an else '-'):>8s} {('yes' if abl else '-'):>7s}")
    print("-" * 68)
    print(f"{'TOTAL '+str(n_tot)+' cells':<40s} {n_ext:>9d} {n_an:>8d} {n_abl:>7d}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["extracted-all", "analysis-all", "status"])
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    results_root = Path(args.results).resolve()
    if not results_root.exists():
        print(f"results dir {results_root} does not exist yet.")
        # 'extracted-all'/'analysis-all' -> not ready; 'status' -> fine
        return 0 if args.command == "status" else 1

    return {
        "extracted-all": cmd_extracted_all,
        "analysis-all": cmd_analysis_all,
        "status": cmd_status,
    }[args.command](results_root, config)


if __name__ == "__main__":
    sys.exit(main())
