# Indic MoE Expert Specialization

Tests whether experts in open multilingual MoE LLMs spontaneously specialize
by Indic language family or script, and whether that specialization is
causal (via ablation) rather than merely correlational. Stress-tests
Mixtral's published claim that MoE routing is syntactic, not semantic.

Full background and design rationale: see the parent project's
[Conversation Log](../Conversation%20Log%20-%20Project%20Scoping.md) and the
weekend pilot (`olmoe-pilot.ipynb`) this codebase supersedes.

## What changed from the pilot

The Kaggle pilot notebook validated the signal exists (see pilot results:
55/55 language pairs significant at p<0.001, ~90 SD median effect size,
Hindi-Urdu JSD lower than Hindi-vs-other-Indo-Aryan). This repo turns that
into a rigorous, unattended, three-model study by fixing four gaps:

1. **Fertility confound closed** — sampling is capped by *token count* per
   language, not sentence count (fertility varies 9x across scripts in this
   language set; sentence-capping silently gave high-fertility languages more
   statistical power).
2. **Per-layer, sentence-level permutation testing** — the pilot tested
   significance at one "representative" layer with a token-level null.
   This runs the permutation test at every layer, and shuffles SENTENCE
   labels as the primary null (tokens within a sentence are correlated, so
   the token-level null is anticonservative — part of why the pilot's effect
   sizes looked like ~90 SDs). Token-level results are kept as a
   supplementary column for pilot comparability.
3. **Random-ablation control** — the pilot's causal ablation only compared
   "language-preferring experts ablated" to baseline. This adds N random-expert
   ablation trials per language-family group, so the result can show ablating
   *specific* experts hurts more than ablating *any* experts of the same count.
4. **Three-model, architecture-aware** — OLMoE (no shared experts), Qwen1.5-MoE
   (60 routed + 4 shared), DeepSeek-V2-Lite (64 routed + 2 shared) each get a
   dedicated adapter (`src/adapters/`) so shared experts are correctly excluded
   from specialization analysis, while the JSD/permutation/ablation logic
   itself stays architecture-agnostic.

## Before running for real

`src/adapters/qwen_moe.py` and `src/adapters/deepseek_v2lite.py` are written
against the *documented* architecture of these models (see the Conversation
Log) but have **not been executed against the live checkpoints**. Both files
say so explicitly at the top with what to check. Before the unattended run:

```bash
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B', device_map='auto')
print(m.model.layers[0].mlp)
"
```

...and confirm the printed module matches what `qwen_moe.py` assumes
(`.gate`, `.experts`, `.shared_expert`, `.shared_expert_gate`). Same for
DeepSeek-V2-Lite (`trust_remote_code=True` required). If it doesn't match,
fix the adapter before trusting any output — these two adapters are the
highest-risk part of this codebase precisely because they're least verified.
OLMoE's adapter (`olmoe.py`) mirrors the hook logic validated in the pilot
notebook and is lower risk.

## Running

**Locally (fast iteration, needs a GPU):**
```bash
pip install -r requirements.txt
python scripts/run_model.py --model olmoe
```

**Unattended on a rented GPU pod (RunPod/Lambda, single 24GB GPU — e.g. RTX 4090):**
```bash
git clone <this-repo> && cd indic-moe-specialization
tmux new -s indic-moe        # survive SSH disconnect
bash run_all.sh
# Ctrl+B, D to detach; tmux attach -t indic-moe to check back later
```

Runs all three models sequentially (one at a time fits a single 24GB GPU in
4-bit). If one model's pipeline crashes, the batch runner logs the failure
and continues to the next model — re-run `python scripts/run_model.py --model
<name>` afterward to resume just the failed one from its last checkpoint.

## Traceability — what's in `results/`

```
results/
  <model_name>/
    manifest.json              # git commit, config hash, timestamp, GPU, package versions
    01_samples.json            # exact text sampled per language + token/sentence counts
    02_routing_raw/
      <language>.pkl           # per-sentence raw expert-selection indices, every layer
    03_analysis/
      jsd_by_layer.json        # hard (top-k counts, primary) + soft (mean softmax, secondary) JSD matrices, every layer
      permutation_tests.csv    # every pair x layer x unit (sentence=primary, token=supplementary): observed JSD, p-value (add-one estimator), effect size
      bootstrap_cis.csv        # every language pair x every layer: point estimate + 95% CI (sentence-level resampling)
    04_ablation/
      ablation_results.csv     # long format: language x condition (baseline/targeted/random_control) x trial
    <model_name>.log           # full stdout/stderr of the run
    _checkpoint.json           # {"stage": "complete"} once done; enables safe re-run/resume
  _flores_cache/                # downloaded FLORES-200, shared across all three models
```

Every number in the eventual paper should be traceable to a row in one of
these CSVs, which is traceable via `manifest.json` to an exact config hash
and git commit. Nothing is aggregated-then-discarded; intermediate artifacts
(per-sentence routing, per-layer permutation tests) are kept, not just final
summary statistics.

## Tests

The statistical core (`src/routing.py`) has CPU-only unit tests — JSD
properties, permutation-test false-positive control under a true null,
detection of planted differences, add-one p-value floor, soft/hard metric
agreement, bootstrap sanity:

```bash
python tests/test_routing.py
```

Run these after any change to routing.py; a silent math bug there produces
plausible-looking wrong numbers in every downstream CSV.

## Config

`config.yaml` is the single source of truth — models, languages, seeds,
sample sizes, permutation/bootstrap counts, ablation parameters. Change an
experiment parameter there, not in code. The config file's hash is recorded
in every `manifest.json`.
