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
   (60 routed + 1 shared, verified live), deepseek-moe-16b-base (64 routed +
   2 shared, not yet verified live) each get a dedicated adapter
   (`src/adapters/`) so shared experts are correctly excluded from
   specialization analysis, while the JSD/permutation/ablation logic itself
   stays architecture-agnostic.

## A note on the DeepSeek model swap (2026-07-03)

The study originally targeted **DeepSeek-V2-Lite**, but its download failed
reproducibly on a rented RunPod pod: shard 3 of 4 failed at the identical
byte offset across three separate attempts (a "xet" fast-download backend
error, then — after disabling xet — a disk-quota error even with 119GB
free). Rather than keep fighting a flaky download path for one specific
model repo, the study swapped to **`deepseek-ai/deepseek-moe-16b-base`** —
the original DeepSeekMoE paper's model, and the same architecture family
V2-Lite represents (fine-grained routed + shared experts, dense first
layer, identical custom `MoEGate` module returning `(topk_idx, topk_weight,
aux_loss)`). This satisfies the same research role in the 3-way
architecture-family comparison; only `config.yaml`'s `hf_id` changed, the
adapter code (`src/adapters/deepseek_moe.py`) required no logic changes.

## Before running for real

`src/adapters/qwen_moe.py` was **verified live** on 2026-07-03 (RunPod RTX
4090, transformers 4.45.2) — `model.model.layers[0].mlp` matched every
assumption the adapter makes (`.gate` returns raw logits directly, `.experts`
is a 60-module ModuleList, `.shared_expert` + `.shared_expert_gate` present),
with one correction: Qwen1.5-MoE-A2.7B has exactly **1** shared expert, not 4
as an earlier design note assumed (now fixed in the adapter's `n_shared_experts`
and its docstring). This doesn't change any ablation/JSD logic — the code
never touched `shared_expert` either way — but it was wrong metadata that
would have ended up in a paper.

`src/adapters/deepseek_moe.py` is still written against the *documented*
architecture only (researched via web search against the model's HF config
and the DeepSeekMoE GitHub repo — see the file's docstring) and has **not
been executed against the live checkpoint**. Before running it for real:

```bash
export HF_HUB_DISABLE_XET=1   # see the xet note below — set this FIRST
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-moe-16b-base', trust_remote_code=True, device_map='auto')
print(m.model.layers[1].mlp)  # layer 0 is dense
"
```

...and confirm the printed module has `.gate` returning `(topk_idx,
topk_weight, aux_loss)` as the adapter assumes. If it doesn't match, fix the
adapter before trusting any output — this is now the only unverified,
highest-risk adapter in the codebase. OLMoE's adapter (`olmoe.py`) mirrors
the hook logic validated in the pilot notebook and is lower risk.

## Running

`requirements.txt` deliberately does NOT pin `torch` or `bitsandbytes` — a
rented pod's preinstalled torch build is already matched to its CUDA
driver/toolkit (e.g. RunPod's PyTorch template ships torch 2.8.0 built for
CUDA 12.8 on a driver 570 pod), and bitsandbytes' prebuilt binaries are
CUDA-version-specific. Pinning either to an older version risks a mismatch:
`bitsandbytes==0.44.1` was found on a RunPod pod (2026-07-03) to raise
`ModuleNotFoundError: No module named 'triton.ops'` deep inside its own
CUDA-kernel integration — its build predates that pod's CUDA 12.8/triton
combination. Unpinned, pip resolves a build that actually matches the
pod's installed toolkit. `pip install -r requirements.txt` will also emit a
dependency-conflict warning about `torchvision`/`torchaudio` wanting a
different torch version than requested — that's expected and harmless
(this pipeline uses neither package); it's informational, not an error,
and does not block the install.

Both `run_all.sh` and `scripts/run_model.py` set `HF_HUB_DISABLE_XET=1`
automatically — HF Hub's fast "xet" download backend failed reproducibly
mid-shard downloading DeepSeek-V2-Lite on a RunPod RTX 4090 (same offset,
two attempts); the standard HTTP downloader doesn't have this problem. You
don't need to set this yourself.

**Locally (fast iteration, needs a GPU):**
```bash
pip install -r requirements.txt
python scripts/run_model.py --model olmoe
```

**Unattended on a rented GPU pod (RunPod/Lambda, single 24GB GPU — e.g. RTX 4090) —
this is a single command, no manual steps in between:**
```bash
# clone onto the pod's OWN container disk, not a mounted network volume
# (RunPod sometimes defaults to a quota-limited shared mount like /workspace
# — check with `df -h .` after cloning; run_all.sh also checks this itself)
cd /root
git clone https://github.com/regiusherder/indic-moe-specialization.git
cd indic-moe-specialization

tmux new -s indic-moe        # survive SSH disconnect
bash run_all.sh
# Ctrl+B, D to detach; tmux attach -t indic-moe to check back later
```

`run_all.sh` does everything in order and needs nothing else from you:
1. Sanity-checks GPU, disk location, and free space (aborts early with a
   clear message if any of these look wrong, rather than failing hours in)
2. Installs dependencies
3. Pre-fetches all three models' weights via `scripts/prefetch_model.py` —
   hardened against a download-hang failure mode hit on 2026-07-03 (a stuck
   transfer that survived Ctrl+C): each attempt runs in a subprocess under a
   hard 45-minute timeout, gets killed and retried (up to 6 attempts) if it
   stalls, rather than hanging forever unattended
4. Runs the actual pipeline (olmoe → qwen_moe → deepseek_moe)

If one model's pipeline crashes after prefetch succeeds, the batch runner
logs the failure and continues to the next model — re-run `python
scripts/run_model.py --model <name>` afterward to resume just the failed
one from its last checkpoint. `scripts/prefetch_model.py --model <name>`
can also be run standalone to pre-cache one model without starting the
pipeline (useful for verifying an adapter live before committing to a full run).

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
