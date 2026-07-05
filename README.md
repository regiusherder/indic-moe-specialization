# Indic MoE Expert Specialization

**Do the experts in open Mixture-of-Experts (MoE) language models spontaneously
specialize by Indic language family or script — and is that specialization
causal or incidental?**

This repository analyzes the router behavior of three open MoE models on
parallel Indic text, with no training involved. It tests, on 11 Indian
languages across two families and many scripts, whether the routing behavior
[Mixtral](https://arxiv.org/abs/2401.04088) reported holds when the languages
are Indic — a setting no prior expert-specialization study covered.

**What Mixtral actually found** (Jiang et al. 2024, §5 "Routing analysis"): the
router shows *no* topic/domain specialization — "*we do not observe obvious
patterns in the assignment of experts based on the topic*" (expert distribution
is near-identical across ArXiv, PubMed, and PhilPapers text, their Figure 7) —
but *does* show "*structured syntactic behavior*", routing by token-level and
positional features (e.g. `self` in Python and `Question` in English go to the
same expert; consecutive tokens and indentation cluster; their Figure 8 /
Table 5). Their evidence is entirely English + code. **Whether that
topic-invariant, syntax-driven picture survives when the "domain" is instead a
different *language* or *script* is exactly what no one had checked — and what
this study measures.** Note that in our setting "does routing separate
languages/families?" is a sharper question than Mixtral's topic test: language
and script are much stronger surface/structural signals than document topic, so
finding language-family structure does not by itself contradict Mixtral — the
informative results are the *script-vs-language* control (Hindi–Urdu) and the
per-architecture differences, below.

<p align="center">
  <img src="results/figures/dendrogram_olmoe.png" width="70%"><br>
  <em>Unsupervised clustering of languages by routing similarity (OLMoE).
  English splits off first; the router then recovers the Dravidian vs
  Indo-Aryan family split on its own — and places Hindi and Urdu as sisters
  despite their completely different scripts.</em>
</p>

## Key findings

Measured across **OLMoE-1B-7B** (64 experts, no shared, English-heavy training),
**Qwen1.5-MoE-A2.7B** (60 routed + 1 shared, multilingual), and
**deepseek-moe-16b-base** (64 routed + 2 shared, multilingual):

1. **Routing is family-structured in OLMoE and Qwen; weaker/unclear for
   DeepSeek at the group level.** Within-family routing divergence (JSD) is
   lower than cross-family in every model (ratios 0.56–0.84), and per-pair,
   100% of language pairs are significant under a sentence-level permutation
   test. But that per-pair test only shows *some* pairs differ — it doesn't
   test the family-structure claim directly. We added that test: shuffle the
   Indo-Aryan/Dravidian labels among the 10 Indic languages and ask whether
   the *true* labeling explains the within/cross JSD gap better than a random
   relabeling would. Result: **significant for OLMoE (p=0.008) and Qwen
   (p=0.035), not significant for DeepSeek (p=0.070)** at the group level,
   even though DeepSeek's pairwise ratio (0.840) looks similar to Qwen's
   (0.835) on paper. The Dravidian/Indo-Aryan split emerges with zero
   supervision, most convincingly in OLMoE and Qwen.

2. **The Hindi–Urdu control splits by architecture — but check the metric.**
   Hindi and Urdu are nearly the same spoken language written in different
   scripts (Devanagari vs Perso-Arabic), so they cleanly separate *language
   identity* from *script*. The English-heavy **OLMoE** routes them together
   (ratio 0.56, language identity wins) and the multilingual **Qwen** routes
   them apart (ratio 1.58, script wins) — both **robust**: the same
   conclusion holds whether you use hard top-k selection counts (primary
   metric) or the full soft routing distribution (secondary metric, rho=0.97
   and 0.89 rank-correlated with hard, respectively). **DeepSeek is the one
   case that isn't robust to metric choice**: ratio 0.93 (roughly tied) under
   hard routing but 1.14 (mild script preference) under soft routing — the
   two metrics disagree on which side of 1.0 it lands, so DeepSeek's
   Hindi-Urdu result should be read as genuinely ambiguous, not just
   "intermediate." Reading OLMoE/Qwen through Mixtral's lens — where routing
   tracked token-level/structural features (of which script is one) rather
   than higher-level meaning — Qwen's script-driven routing is consistent
   with it and OLMoE (routing by language identity *across* scripts) runs
   contrary to it: the pattern is architecture-dependent, not universal, on
   Indic text.

3. **Specialization broadly increases with layer depth**, with a distinct
   signature per architecture. In OLMoE the Dravidian/Indo-Aryan ordering is
   consistent at every layer (Dravidian diverges more from English throughout)
   and overall divergence rises toward the final layers; Qwen is noisier with a
   sharp late-layer jump; DeepSeek shows the reversed family ordering.

4. **Ablation gives causal backing — but read this one carefully, we corrected
   it mid-audit.** Two tests, and they disagree in an informative way:
   - **vs. random experts, same-family baseline** (ablate family F's
     preferred experts; is F hurt more than F's languages are hurt by
     ablating *random* experts?): **specific in 8 of 9 cases** — every family
     in every model except DeepSeek's Dravidian group. *Caveat:* English
     (Indo-European, a single language) also passes this test in all 3
     models, which weakens how discriminating the test is — with only one
     member, "ablate English's own top experts" is close to "ablate whatever
     this specific language happens to prefer," so passing isn't strong
     evidence of *family-level* organization the way it is for the 6-language
     Indo-Aryan or 4-language Dravidian groups.
   - **vs. the other Indic family** (does ablating family F hurt F more than
     it hurts the *other* Indic family, under the same ablation?) — this is
     the cleaner test, immune to the baseline-choice problem above, and it is
     positive in **all 6 Indic cases**, including DeepSeek's Dravidian group.

   Read together: the *direction* is consistently family-specific (Test 2,
   the more trustworthy one, is clean 6/6), but the *strength* varies a lot
   by model and family (OLMoE's Dravidian differential is a small +0.03;
   Qwen's is a much larger +0.35) — this is evidence of causal, family-linked
   experts, not proof that every family is equally cleanly separated in every
   architecture.

5. **A confound we designed against isn't fully closed for OLMoE.**
   Token-capped sampling (below) was meant to equalize routing-decision count
   per language regardless of tokenizer fertility. It does that. But it
   doesn't necessarily break every relationship between fertility and
   routing: a direct check finds tokenizer fertility (tokens/char) still
   correlates with a language's JSD-vs-English in **OLMoE** (Spearman
   rho=0.72, p=0.019) — higher-fertility languages (mostly Dravidian, which
   fragment more) tend to diverge from English more, and family and
   fertility are correlated with each other in this language set, so this
   isn't fully separable from the family-structure finding. Qwen (rho=0.31,
   p=0.39) and DeepSeek (rho=−0.58, p=0.08) don't show a significant
   correlation. This doesn't overturn OLMoE's family-structure result (the
   group-level permutation test above still controls for exactly this kind
   of confound and remains significant), but it means fertility is not fully
   ruled out as a *contributing* factor for OLMoE specifically, and is worth
   controlling for more directly in any follow-up (e.g. deliberately pairing
   high- and low-fertility languages within each family).

### The numbers

**Family-structured routing** (mean JSD across layers; lower within-family than
cross-family means the router separates the two Indic families):

| Model | within-family JSD | cross-family JSD | ratio | median effect size |
|---|---|---|---|---|
| OLMoE | 0.0217 | 0.0386 | **0.56** | 90 SD |
| Qwen1.5-MoE | 0.0451 | 0.0540 | 0.84 | 113 SD |
| deepseek-moe-16b | 0.0429 | 0.0511 | 0.84 | 73 SD |

Effect sizes are standard deviations above the **sentence-level** permutation
null (the honest unit; token-level nulls overstate significance). All 55
language pairs are significant at p<0.05 in every model.

**The Hindi–Urdu control** (same spoken language, different script — isolates
language identity from orthography):

| Model | Hindi–Urdu JSD | Hindi vs other Indo-Aryan (mean) | ratio | verdict |
|---|---|---|---|---|
| OLMoE | 0.0148 | 0.0262 | **0.56** | language identity > script |
| Qwen1.5-MoE | 0.0484 | 0.0306 | 1.58 | script dominates |
| deepseek-moe-16b | 0.0367 | 0.0394 | 0.93 | roughly tied |

A ratio below 1 means Urdu routes *closer* to Hindi than Hindi's own family
relatives do — i.e. routing follows language, not script. The English-heavy
OLMoE shows this most strongly; the multilingual Qwen shows the opposite.

Full numbers are in [`results/figures/findings_summary.txt`](results/figures/findings_summary.txt);
all figures are in [`results/figures/`](results/figures/).

## Method in brief

- **Data.** FLORES-200 devtest (parallel across all languages), sampled to an
  **equal token budget per language** (not equal sentence count). Indic scripts
  fragment into many more subword tokens per unit of meaning than Latin script,
  so equal-sentence sampling would hand high-fertility languages more routing
  decisions and inflate their apparent divergence. Equal tokens gives every
  language equally-precise routing distributions and closes that confound.

- **Tokenization integrity check.** Before trusting any routing result, we
  verified every model's tokenizer actually encodes each of the 11 languages
  into real subwords, checking for three failure modes on the exact text each
  model was fed: `<unk>` tokens, leaked control tokens (e.g. `<|endoftext|>`
  appearing mid-text), and Unicode replacement characters (a sign of failed
  byte-fallback). Result: **33/33 language×model combinations clean** — the
  only flag anywhere is 3 replacement characters (0.015% of tokens) in
  DeepSeek's *English* sample, traced to em-dash/en-dash punctuation in the
  source FLORES-200 text, with zero occurrences in any of the 10 Indic
  languages the study measures. Reproducible via
  `scripts/verify_tokenization.py`; full log in
  [`results/figures/tokenization_audit.txt`](results/figures/tokenization_audit.txt).

- **Routing extraction.** PyTorch forward hooks on each layer's router capture,
  per token, the full softmax over **routed** experts and the top-k selection.
  Shared experts (Qwen, DeepSeek) fire on every token and carry no
  specialization signal, so they are excluded. Routers are kept in full
  precision even under 4-bit quantization, since the router logits are the
  measured quantity.

- **Statistics.** Pairwise Jensen-Shannon divergence between languages' expert-
  usage distributions, computed **per layer**. Significance via a permutation
  test that shuffles **sentence** labels (the independent unit) rather than
  tokens (which are within-sentence correlated and yield an anticonservative
  null). Sentence-level bootstrap confidence intervals throughout.

- **Causal test.** For each language family, identify its most
  disproportionately-used experts, ablate them, and measure the loss increase —
  compared against a baseline of ablating the same number of *random* experts,
  restricted to that same family's languages (an earlier pooled-baseline
  version of this test was corrected mid-project — see `WALKTHROUGH.md`-style
  notes in the git history / `scripts/analyze_results.py`'s docstring).

- **Robustness checks** (`scripts/robustness_checks.py`, no GPU): (1) does the
  soft-routing metric agree with the primary hard-routing metric on the
  headline numbers? (2) does tokenizer fertility still correlate with
  JSD-vs-English after token-capped sampling — i.e. is the fertility confound
  actually closed? (3) is the within/cross-family JSD gap itself significant
  under a label-shuffling permutation test, not just individual language
  pairs? Results are reported honestly including where they don't fully
  support the headline claims (DeepSeek's Hindi-Urdu result flips sign
  between metrics; OLMoE shows a residual fertility correlation) — see
  finding 5 above and
  [`results/figures/robustness_checks.txt`](results/figures/robustness_checks.txt).

Every architecture-specific detail lives in `src/adapters/`; the analysis code
(`src/routing.py`, `src/ablation.py`) is model-agnostic. Adding a fourth model
is one adapter plus three config lines.

## Repository layout

```
config.yaml                  Single source of truth: models, languages, seeds, all parameters
run_all.sh                   One-command entrypoint for a GPU pod (checks -> prefetch -> run)
requirements.txt / Dockerfile
src/
  adapters/                  The only architecture-specific code
    base.py                  Interface every adapter implements
    olmoe.py qwen_moe.py deepseek_moe.py
  data.py                    FLORES-200 download + token-capped sampling
  routing.py                 JSD, per-layer permutation tests, bootstrap CIs
  ablation.py                Targeted + random-control expert ablation
  pipeline.py                End-to-end orchestration with per-stage checkpointing
  manifest.py                Run provenance (git commit, config hash, GPU, seeds)
scripts/
  run_model.py               Run one model
  run_all_models.py          Run all three, continue past a per-model failure
  prefetch_model.py          Robust model-weight download (parallel, resumable)
  analyze_results.py         results/ -> figures + findings summary (no GPU needed)
  explore_families.py        Per-language (not just per-family-mean) deep dive:
                             pairwise breakdowns, family outliers, per-language
                             ablation, bootstrap CI widths (no GPU needed)
  robustness_checks.py       Soft-vs-hard metric agreement, fertility-confound
                             check, group-level family permutation test (no GPU)
  verify_tokenization.py     Confirms no language collapses to <unk>/control
                             tokens/replacement chars (no GPU needed)
tests/
  test_routing.py            Unit tests for the statistical core
results/                     Published outputs (figures + per-layer JSD, permutation,
                             bootstrap, ablation, and run manifests). Raw per-token
                             routing pickles are regenerable and not committed.
```

## Reproducing

Runs on a single 24–48 GB GPU; all three models load in 4-bit sequentially.

```bash
git clone https://github.com/regiusherder/indic-moe-specialization.git
cd indic-moe-specialization
bash run_all.sh                       # checks env, prefetches weights, runs all 3 models
python scripts/analyze_results.py --results ./results --out ./results/figures
```

`run_all.sh` is self-contained: it verifies GPU/disk, pins the model cache to
local disk, prefetches all model weights with a parallel resumable downloader,
then runs the pipeline offline. Every model's run writes a `manifest.json`
recording the git commit, config hash, model revision (SHA-pinned), GPU, and
FLORES checksum, so any result traces back to exactly what produced it.

Run just the analysis on the committed results without a GPU:

```bash
pip install -r requirements.txt
python scripts/analyze_results.py --results ./results --out /tmp/figs
python scripts/explore_families.py --results ./results --out /tmp/figs
python scripts/robustness_checks.py --results ./results --out /tmp/figs
python scripts/verify_tokenization.py --results ./results
```

Unit tests for the statistics (CPU-only):

```bash
python tests/test_routing.py
```

## Models analyzed

| Model | Architecture | Routed / shared experts | top-k | Indic exposure |
|---|---|---|---|---|
| `allenai/OLMoE-1B-7B-0125` | standard MoE | 64 / 0 | 8 | minimal (English-heavy) |
| `Qwen/Qwen1.5-MoE-A2.7B` | shared + routed | 60 / 1 | 4 | moderate (multilingual) |
| `deepseek-ai/deepseek-moe-16b-base` | fine-grained shared + routed | 64 / 2 | 6 | moderate (multilingual) |

Languages (FLORES-200): English (control) plus Hindi, Marathi, Bengali,
Gujarati, Punjabi, Urdu (Indo-Aryan) and Tamil, Telugu, Malayalam, Kannada
(Dravidian).

## References

- Jiang et al. (2024), **Mixtral of Experts**, arXiv:2401.04088. §5 "Routing
  analysis" is the source of the routing-behavior claim this study tests:
  no observed topic/domain specialization, but structured syntactic /
  token-level routing (Figures 7–8, Table 5). Quotes above are verbatim from
  that section.
- Muennighoff et al. (2024), **OLMoE: Open Mixture-of-Experts Language Models**,
  arXiv:2409.02060.
- Team, **Qwen1.5-MoE**, https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B.
- Dai et al. (2024), **DeepSeekMoE**, arXiv:2401.06066 (architecture of
  `deepseek-moe-16b-base`).
- NLLB Team (2022), **FLORES-200** evaluation benchmark, arXiv:2207.04672.

## License

Code released under the MIT License. Model weights and the FLORES-200 dataset
are governed by their respective upstream licenses.
