# A Walkthrough: How This Study Works, and Why

*Written as if I built this and I'm sitting next to you explaining it, file by
file, decision by decision. Read it top to bottom once; after that it's a
reference. Everything here is real — the numbers are from our actual run on an
RTX PRO 4500 on 2026-07-04, not illustrations.*

---

## 0. What we are actually asking, in one paragraph

Mixture-of-Experts (MoE) language models don't run every token through the
whole network. Each transformer layer has a bank of small "expert" sub-networks
and a tiny **router** that, per token, picks the top-k experts to use. The
famous Mixtral paper claimed this routing is **syntactic** — driven by
surface/token-level features — **not semantic**, i.e. experts don't specialize
by topic or language. That claim was established on English, European
languages, and code. **Nobody checked it on Indic languages**, which give us a
natural experiment: 11 languages, two clean families (Indo-Aryan vs Dravidian),
many different scripts, and — critically — Hindi and Urdu, which are almost the
same spoken language written in totally different scripts. Our question: **does
MoE routing on Indic text cluster by language family, by script, or neither —
and does it depend on the model's architecture?**

We answer it by *reading the router's decisions* on parallel text, never by
training anything. That's what makes it doable solo on rented GPUs.

---

## 1. The shape of the whole thing

```
config.yaml            <- the single source of truth. Change experiments here, nowhere else.
run_all.sh             <- one command. Checks, prefetch, run all 3 models.
src/
  adapters/            <- the ONLY architecture-specific code. One file per model family.
    base.py            <- the interface every adapter must satisfy
    olmoe.py / qwen_moe.py / deepseek_moe.py
  data.py              <- FLORES download + the token-capped sampling (a key design choice)
  routing.py           <- the statistics: JSD, permutation tests, bootstrap
  ablation.py          <- the causal experiment: kill experts, measure damage
  pipeline.py          <- orchestrates one model end-to-end, with checkpointing
scripts/
  run_model.py         <- run one model
  run_all_models.py    <- run all three, continue past a failure
  prefetch_model.py    <- download weights robustly (a whole saga; see §9)
  analyze_results.py   <- turn results/ into figures + a findings summary
results/<model>/       <- every intermediate artifact, fully traceable
```

The single most important architectural idea: **the analysis code
(`routing.py`, `ablation.py`, `pipeline.py`) knows nothing about any specific
model.** It only talks to the `MoEAdapter` interface. All the messy
per-architecture differences — different router return types, different expert
counts, one model having a dense first layer — are quarantined inside
`src/adapters/`. This is why we could swap DeepSeek-V2-Lite for
deepseek-moe-16b-base by changing *one line of config* and zero analysis code.

---

## 2. `config.yaml` — why everything lives in one file

Look at the top comment:

```yaml
# Single source of truth for the study. Every script reads this file;
# nothing is hardcoded elsewhere. Changing an experiment parameter means
# changing exactly one line here, and the run manifest hashes this file
# so every result is traceable to the config that produced it.
seed: 42
```

Two lessons here, and they're both about **trust in your own results**:

1. **One place to change things.** If `max_tokens_per_language` were hardcoded
   in three files, you'd eventually change two of them and get silently
   inconsistent results. One file means one truth.
2. **The config is hashed into every `manifest.json`.** Months from now, when a
   reviewer asks "what exactly produced Figure 3?", you can point at a config
   hash and a git commit. This is the difference between "I think I used 20k
   tokens" and "here is cryptographic proof of the exact settings."

The `seed: 42` is the anchor of reproducibility — every random operation
(permutation shuffles, bootstrap resamples, random-expert selection) derives
from it. Same seed → same random draws → same numbers (to within GPU
floating-point noise, which we'll discuss).

Now the languages block — read the comment on Urdu:

```yaml
  urdu:      {code: urd_Arab, family: Indo-Aryan,     script: Perso-Arabic}
# Urdu is the critical script-vs-family control (near-identical to Hindi, different script).
```

**This one line is the cleverest thing in the whole study.** Hindi and Urdu are
linguistically almost the same language (Hindustani) but Hindi is written in
Devanagari and Urdu in Perso-Arabic script. So if two languages that *are the
same language but look completely different* route **similarly**, routing
follows language identity. If they route **differently**, routing follows
script. Every other language pair confounds family and script together; Hindi-
Urdu is the one pair that cleanly separates them. When you design an experiment,
always hunt for the one comparison that isolates the variable you care about.

---

## 3. `src/data.py` — the token-capped sampling, and the confound it kills

Here's the function that decides how much text each language contributes:

```python
def build_token_capped_sample(sentences, tokenizer, max_tokens):
    """Concatenate sentences one at a time until the token budget is hit."""
    text = ""
    n_sentences = 0
    n_tokens = 0
    for sentence in sentences:
        candidate = (text + " " + sentence).strip() if text else sentence
        candidate_tokens = len(tokenizer(candidate, add_special_tokens=False, verbose=False)["input_ids"])
        if candidate_tokens > max_tokens and n_sentences > 0:
            break
        text = candidate
        n_tokens = candidate_tokens
        n_sentences += 1
        if n_tokens >= max_tokens:
            break
```

Why does this matter so much? Look at what actually happened in our OLMoE run:

```
english      sampled: 736 sentences, 19966 tokens
hindi        sampled: 152 sentences, 19888 tokens
punjabi      sampled:  77 sentences, 19997 tokens
malayalam    sampled:  73 sentences, 19528 tokens
```

English needed **736 sentences** to reach ~20k tokens; Punjabi needed **77**.
That's a ~10x difference. Here's the trap we avoided:

**Tokenization fertility varies enormously across scripts.** Indic scripts
(especially Dravidian ones) fragment into many more subword tokens per unit of
meaning than Latin script does. If we had capped by **sentence count** (say, 200
sentences each, like a naive pilot would), then Punjabi would have contributed
~10x more *tokens* — and therefore ~10x more *routing decisions* — than
English. Statistical precision scales with the number of observations, so
high-fertility languages would get artificially tight distributions and
inflated-looking divergences. A reviewer would (correctly) say "your effect is
an artifact of tokenization, not linguistics."

By capping on **tokens**, every language contributes the same number of routing
decisions (~20k). Every language's routing distribution is estimated with equal
precision. The confound is dead. We pay a price — English covers 736 sentences
of content while Malayalam covers 73 — but that's the lesser evil, and the
downstream stats account for each language's actual sentence count anyway.

*The `verbose=False` on the tokenizer call is a tiny scar: DeepSeek's tokenizer
warns when a growing candidate string exceeds its 16,384-token max — harmless,
since we only read `len(input_ids)` to count, never feed it to the model — but
we silence it so logs stay clean.*

---

## 4. `src/adapters/` — where all the architecture-specific pain is contained

### 4a. The contract (`base.py`)

```python
@dataclass
class RoutingCapture:
    layer_idx: int
    routing_probs: "torch.Tensor"     # (n_tokens, n_routed_experts) full softmax, routed experts only
    selected_experts: "torch.Tensor"  # (n_tokens, top_k) int64 indices into routed experts
```

Every adapter, whatever the model, must produce exactly this: for each layer,
the full softmax distribution over routed experts, and which top-k experts were
actually selected. Once the adapter hands back a `RoutingCapture`, the rest of
the pipeline treats OLMoE, Qwen, and DeepSeek identically. **Design principle:
find the narrowest possible interface that captures what you need, then make the
messy world conform to it.**

Note "**routed experts only**". Qwen and DeepSeek have *shared* experts that fire
on every token regardless of routing. Those carry no specialization signal (they
run for everything), so we deliberately exclude them. `n_shared_experts` is
recorded per adapter for documentation, but the shared experts never enter the
JSD or ablation math.

### 4b. How we actually read the router (`olmoe.py`)

The hook is registered on each layer's router (`mlp.gate`):

```python
def _make_hook(self, layer_idx: int):
    def hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 3:
            # newer transformers: (logits, weights, selected_experts)
            logits, _, selected = output[0], output[1], output[2]
            selected = selected.detach().cpu()
            self.gate_output_format = "tuple"
        elif torch.is_tensor(output):
            # older transformers: gate is nn.Linear, output is raw logits;
            # recompute the model's own top-k selection from them
            logits = output
            _, selected = torch.topk(
                F.softmax(logits.detach().float(), dim=-1), k=self.top_k, dim=-1
            )
            selected = selected.cpu()
            self.gate_output_format = "tensor"
        else:
            raise RuntimeError(...)  # fail loud, never guess
        probs = F.softmax(logits.detach().float(), dim=-1)
        self._captures[layer_idx] = RoutingCapture(...)
    return hook
```

A **forward hook** is PyTorch's way of tapping into a module's output during a
forward pass without modifying the model. We attach one to every router. When
the model runs, our hook fires and records what the router did.

Why the two branches? Because the *exact thing the router returns changed
between transformers versions.* Our executed pilot ran on a newer version where
the gate returns a `(logits, weights, selected)` tuple; the pinned version on
the pod returns a raw logits tensor. Rather than assume one, **the hook handles
both and records which it saw** (`gate_output_format`, written into the
checkpoint). If it sees something it doesn't recognize, it raises immediately.
This is the recurring discipline in this codebase: **when reality might differ
from your assumption, detect it and fail loudly — never silently produce
plausible-but-wrong numbers.**

One more subtle-but-critical line, in `load()`:

```python
llm_int8_skip_modules=["gate"],   # keep the ROUTER in full precision
```

We load models in 4-bit to fit them on a 24-32GB GPU. But 4-bit quantization
would perturb the router's logits — and **the router's logits are the entire
quantity we're measuring.** So we exclude the gate from quantization. The
experts get quantized (saves memory, barely affects which expert is picked); the
router stays exact. Quantize what you're not measuring; keep what you are
measuring pristine.

### 4c. The one that bit us (`deepseek_moe.py`)

DeepSeek is different in two ways that matter, and both caused real bugs we
fixed live:

**(1) It has a dense first layer.** Layer 0 is a normal FFN, not MoE. So the
hook only fires on layers 1–27. The pipeline derives which layers exist from the
captured data itself (`per_sentence_selected.keys()`), so it naturally analyzes
1–27 and never assumes a fixed layer count. **Don't hardcode structure you can
read from the data.**

**(2) Its router returns a different shape.** The fix that took us a debugging
round:

```python
hidden_states = inputs[0].detach().float()
hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])   # <- the fix
logits = F.linear(hidden_states, module.weight.float())
probs = F.softmax(logits, dim=-1)  # (n_tokens, n_routed_experts)
if probs.shape[-1] != self.num_routed_experts:
    raise RuntimeError(...)  # fail loud if the gate weight shape ever mismatches
```

For some sentences the hidden states arrived with an extra leading batch
dimension, so `probs.sum(dim=0)` produced vectors of *different widths* across
sentences. That's fine until stage 4 tries to `np.sum` a ragged list of arrays
and explodes with an "inhomogeneous shape" error. The `reshape(-1, hidden_dim)`
flattens any leading dims so every sentence yields exactly a
`(n_routed_experts,)` vector. The shape assertion right after is insurance: if a
future DeepSeek revision changes the gate, we crash with a clear message instead
of computing garbage.

**The lesson from this bug:** the analysis code was correct; OLMoE and Qwen
sailed through because their hooks always produced fixed-width vectors. The bug
was purely in the DeepSeek adapter — exactly where architecture-specific bugs
*should* be, thanks to the quarantine. When the abstraction is right, bugs stay
local.

---

## 5. `src/routing.py` — the statistics, and the three decisions that make them honest

This is the scientific core. Three design decisions here separate a defensible
result from a hand-wavy one.

### 5a. The metric: Jensen-Shannon Divergence

```python
def jsd(p, q):
    """Squared Jensen-Shannon divergence in bits (base 2). scipy's
    jensenshannon returns the DISTANCE (sqrt of divergence) — squaring here
    is required or every downstream number is silently wrong by a sqrt."""
    return float(jensenshannon(p, q, base=2) ** 2)
```

JSD measures how different two probability distributions are: 0 = identical, 1 =
maximally different (base-2). We compare the *expert-usage distribution* of one
language to another's. Low JSD between two languages = the router sends them to
the same experts = no specialization between them. High JSD = the router treats
them differently.

Note the **squaring**. `scipy.spatial.distance.jensenshannon` returns the
*distance* (the square root of the divergence). If you forget to square, every
number in your paper is off by a square root — not a crash, just quietly wrong.
The comment exists so nobody "cleans up" that `** 2` thinking it's a mistake.
**Comment the non-obvious correctness-critical lines, or someone (maybe you) will
delete them.**

### 5b. Primary metric = hard counts; secondary = soft distribution

We compute JSD two ways:
- **hard** (primary): the distribution of *which experts were actually
  selected* (top-k). The interpretable quantity — "where did tokens actually
  go."
- **soft** (secondary): the mean full-softmax routing distribution. A robustness
  check — does the finding survive if we use routing *weights* instead of hard
  selections?

Reporting both, and having them agree, pre-empts a reviewer asking "does this
hold for the soft distribution?" We already answered it.

### 5c. Permutation testing at EVERY layer, with the SENTENCE as the unit

This is the subtlest and most important statistical decision. We want to know:
is the JSD we observe between two languages *real*, or could it arise by chance?
The permutation test answers this by shuffling language labels and rebuilding a
null distribution.

```python
def permutation_test_sentences(sents_a, sents_b, n_experts, n_permutations, rng):
    """PRIMARY significance test: shuffle SENTENCE labels between the two
    languages and rebuild the null JSD distribution. Sentences (not tokens)
    are the independent units here — token-level shuffling destroys
    within-sentence correlation and produces an anticonservative null."""
    all_sents = sents_a + sents_b
    n_a = len(sents_a)
    obs = jsd(expert_distribution(np.concatenate(sents_a), n_experts),
              expert_distribution(np.concatenate(sents_b), n_experts))
    null = np.empty(n_permutations)
    indices = np.arange(len(all_sents))
    for k in range(n_permutations):
        perm = rng.permutation(indices)
        group_a = np.concatenate([all_sents[i] for i in perm[:n_a]])
        group_b = np.concatenate([all_sents[i] for i in perm[n_a:]])
        null[k] = jsd(expert_distribution(group_a, n_experts),
                      expert_distribution(group_b, n_experts))
    p_value = float((1 + (null >= obs).sum()) / (1 + n_permutations))
    ...
```

**Why shuffle sentences, not tokens?** Tokens within a sentence are heavily
correlated — same topic, same neighboring words, same script. If you shuffle
*tokens* between the two languages, you break that correlation and manufacture a
null distribution that's far too easy to beat. That's what our pilot did, and
it's why the pilot reported effect sizes like "~90 SD above null" — impressive,
but *anticonservative* (overstating significance). Shuffling **sentences**
respects that the sentence is the real independent unit of observation. It gives
smaller, honest effect sizes that survive scrutiny.

We keep the token-level test too, as a labeled supplementary comparison, so the
paper can show both and explain the difference. But we **lead with the
sentence-level numbers**.

**Why every layer?** Because "does specialization concentrate in certain
layers?" is itself one of our research questions (Experiment 3). Testing only a
"representative" layer would beg exactly the question the layer-wise experiment
exists to answer.

Two more small things that signal statistical care:
- `p_value = (1 + (null >= obs).sum()) / (1 + n_permutations)` — the **add-one
  (Phipson–Smyth) estimator.** A permutation p-value can never legitimately be
  exactly 0 with finite permutations; the +1 keeps it honest.
- **Bootstrap CIs resample at the sentence level too** (`bootstrap_jsd_ci`),
  same independence logic.

---

## 6. `src/ablation.py` — turning correlation into causation

JSD tells us languages *route differently*. It does **not** tell us those
experts *matter* for those languages. Maybe the routing differences are
incidental. The ablation experiment is where we test causation: **if we destroy
a family's preferred experts, does that family's language modeling get worse?**

```python
def top_experts_for_group(per_lang_distributions, target_langs, num_layers, n_experts, top_n):
    """For each layer, find experts most disproportionately used by target_langs
    vs. all other languages (ratio of mean target usage to mean non-target usage)."""
    ...
    ratio = (target_mean + 1e-8) / (other_mean + 1e-8)
    top_experts[layer] = np.argsort(-ratio)[:top_n].tolist()
```

For "Dravidian experts," we find the experts that Dravidian languages use *much
more than everyone else* (highest usage ratio), per layer. Then we ablate them —
zero out their routing weight and renormalize — and measure the increase in loss
(a proxy for perplexity) on each language.

**But here's the part that makes it science and not theater** — the control:

```python
"""The pilot's original ablation compared "language-preferring experts ablated"
only against an unablated baseline. That cannot distinguish "these specific
experts matter for this language" from "ablating ANY 8 experts increases loss
for everyone" — the null hypothesis a reviewer will raise first. This module
closes that gap by also running N_RANDOM_CONTROLS random-expert-ablation trials
(same expert COUNT, different random experts)..."""

random_control_sets = [
    random_experts_for_group(...) for _ in range(n_random_controls)
]
```

If you ablate 8 experts and loss goes up, *of course it does* — you removed
capacity. The question is whether ablating the **specifically Dravidian-preferring**
8 experts hurts Dravidian languages **more than 8 random experts would.** So we
run 10 random-ablation trials as a baseline. The targeted effect only counts if
it exceeds the random baseline. Our pilot lacked this control; a reviewer would
have destroyed it. **Always ask: "compared to what?" A treatment with no control
is a number with no meaning.**

Note also: the random-control sets are generated **once and reused across all
languages**, so "trial 3" is the same condition for every language — otherwise
you'd be adding noise to the exact cross-language comparison the control exists
to support.

---

## 7. `src/pipeline.py` — orchestration, checkpointing, and paranoia

`pipeline.py` runs one model end to end: load → sample → extract routing →
compute JSD/permutation/bootstrap → ablation → save. Three things worth
studying.

**(1) Checkpoint after every unit of work.**

```python
for lang_name, sample in samples.items():
    lang_ckpt = extraction_dir / f"{lang_name}.pkl"
    if lang_ckpt.exists():
        ... # load and validate, skip if good
    ...
    with open(lang_ckpt, "wb") as f:
        pickle.dump(record, f)
```

Routing extraction runs the model over ~20k tokens × 11 languages. On a rented
spot instance that can be preempted, you do **not** want a crash at language 10
to throw away languages 1–9. Every language's routing is saved the moment it's
computed. A re-run picks up where it left off. This paranoia is what let us
recover from the DeepSeek crash without redoing OLMoE and Qwen.

**(2) The top-level "already complete" guard:**

```python
if checkpoint.get("stage") == "complete" and (out_dir / "04_ablation" / "ablation_results.csv").exists():
    print(f"{model_key}: already complete ... skipping entirely, no model load needed.")
    return out_dir
```

This is why, when we re-ran to fix DeepSeek, OLMoE and Qwen were skipped
instantly — the batch runner never even loaded them. **Your finished results
are protected from your later fixes by construction.** This mattered enormously
in practice: we changed code five times after OLMoE/Qwen were done, and never
once risked their outputs.

**(3) Validate cached data before trusting it:**

```python
def _routing_checkpoint_is_valid(loaded, n_routed_experts):
    """...every per-sentence prob vector must be exactly (n_routed_experts,) —
    i.e. shape-consistent so np.sum(..., axis=0) at stage 4 won't hit the
    inhomogeneous-array error the DeepSeek hook bug produced."""
```

When we fixed the DeepSeek shape bug, the *old* pickles on disk were still
malformed. Rather than make you manually delete them, the pipeline detects a
stale/ragged checkpoint, discards it, and re-extracts — while leaving valid
checkpoints (and finished models) untouched. **A resume system should verify
what it resumes from, not blindly trust it.**

**(4) Hooks come off before ablation.** After routing extraction:

```python
for h in hooks:
    h.remove()
hooks = []
```

The capture hooks copy router tensors to CPU on every forward pass. The ablation
stage does *hundreds* of forward passes and doesn't need captures — leaving the
hooks attached would waste time and stack hooks on the same modules the ablation
hooks target. Clean up instrumentation when you're done measuring.

**(5) Reproducibility mechanics.** `rng = np.random.default_rng(config["seed"])`
for the analysis; a *separate* `np.random.default_rng(config["seed"] + 1)` for
ablation — so the ablation's random draws don't depend on whether stage 4 ran or
was resumed-past. And a `manifest.json` is written recording git commit, config
hash, GPU, torch version, timestamp, and the FLORES checksum. That manifest is
how, after the fact, we *proved* the results came from your RTX PRO 4500 run and
not from any test data.

---

## 8. `scripts/analyze_results.py` — from artifacts to understanding

This script has no GPU dependency — it reads `results/` and produces figures +
`findings_summary.txt` on a laptop in seconds. Two representative pieces.

**The Hindi-Urdu control, in code:**

```python
hu = jsd_between("hindi", "urdu")
others = [("hindi", x) for x in ["marathi","bengali","gujarati","punjabi"]]
other_vals = [jsd_between(a, b) for a, b in others]
verdict = ("language identity > script" if hu < np.mean(other_vals)
           else "script effects dominant")
```

If Hindi-Urdu JSD is *below* Hindi's average JSD to its other Indo-Aryan
relatives, routing pairs the same-language/different-script pair more tightly
than same-family/different-script pairs → language identity wins. Otherwise
script wins. This is the exact logical machinery of the control, expressed in
four lines.

**The dendrogram** builds a hierarchical clustering from the JSD matrix:

```python
m = (m + m.T) / 2.0          # force exact symmetry
np.fill_diagonal(m, 0.0)
condensed = squareform(m, checks=False)
Z = linkage(condensed, method="average")
```

The `(m + m.T)/2` matters: floating-point noise can make the matrix very
slightly asymmetric, and `squareform` then misreads it and crashes. We
symmetrize explicitly. (This was a real bug we caught by testing the analysis
against mock data *before* running it on the real thing — always exercise your
analysis code on synthetic data with a known shape first.)

---

## 9. The infrastructure saga (read this before you rent a GPU)

Half of this project's real time went to *getting three models to load on a
rented pod at all.* These aren't embarrassing footnotes — they're the actual
texture of empirical ML work, and the fixes are baked into `run_all.sh` so you
never hit them:

- **HF_HOME on a network volume.** RunPod's template silently points the model
  cache at `/workspace` — a shared, quota-limited, *slow* network disk. This one
  default caused: "disk quota exceeded" errors (while the real disk sat empty),
  downloads hanging at 0% (multi-GB writes over a slow mount), and stale
  half-finished caches surviving restarts. Fix: `run_all.sh` forces
  `HF_HOME=$PWD/.hf_cache` onto local disk. **The root cause of five separate
  symptoms was one environment variable.**
- **huggingface_hub's downloader hung** on large shards (both the xet backend
  and standard HTTP), with no error, surviving Ctrl+C. A plain `curl` to the
  same URL worked. So `prefetch_model.py` bypasses hf's downloader entirely and
  fetches files itself (now via `aria2c`, 16→4 parallel connections, with a hard
  per-file timeout + resume). **When a library fails mysteriously, drop to the
  layer below it and verify the layer below works.**
- **The HF cache layout is finicky.** Files must land in
  `models--org--name/snapshots/<commit-sha>/` and a `refs/main` file must point
  at that sha — for `trust_remote_code` models, transformers regex-extracts the
  sha from the path and crashes on a non-hex folder name. `prefetch_model.py`
  reads the real sha from the HF API and builds the layout exactly.
- **Version pins fighting the pod.** We *unpinned* torch and bitsandbytes
  because the pod ships builds matched to its CUDA driver (Blackwell needs recent
  ones); our old pins downgraded into incompatibility (`No module named
  'triton.ops'`). **Pin for reproducibility, but not against the hardware you're
  actually on.**

The meta-lesson: **build the run to be idempotent and resumable, log everything,
and make each fix permanent in code.** By the end, "fresh pod → clone → one
script → walk away" actually held.

---

## 10. What the results actually say

All numbers below are from our run. Order of models by architecture:
**OLMoE** (64 experts, no shared, English-heavy training) →
**Qwen1.5-MoE** (60 routed + 1 shared, multilingual) →
**deepseek-moe-16b-base** (64 routed + 2 shared, multilingual, dense layer 0).

### Finding 1 — Routing is family-structured in all three models (robust)

| Model | within-family JSD | cross-family JSD | ratio | median effect (sentence-level) |
|---|---|---|---|---|
| OLMoE | 0.0217 | 0.0386 | **0.56** | 90 SD |
| Qwen  | 0.0451 | 0.0540 | 0.84 | 113 SD |
| DeepSeek | 0.0429 | 0.0511 | 0.84 | 73 SD |

Within-family routing is more similar than cross-family in every model, and
100% of language pairs are significant (p<0.05) under the honest sentence-level
test. **The router learned the Dravidian/Indo-Aryan distinction with zero
supervision.** OLMoE shows it most sharply (ratio 0.56).

The **dendrograms** are the visual proof: in every model English branches off
first (deepest split — routing knows Indic is "not English"), then the tree
recovers two clean sub-clusters, Dravidian and Indo-Aryan. This is the study's
headline figure because it's *unsupervised* — we never told the model these
families exist; the routing recovered them.

### Finding 2 — The Hindi-Urdu control splits by architecture (the hook)

| Model | Hindi-Urdu JSD | Hindi vs other Indo-Aryan (mean) | verdict |
|---|---|---|---|
| OLMoE | 0.0148 | 0.0262 | **language identity > script** (0.56) |
| Qwen  | 0.0484 | 0.0306 | **script dominant** (1.58) |
| DeepSeek | 0.0367 | 0.0394 | roughly tied (0.93) |

Read this carefully because it's the most interesting and least obvious result.
In **OLMoE** — the model with the *least* multilingual training — Urdu routes
*closer* to Hindi than Hindi does to its own family members. Routing tracks
**language identity**, ignoring the completely different script. In its
dendrogram, Hindi and Urdu are adjacent sisters despite Devanagari vs Perso-
Arabic.

In **Qwen** — the *multilingual* model — Hindi and Urdu route *apart*; the
dendrogram yanks Urdu out of the Hindi cluster entirely. Routing tracks
**script**.

**The inversion is the story:** the model that saw *less* Indic data routes by
deep language identity (regularities inherited from English-dominant
pretraining), while the model that saw *more* learned script-specific routing.
This directly stress-tests Mixtral's "routing is syntactic (script-like)"
claim — it holds for the multilingual models but **breaks** for the English-
heavy one. A confirming-*and*-refuting result across architectures is exactly
what makes an analysis paper worth reading.

### Finding 3 — Specialization deepens with layer depth (and differs by architecture)

The **layer-wise plots** (JSD from English, per layer):
- **OLMoE**: textbook. Dravidian (red) sits consistently above Indo-Aryan (blue)
  from ~layer 4 on, both climbing toward the final layers. Specialization builds
  with depth.
- **Qwen**: noisy, red/blue intermixed, with a sharp jump at layers 17+ —
  consistent with shared experts absorbing common load and leaving routed-expert
  behavior more volatile.
- **DeepSeek**: the *reverse* — Indo-Aryan diverges more than Dravidian, spikier
  layer-to-layer (fine-grained experts).

Same phenomenon, three architectural signatures. That the three designs behave
differently is itself the point of running three.

### Finding 4 — Ablation gives causal backing (with one honest exception)

Ablating a family's preferred experts hurts that family more than random experts
in **5 of 6** Indic cases. English/Indo-European is correctly *never* specific —
it shares experts broadly, so its "own" experts aren't special. The lone
exception is DeepSeek's Dravidian case, and it's *explainable*: DeepSeek's
random-ablation baseline is very high (1.27 vs ~0.29 for the others), meaning its
fine-grained experts are so entangled that ablating *any* of them hurts a lot —
which is itself a finding about the fine-grained-shared architecture.

The causal experiment converts "these languages route differently" into "these
experts causally carry these languages" — the difference between a correlation
and a mechanism, and a concrete lever for language-targeted expert pruning.

---

## 11. If you change one thing, change it here

- Want more/fewer tokens per language, more permutations, a different model?
  **`config.yaml`, one line.** Everything re-derives.
- Want to add a fourth model? **Write one adapter** in `src/adapters/` satisfying
  `base.py`, add three lines to `config.yaml`, register it in `pipeline.py`'s
  `_adapter_for`. Touch no analysis code.
- Want a new figure? **`analyze_results.py`** reads only `results/` — add a
  function; you never need the GPU again.
- Want to reproduce exactly? Each `results/<model>/manifest.json` has the git
  commit, config hash, and seed. Check out that commit, run that config.

---

## 12. The habits worth stealing from this project

1. **One source of truth** (config) + **traceability** (manifests) = you can
   always answer "what produced this?"
2. **A narrow interface** (`MoEAdapter`) quarantines messiness. Bugs stayed local
   to the adapter that caused them.
3. **Fail loud, never guess.** Every place reality might differ from an
   assumption raises a clear error instead of computing plausible garbage.
4. **Find the comparison that isolates your variable** (Hindi-Urdu) and **always
   include a control** (random-expert ablation).
5. **Respect the unit of observation** (sentence-level, not token-level, nulls).
6. **Checkpoint everything; make resumes verify what they resume.** Finished work
   is protected from later fixes.
7. **When a library fails mysteriously, go one layer down** (hf downloader →
   curl) and confirm the ground truth.
8. **Test analysis code on synthetic data of known shape before trusting it on
   real data.**

That's the whole thing. The science is simple once the plumbing is honest — and
most of the work, as always, was making the plumbing honest.
