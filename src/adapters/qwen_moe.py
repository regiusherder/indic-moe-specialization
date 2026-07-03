"""Qwen1.5-MoE-A2.7B adapter. DeepSeekMoE-style: 60 fine-grained routed experts
(top-4) + 1 shared expert (a single wider MLP, gated by `shared_expert_gate`)
that fires on every token unconditionally.

VERIFIED against the live checkpoint on 2026-07-03 (RunPod, transformers
4.45.2): `model.model.layers[0].mlp` is `Qwen2MoeSparseMoeBlock` with
`.gate` (Linear(2048→60), returns raw logits directly — NOT a tuple like
OLMoE's newer format), `.experts` (ModuleList of 60 `Qwen2MoeMLP`), a single
`.shared_expert` (one wider `Qwen2MoeMLP`, not 4 as an earlier design note
assumed — the "DeepSeekMoE-style" architecture family can have any number of
shared experts; Qwen1.5-MoE-A2.7B specifically has exactly 1), and
`.shared_expert_gate` (Linear(2048→1)). This adapter hooks `.gate` for
routing capture and rewrites its logits for ablation (reimplementing top-k
dispatch on the routed experts only, skipping the shared expert entirely,
since shared-expert output is never zeroed by design).
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import MoEAdapter, RoutingCapture


class QwenMoEAdapter(MoEAdapter):
    n_shared_experts = 1  # Qwen1.5-MoE-A2.7B: 1 shared expert, always active (verified live)

    def load(self, hf_id: str, revision: str | None, quantization: str):
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                # router stays full precision — see olmoe.py for rationale;
                # also covers Qwen's shared_expert_gate
                llm_int8_skip_modules=["gate", "shared_expert_gate"],
            )

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            revision=revision,
            quantization_config=bnb_config,
            dtype=torch.float16 if bnb_config is None else None,
            device_map="auto",
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_routed_experts = self.model.config.num_experts
        self.top_k = self.model.config.num_experts_per_tok
        self._captures: dict[int, RoutingCapture] = {}
        self.resolved_revision = getattr(self.model.config, "_commit_hash", revision)

        # fail loud if the module shape isn't what this adapter assumes
        sample_mlp = self.model.model.layers[0].mlp
        for attr in ("gate", "experts", "shared_expert", "shared_expert_gate"):
            if not hasattr(sample_mlp, attr):
                raise AttributeError(
                    f"QwenMoEAdapter assumes model.model.layers[i].mlp.{attr} exists; "
                    f"it doesn't on this checkpoint/transformers version. "
                    f"Print `model.model.layers[0].mlp` and update this adapter before proceeding."
                )

    def register_hooks(self) -> list:
        handles = []
        for i, layer in enumerate(self.model.model.layers):
            handles.append(layer.mlp.gate.register_forward_hook(self._make_hook(i)))
        return handles

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Qwen2MoeSparseMoeBlock's internal gate Linear returns raw logits directly (not a tuple)
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"Qwen gate hook expected a raw logits tensor but got {type(output)} at "
                    f"layer {layer_idx} — the transformers version changed the router return "
                    f"format. Inspect model.model.layers[0].mlp.gate and update this adapter "
                    f"(and its ablation hook, which rewrites these same logits)."
                )
            logits = output.detach().float()
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_idx = probs.topk(self.top_k, dim=-1)
            self._captures[layer_idx] = RoutingCapture(
                layer_idx=layer_idx,
                routing_probs=probs.cpu(),
                selected_experts=top_k_idx.cpu(),
            )
        return hook

    def get_captures(self) -> dict[int, RoutingCapture]:
        return self._captures

    def ablate_experts(self, experts_by_layer: dict[int, list[int]]):
        return _QwenAblator(self.model, experts_by_layer, self.top_k)


class _QwenAblator:
    """Ablates routed experts only; shared experts are never touched (by design —
    they are excluded from the specialization hypothesis being tested)."""

    def __init__(self, model, experts_by_layer, top_k):
        self.model = model
        self.experts_by_layer = experts_by_layer
        self.top_k = top_k
        self.hooks = []

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx not in self.experts_by_layer:
                continue
            to_zero = self.experts_by_layer[layer_idx]
            self.hooks.append(layer.mlp.gate.register_forward_hook(self._make_ablation_hook(to_zero)))
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _make_ablation_hook(self, experts_to_zero):
        def hook(module, inputs, output):
            # Rewrite the raw router logits so ablated experts get -inf,
            # forcing top-k selection to skip them. This changes what the
            # *rest* of the block (which recomputes softmax/topk from these
            # logits downstream) actually dispatches to — verify this matches
            # the live Qwen2MoeSparseMoeBlock.forward control flow before trusting results.
            logits = output.clone()
            logits[:, experts_to_zero] = float("-inf")
            return logits
        return hook
