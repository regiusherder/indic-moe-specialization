"""DeepSeek-V2-Lite adapter. DeepSeekMoE architecture: 64 fine-grained routed
experts (top-6) + 2 shared experts, always active.

IMPORTANT — unverified against a live checkpoint, same caveat as qwen_moe.py:
DeepSeek-V2's HF implementation (`modeling_deepseek.py`, loaded via
trust_remote_code=True since it is NOT in mainline transformers) exposes
`.gate` returning `(topk_idx, topk_weight, aux_loss)` per the official repo,
which is a DIFFERENT return shape than both OLMoE and Qwen. This adapter
hooks accordingly but MUST be checked against the actual loaded module
before a real run — trust_remote_code models can change their internals
between revisions without a version bump.

Action before running for real: load with trust_remote_code=True, print
`model.model.layers[0].mlp.gate` and inspect its forward signature/return in
the cached modeling file under the HF cache, and adjust this hook if it drifts.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import MoEAdapter, RoutingCapture


class DeepSeekV2LiteAdapter(MoEAdapter):
    n_shared_experts = 2  # DeepSeek-V2-Lite: 2 shared experts, always active

    def load(self, hf_id: str, revision: str | None, quantization: str):
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                # DeepSeek's MoEGate is a custom module (not nn.Linear) so bnb
                # shouldn't quantize it anyway; listing it is a harmless belt-
                # and-suspenders in case a revision refactors it into a Linear.
                llm_int8_skip_modules=["gate"],
            )

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            revision=revision,
            quantization_config=bnb_config,
            dtype=torch.float16 if bnb_config is None else None,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_routed_experts = self.model.config.n_routed_experts
        self.top_k = self.model.config.num_experts_per_tok
        self._captures: dict[int, RoutingCapture] = {}
        self.resolved_revision = getattr(self.model.config, "_commit_hash", revision)

        sample_mlp = self.model.model.layers[1].mlp  # layer 0 is dense in DeepSeek-V2-Lite; MoE starts layer 1
        if not hasattr(sample_mlp, "gate"):
            raise AttributeError(
                "DeepSeekV2LiteAdapter assumes model.model.layers[i>=1].mlp.gate exists; "
                "it doesn't on this checkpoint/trust_remote_code revision. "
                "Inspect the cached modeling_deepseek.py and update this adapter before proceeding."
            )

    def register_hooks(self) -> list:
        handles = []
        for i, layer in enumerate(self.model.model.layers):
            if not hasattr(layer.mlp, "gate"):
                continue  # dense (non-MoE) layer, e.g. layer 0 in V2-Lite — nothing to hook
            handles.append(layer.mlp.gate.register_forward_hook(self._make_hook(i)))
        return handles

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # DeepSeek-V2 gate forward returns (topk_idx, topk_weight, aux_loss);
            # it does NOT expose full pre-topk softmax probs directly, so we
            # recompute them from the router's raw logits via the module's own
            # weight matrix applied to the hook's input — this mirrors the
            # official repo's `self.weight` Linear-less matmul router.
            hidden_states = inputs[0].detach().float()
            logits = F.linear(hidden_states, module.weight.float())
            probs = F.softmax(logits, dim=-1)
            topk_idx = output[0].detach().cpu()
            self._captures[layer_idx] = RoutingCapture(
                layer_idx=layer_idx,
                routing_probs=probs.cpu(),
                selected_experts=topk_idx,
            )
        return hook

    def get_captures(self) -> dict[int, RoutingCapture]:
        return self._captures

    def ablate_experts(self, experts_by_layer: dict[int, list[int]]):
        return _DeepSeekAblator(self.model, experts_by_layer)


class _DeepSeekAblator:
    def __init__(self, model, experts_by_layer):
        self.model = model
        self.experts_by_layer = experts_by_layer
        self.hooks = []

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx not in self.experts_by_layer or not hasattr(layer.mlp, "gate"):
                continue
            to_zero = self.experts_by_layer[layer_idx]
            self.hooks.append(layer.mlp.gate.register_forward_hook(self._make_ablation_hook(to_zero)))
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @staticmethod
    def _make_ablation_hook(experts_to_zero):
        def hook(module, inputs, output):
            topk_idx, topk_weight, aux_loss = output
            mask = torch.zeros_like(topk_idx, dtype=torch.bool)
            for expert_id in experts_to_zero:
                mask = mask | (topk_idx == expert_id)
            topk_weight = topk_weight.clone()
            topk_weight[mask] = 0.0
            row_sum = topk_weight.sum(dim=-1, keepdim=True)
            row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
            topk_weight = topk_weight / row_sum
            return (topk_idx, topk_weight, aux_loss)
        return hook
