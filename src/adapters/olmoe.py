"""OLMoE-1B-7B adapter. Standard Mixtral-style MoE: all experts are "routed",
no shared experts. Router lives at model.model.layers[i].mlp.gate.

The gate's forward return format DIFFERS across transformers versions:
  - older (~4.45): gate is nn.Linear, hook output is a raw logits tensor
  - newer (~4.5x, what the executed pilot ran on): gate returns a
    (logits, weights, selected_experts) tuple — the pilot's output[2] hook
Both formats are handled below; the detected format is logged at load time
and recorded on the adapter so the manifest can capture it. If the hook sees
a shape it can't classify it raises immediately rather than capturing garbage.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import MoEAdapter, RoutingCapture


class OLMoEAdapter(MoEAdapter):
    n_shared_experts = 0
    gate_output_format = "undetected"  # set on first hook firing: "tuple" or "tensor"

    def load(self, hf_id: str, revision: str | None, quantization: str):
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                # keep the ROUTER in full precision: quantizing the gate
                # perturbs the exact routing logits this study measures.
                # (matches module-name suffix "gate" = router; experts'
                # "gate_proj" is a different name and stays quantized)
                llm_int8_skip_modules=["gate"],
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

    def register_hooks(self) -> list:
        handles = []
        for i, layer in enumerate(self.model.model.layers):
            handles.append(layer.mlp.gate.register_forward_hook(self._make_hook(i)))
        return handles

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
                raise RuntimeError(
                    f"OLMoE gate hook got unrecognized output type {type(output)} "
                    f"(len={len(output) if isinstance(output, tuple) else 'n/a'}) at layer {layer_idx}. "
                    f"The transformers version changed the router return format again — "
                    f"inspect model.model.layers[0].mlp.gate and update this adapter."
                )
            probs = F.softmax(logits.detach().float(), dim=-1)
            self._captures[layer_idx] = RoutingCapture(
                layer_idx=layer_idx,
                routing_probs=probs.cpu(),
                selected_experts=selected,
            )
        return hook

    def get_captures(self) -> dict[int, RoutingCapture]:
        return self._captures

    def ablate_experts(self, experts_by_layer: dict[int, list[int]]):
        return _OLMoEAblator(self.model, experts_by_layer)


class _OLMoEAblator:
    def __init__(self, model, experts_by_layer):
        self.model = model
        self.experts_by_layer = experts_by_layer
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

    @staticmethod
    def _make_ablation_hook(experts_to_zero):
        def hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) >= 3:
                # newer transformers: zero the routing weights of ablated
                # experts and renormalize the survivors
                logits, weights, selected = output[0], output[1], output[2]
                mask = torch.zeros_like(selected, dtype=torch.bool)
                for expert_id in experts_to_zero:
                    mask = mask | (selected == expert_id)
                weights = weights.clone()
                weights[mask] = 0.0
                row_sum = weights.sum(dim=-1, keepdim=True)
                row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
                weights = weights / row_sum
                return (logits, weights, selected)
            elif torch.is_tensor(output):
                # older transformers: gate returns raw logits and the MoE block
                # does softmax/top-k downstream — force ablated experts to -inf
                # so they can never be selected
                logits = output.clone()
                logits[..., experts_to_zero] = float("-inf")
                return logits
            else:
                raise RuntimeError(
                    f"OLMoE ablation hook got unrecognized gate output type {type(output)} — "
                    f"see the capture hook's error guidance in this file."
                )
        return hook
