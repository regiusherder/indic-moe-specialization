"""Common interface every model adapter must implement.

Each MoE architecture (OLMoE, Qwen1.5-MoE, DeepSeek-V2-Lite) exposes routing
internals differently and has a different shared/routed expert split. Rather
than branching on model name throughout the pipeline, each adapter normalizes
its model to this interface so routing.py and ablation.py are architecture-agnostic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RoutingCapture:
    """One forward pass worth of routing data for a single layer."""
    layer_idx: int
    routing_probs: "torch.Tensor"     # (n_tokens, n_routed_experts) full softmax, routed experts only
    selected_experts: "torch.Tensor"  # (n_tokens, top_k) int64 indices into routed experts


class MoEAdapter(ABC):
    """Subclass per architecture. Must NOT assume anything about other adapters."""

    #: number of experts that are always active regardless of routing (0 if none)
    n_shared_experts: int = 0

    @abstractmethod
    def load(self, hf_id: str, revision: str | None, quantization: str):
        """Load tokenizer + model, store on self. Must set self.tokenizer, self.model,
        self.num_layers, self.num_routed_experts, self.top_k."""
        ...

    @abstractmethod
    def register_hooks(self) -> list:
        """Attach forward hooks that populate self._captures: dict[layer_idx -> RoutingCapture].
        Returns the list of hook handles (caller is responsible for .remove()'ing them)."""
        ...

    @abstractmethod
    def get_captures(self) -> dict[int, RoutingCapture]:
        """Return this forward pass's captured routing data, keyed by layer index.
        Must be called immediately after a forward pass and before the next one —
        implementations should clear internal state on each fresh forward."""
        ...

    @abstractmethod
    def ablate_experts(self, experts_by_layer: dict[int, list[int]]):
        """Context manager: zero out (and renormalize) routing weight for the given
        routed-expert indices, per layer, for the duration of the `with` block."""
        ...

    def clear_captures(self):
        self._captures = {}
