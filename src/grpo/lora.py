"""LoRA (Hu et al. 2021) by hand: ``W x + (α/r) · B A x`` with A random and B zero-initialised.

Only A and B train, so optimiser state and checkpoints are tiny, which suits RL runs that
have to resume across short GPU sessions. With B = 0 at the start, the adapted model is
exactly the base model. That gives the frozen reference policy for the KL term for free:
switch the adapters off.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = (x.to(self.A.dtype) @ self.A.t()) @ self.B.t()
        return y + (self.scale * d).to(y.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight.float() + self.scale * self.B @ self.A


def inject(model: nn.Module, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=16, alpha=32.0):
    """Replace every ``nn.Linear`` whose attribute name is in ``targets``. Returns the adapters."""
    for p in model.parameters():
        p.requires_grad_(False)
    found = []
    for _, mod in list(model.named_modules()):
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, r, alpha).to(child.weight.device)
                setattr(mod, name, lora)
                found.append(lora)
    if not found:
        raise ValueError(f"no Linear named {targets}")
    return found


def adapters(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


@contextmanager
def disabled(model: nn.Module):
    """The base (reference) model: all adapters off inside the block."""
    ads = adapters(model)
    for a in ads:
        a.enabled = False
    try:
        yield model
    finally:
        for a in ads:
            a.enabled = True


def lora_state(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
