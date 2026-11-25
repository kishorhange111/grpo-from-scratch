"""The GRPO objective, and Dr. GRPO's two corrections to it.

For a prompt q, the policy samples a *group* of G completions o_1..o_G, each with scalar
reward r_i. GRPO (Shao et al. 2024, DeepSeekMath) needs no value network. Each
completion's advantage is its reward compared with the rest of its group:

    GRPO:     Â_i = (r_i − mean(r)) / (std(r) + ε)
    Dr. GRPO: Â_i =  r_i − mean(r)                       (Liu et al. 2025)

Every token t of o_i gets the same Â_i. The per-token surrogate is PPO's clipped ratio
plus a KL penalty to the reference policy:

    ℓ_{i,t} = −min(ρ Â, clip(ρ, 1 − ε, 1 + ε) Â) + β · KL_k3,
    ρ = π_θ / π_old,  KL_k3 = π_ref/π_θ − log(π_ref/π_θ) − 1  (≥ 0, unbiased; Schulman 2020)

**Aggregation** is where the two differ. GRPO averages over each sequence's tokens and then
over sequences, so long wrong answers are penalised *less* per token. Dr. GRPO argues this
biases the policy toward long incorrect outputs, and that dividing by std(r) up-weights
too-easy and too-hard prompts. It therefore sums over tokens and divides by a constant
(the generation budget), and it drops the std.
"""

from __future__ import annotations

import torch


def advantages(rewards: torch.Tensor, scale: bool = True, eps: float = 1e-4) -> torch.Tensor:
    """``rewards (n_prompts, G)`` -> advantages of the same shape."""
    a = rewards - rewards.mean(1, keepdim=True)
    if scale:
        a = a / (rewards.std(1, keepdim=True) + eps)
    return a


def token_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """log π(tokens) for ``logits (b, t, V)`` predicting ``tokens (b, t)``, in fp32."""
    return torch.log_softmax(logits.float(), -1).gather(-1, tokens[..., None])[..., 0]


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    d = logp_ref - logp
    return d.exp() - d - 1


def loss(
    logp: torch.Tensor,  # (b, t) under the current policy, with grad
    logp_old: torch.Tensor,  # (b, t) under the sampling policy
    adv: torch.Tensor,  # (b,)
    mask: torch.Tensor,  # (b, t) 1 on completion tokens
    logp_ref: torch.Tensor | None = None,
    beta: float = 0.0,
    clip: float = 0.2,
    agg: str = "grpo",  # "grpo": token-mean per sequence then mean; "dr": sum / constant
    max_len: int | None = None,
):
    ratio = (logp - logp_old).exp()
    a = adv[:, None]
    per_tok = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    if beta and logp_ref is not None:
        per_tok = per_tok + beta * kl_k3(logp, logp_ref)
    mask = mask.to(per_tok.dtype)
    if agg == "grpo":
        out = ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    elif agg == "dr":
        out = (per_tok * mask).sum() / (mask.shape[0] * (max_len or mask.shape[1]))
    else:
        raise ValueError(agg)
    with torch.no_grad():
        clipped = ((ratio - 1).abs() > clip).to(mask.dtype)
        frac = float((clipped * mask).sum() / mask.sum().clamp(min=1))
    return out, {"clip_frac": frac}
