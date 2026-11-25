import random

import pytest
import torch
from torch import nn

from grpo import rl
from grpo.lora import LoRALinear, adapters, disabled, inject, lora_state
from grpo.task import make_problem, parse, problems, reward


# ------------------------------------------------------------------------------ objective
def test_advantages():
    r = torch.tensor([[1.0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    a = rl.advantages(r)
    torch.testing.assert_close(a.mean(1), torch.zeros(3))
    assert a[1].abs().max() == 0  # an all-correct group carries no signal
    assert a[0].std() == pytest.approx(1.0, abs=1e-3)
    d = rl.advantages(r, scale=False)
    torch.testing.assert_close(d[2], torch.tensor([-0.25, -0.25, -0.25, 0.75]))
    # std scaling up-weights the hard prompt (1 of 4 correct) relative to Dr. GRPO
    assert a[2, 3] / d[2, 3] > a[0, 0] / d[0, 0]


def test_on_policy_gradient_is_reinforce():
    """At ratio 1 and β = 0, ∂loss/∂logp = −Â / |o| (GRPO) or −Â / (B·L) (Dr. GRPO)."""
    lp = torch.randn(3, 5, requires_grad=True)
    adv, mask = torch.tensor([1.0, -2.0, 0.5]), torch.ones(3, 5)
    mask[2, 3:] = 0
    l, _ = rl.loss(lp, lp.detach(), adv, mask)
    (g,) = torch.autograd.grad(l, lp)
    exp = -adv[:, None] / mask.sum(1, keepdim=True) / 3 * mask
    torch.testing.assert_close(g, exp)
    l, _ = rl.loss(lp, lp.detach(), adv, mask, agg="dr", max_len=10)
    (g,) = torch.autograd.grad(l, lp)
    torch.testing.assert_close(g, -adv[:, None] / 30 * mask)


def test_clipping_stops_the_gradient_outside_the_trust_region():
    old = torch.zeros(2, 1)
    lp = torch.tensor([[0.5], [0.5]], requires_grad=True)  # ratio e^0.5 ≈ 1.65 > 1.2
    l, st = rl.loss(lp, old, torch.tensor([1.0, -1.0]), torch.ones(2, 1))
    (g,) = torch.autograd.grad(l, lp)
    assert g[0, 0] == 0  # positive advantage, ratio already above 1 + ε: clipped
    assert g[1, 0] != 0  # negative advantage: the min keeps the unclipped (worse) term
    assert st["clip_frac"] == 1.0


def test_k3_kl():
    a = torch.randn(100)
    assert rl.kl_k3(a, a).abs().max() == 0
    assert (rl.kl_k3(a, torch.randn(100)) >= 0).all()
    # unbiased: E_{x~π}[k3] = KL(π || ref) for categorical π, ref
    p, q = torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.2, 0.2, 0.6])
    est = (p * rl.kl_k3(p.log(), q.log())).sum()
    torch.testing.assert_close(est, (p * (p / q).log()).sum())


def test_dr_aggregation_does_not_favour_long_wrong_answers():
    """GRPO's per-sequence mean gives each token of a long sequence less weight."""
    adv = torch.tensor([-1.0, -1.0])
    mask = torch.zeros(2, 20)
    mask[0, :2], mask[1, :] = 1, 1  # short and long wrong answers
    lp = torch.zeros(2, 20, requires_grad=True)
    (g,) = torch.autograd.grad(rl.loss(lp, lp.detach(), adv, mask)[0], lp)
    assert g[0, 0] / g[1, 0] == pytest.approx(10.0)
    (g,) = torch.autograd.grad(rl.loss(lp, lp.detach(), adv, mask, agg="dr")[0], lp)
    assert g[0, 0] == pytest.approx(g[1, 0])


def test_token_logprobs():
    logits = torch.randn(2, 4, 7)
    tok = torch.randint(0, 7, (2, 4))
    lp = rl.token_logprobs(logits, tok)
    torch.testing.assert_close(
        lp, -torch.nn.functional.cross_entropy(logits.transpose(1, 2), tok, reduction="none")
    )


# ----------------------------------------------------------------------------------- LoRA
def test_lora_starts_as_identity_and_merges():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    lo = LoRALinear(nn.Linear(8, 6), r=4)
    lo.base.load_state_dict(base.state_dict())
    x = torch.randn(3, 8)
    torch.testing.assert_close(lo(x), base(x))
    with torch.no_grad():
        lo.B.normal_()
    merged = x @ lo.merged_weight().t() + lo.base.bias
    torch.testing.assert_close(lo(x), merged, atol=1e-5, rtol=1e-5)
    assert not lo.base.weight.requires_grad and lo.A.requires_grad


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    c = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(c).eval()


def test_inject_into_qwen_disable_gives_reference_and_only_adapters_train():
    m = _tiny_qwen()
    x = torch.randint(0, 64, (2, 10))
    ref = m(x).logits
    ads = inject(m, r=4)
    assert len(ads) == 2 * 4 and adapters(m) == ads
    torch.testing.assert_close(m(x).logits, ref)
    for a in ads:
        with torch.no_grad():
            a.B.normal_(std=0.5)
    assert (m(x).logits - ref).abs().max() > 1e-3
    with disabled(m):
        torch.testing.assert_close(m(x).logits, ref)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)
    assert set(lora_state(m)) == trainable
    with pytest.raises(ValueError):
        inject(nn.Sequential(nn.ReLU()))


def test_left_padded_logprobs_match_unpadded():
    """The position ids passed to the policy forward must match ``generate``'s."""
    from grpo.train import completion_mask, seq_logprobs

    m = _tiny_qwen()
    seq = torch.randint(1, 64, (1, 9))
    n_prompt = 6
    alone = seq_logprobs(m, seq, torch.ones_like(seq), n_prompt)
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), seq], 1)
    attn = torch.cat([torch.zeros(1, 3, dtype=torch.long), torch.ones_like(seq)], 1)
    torch.testing.assert_close(
        seq_logprobs(m, padded, attn, n_prompt + 3), alone, atol=1e-5, rtol=1e-5
    )
    cm = completion_mask(torch.tensor([[5, 9, 2, 0, 0], [5, 5, 5, 5, 5]]), eos_ids=[2])
    assert cm.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


# ----------------------------------------------------------------------------------- task
def test_task():
    e, v = make_problem(random.Random(0))
    assert eval(e) == v
    tr, ev = problems(2000, "train"), problems(500, "eval")
    assert not {x for x, _ in tr} & {x for x, _ in ev}
    assert problems(10, "eval") == ev[:10]
    assert parse("so <answer> 1,234 </answer>") == 1234
    assert parse("<answer>-5</answer> then <answer>7</answer>") == 7
    assert parse("the answer is 7") is None
    assert reward("<answer>42</answer>", 42) == 1.0 and reward("<answer>41</answer>", 42) == 0
