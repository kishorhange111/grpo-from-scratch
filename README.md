# grpo-from-scratch

[![CI](https://github.com/kishorhange111/grpo-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/grpo-from-scratch/actions/workflows/ci.yml)

Reinforcement learning with verifiable rewards, from scratch. **GRPO** (DeepSeekMath) and
**Dr. GRPO** (Liu et al. 2025) fine-tune Qwen2.5-0.5B-Instruct through a hand-written LoRA
on multi-step arithmetic. The reward is exact match, with no reward model and no value
network. The trainer checkpoints adapters, optimiser and RNG, so a run continues across
GPU sessions that are reclaimed after about 40 minutes.

It is the pretrained-LLM counterpart of
[rlvr-grpo-from-scratch](https://github.com/kishorhange111/rlvr-grpo-from-scratch), which
compares GRPO, Dr. GRPO, RLOO and online DPO across seeds on a tiny character-level GPT on
a CPU. Here the same objectives act on a real instruction-tuned model, through adapters,
with batched generation on a GPU.

## Objective ([`rl.py`](src/grpo/rl.py))

Each problem gets G sampled completions. A completion's advantage is its reward relative
to its group:

| | advantage | token aggregation | KL to reference |
|---|---|---|---|
| GRPO | (r − mean) / std | mean over each sequence, then over sequences | β = 0.04 (k3) |
| Dr. GRPO | r − mean | sum over all tokens / (batch × max length) | none |

The per-token loss is PPO's clipped surrogate, taken over μ = 2 gradient steps per batch so
that the ratio and clipping actually act. Dr. GRPO's argument is twofold. Per-sequence
averaging means each token of a long wrong answer is penalised less. Dividing by std
up-weights prompts that are nearly always solved, or nearly never. Both effects are tested
directly.

## LoRA ([`lora.py`](src/grpo/lora.py))

LoRA adds `W x + (α/r)·B A x` with B = 0 at the start, applied to q/k/v/o, with r = 16. The
starting policy is therefore exactly the base model, and the **reference policy for the KL
term is the same network with adapters switched off**. No second copy of the model is
kept in memory.

## Tests

There are 10, on CPU, using a tiny random Qwen2. They check:
- at ratio 1 the gradient is exactly REINFORCE with the group baseline, for both
  aggregations;
- clipping zeroes the gradient on the right side of the trust region only;
- the k3 KL estimator is non-negative and unbiased;
- GRPO weights the tokens of a short wrong answer 10× more than a 10× longer one, while
  Dr. GRPO weights them equally;
- the LoRA starts as an identity and merges correctly; with adapters disabled it recovers
  the reference exactly; only A and B train;
- **left-padded log-probs match unpadded ones**, i.e. position ids agree with `generate`;
- the task's train and eval sets are disjoint, and the answer parser works.

## Experiment ([`train.py`](src/grpo/train.py))

The task ([`task.py`](src/grpo/task.py)) uses problems like `(47 + 18) * 6 - 39 = ?`, and
the answer must appear in `<answer></answer>`. Format is logged but never rewarded. Each
step draws 8 problems × 8 samples at T = 1 (no top-k, top-p or repetition penalty, so the
sampler *is* the policy). Greedy accuracy on 500 held-out problems is measured every 50
steps. The script logs reward, response length (correct vs wrong), clip fraction and
gradient norm per step.

```
pytest -q                                              # 10 tests, CPU
python -m grpo.train --run grpo --out results          # GPU, resumable
python -m grpo.train --run dr   --out results
```

**Results are pending** from the Colab runs.
