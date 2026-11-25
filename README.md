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
the answer must appear in `<answer></answer>`. There is no separate format reward, but an
untagged answer scores 0, so the format is learned through the correctness reward. Each
step draws 8 problems × 8 samples at T = 1 (no top-k, top-p or repetition penalty, so the
sampler *is* the policy). Greedy accuracy on 500 held-out problems is measured every 50
steps. The script logs reward, response length (correct vs wrong), clip fraction and
gradient norm per step.

```
pytest -q                                              # 10 tests, CPU
python -m grpo.train --run grpo --out results          # GPU, resumable
python -m grpo.train --run dr   --out results
```

## Results (Colab T4, one seed per run, one 34-minute session each)

The per-step logs are in [`results/grpo/history.jsonl`](results/grpo/history.jsonl) and
[`results/dr/history.jsonl`](results/dr/history.jsonl). Each run got one GPU session, so
GRPO reached step 92 and Dr. GRPO, which is cheaper per step, reached step 111. Neither
reached the planned 200.

Greedy accuracy on the 500 held-out problems:

| step | GRPO | Dr. GRPO |
|---|---|---|
| 0 (base model) | 0.0% | 0.0% |
| 50 | **84.8%** | **83.6%** |
| 100 | – | 84.8% |

Training reward (mean over 64 samples per step, T = 1), averaged over windows of steps:

| steps | GRPO | Dr. GRPO |
|---|---|---|
| 0–9 | 0.009 | 0.141 |
| 10–19 | 0.350 | 0.623 |
| 20–29 | 0.658 | 0.723 |
| 30–49 | 0.723 | 0.789 |
| 50–69 | 0.723 | 0.787 |
| 70–91 | 0.748 | 0.791 |

- **The base model scores 0%** because greedy decoding never emits the `<answer>` tag
  within 200 tokens (format 0.0). Almost all of the jump to about 85% is therefore the
  model learning to finish with a tagged answer, which the correctness reward teaches
  implicitly. Format reaches 100% by step 50 in both runs. The gain is not mostly new
  arithmetic ability.
- **At step 50 the two objectives are tied** on held-out accuracy (84.8% against 83.6%,
  within the ±1.6-point standard error for 500 problems). Dr. GRPO climbs faster in the
  first 20 steps of training reward. With one seed each, this does not establish that it
  learns faster in general.
- **Response length doesn't grow.** The mean per step stays between 99 and 147 tokens in
  both runs (110–127 averaged over the windows above), and correct and wrong answers have
  similar lengths. The length bias that Dr. GRPO removes did
  not show up as runaway length on this short task.
- **Clipping almost never fires.** The clip fraction is at most 0.5% on any step, so with
  μ = 2 and lr = 5e-5 the update is effectively REINFORCE with a group baseline.
- **Dr. GRPO is 19% cheaper per step** (16.8 s against 20.8 s), because dropping the KL
  term removes the reference forward pass.
