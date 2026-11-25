"""GRPO / Dr. GRPO fine-tuning of Qwen2.5-0.5B-Instruct with LoRA on arithmetic.

    python -m grpo.train --run grpo --out results --steps 200
    python -m grpo.train --run dr   --out results --steps 200

Each step samples ``P`` problems × ``G`` completions at T = 1, scores them, and takes
``mu`` gradient steps on that batch (so the PPO ratio and clipping matter). Accuracy on
500 held-out problems (greedy) is measured at step 0 and every ``--eval-every`` steps.
Adapters, optimiser and RNG state are checkpointed every few steps, so a reclaimed GPU
session resumes where it stopped. Everything is appended to ``<out>/<run>/history.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from . import rl
from .lora import disabled, inject, lora_state
from .task import parse, problems, prompt_messages, reward

RUNS = {
    "grpo": dict(scale=True, agg="grpo", beta=0.04),
    "dr": dict(scale=False, agg="dr", beta=0.0),
}


def encode(tok, exprs, device):
    texts = [
        tok.apply_chat_template(prompt_messages(e), tokenize=False, add_generation_prompt=True)
        for e in exprs
    ]
    return tok(texts, return_tensors="pt", padding=True).to(device)


def positions(attn):
    return (attn.cumsum(-1) - 1).clamp(min=0)


def seq_logprobs(model, ids, attn, n_prompt, micro=8):
    """log π of every completion token (b, t_completion), using the same position ids as
    ``generate`` does for left-padded prompts."""
    out = []
    for i in range(0, len(ids), micro):
        x, a = ids[i : i + micro], attn[i : i + micro]
        logits = model(input_ids=x, attention_mask=a, position_ids=positions(a)).logits
        out.append(rl.token_logprobs(logits[:, n_prompt - 1 : -1], x[:, n_prompt:]))
    return torch.cat(out)


def completion_mask(comp, eos_ids):
    """1 up to and including the first EOS, 0 after."""
    is_eos = torch.zeros_like(comp, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= comp == e
    after = (is_eos.cumsum(1) - is_eos.long()) > 0
    return (~after).long()


@torch.no_grad()
def evaluate(model, tok, data, device, max_new, bs=100):
    correct = formatted = length = 0
    for i in range(0, len(data), bs):
        chunk = data[i : i + bs]
        enc = encode(tok, [e for e, _ in chunk], device)
        g = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=1.0,
            pad_token_id=tok.pad_token_id,
        )
        comp = g[:, enc.input_ids.shape[1] :]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        for t, (_, v), c in zip(texts, chunk, comp, strict=True):
            correct += reward(t, v)
            formatted += parse(t) is not None
            length += int((c != tok.pad_token_id).sum())
    n = len(data)
    return {"acc": correct / n, "format": formatted / n, "len": length / n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS), default="grpo")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--P", type=int, default=8, help="prompts per step")
    ap.add_argument("--G", type=int, default=8, help="completions per prompt")
    ap.add_argument("--mu", type=int, default=2, help="gradient steps per batch")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--budget-min", type=float, default=1e9, help="stop after this many min")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    cfg = RUNS[a.run]
    out = os.path.join(a.out, a.run)
    os.makedirs(out, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(dev)
    model.config.use_cache = True
    inject(model, r=16, alpha=32)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    train, evals = problems(20000, "train"), problems(a.n_eval, "eval")
    ck, hist = os.path.join(out, "ckpt.pt"), os.path.join(out, "history.jsonl")
    step = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["lora"], strict=False)
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        step = s["step"]
        torch.set_rng_state(s["rng"])
        torch.cuda.set_rng_state(s["cuda_rng"])
        log(f"resumed at step {step}")

    def save():
        torch.save(
            {
                "lora": lora_state(model),
                "opt": opt.state_dict(),
                "step": step,
                "scaler": scaler.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            ck + ".tmp",
        )
        os.replace(ck + ".tmp", ck)

    def write(rec):
        with open(hist, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(json.dumps(rec))

    def maybe_eval():
        model.eval()
        r = evaluate(model, tok, evals, dev, a.max_new)
        write({"step": step, "eval": r})
        model.train()

    have_eval = set()
    if os.path.exists(hist):
        have_eval = {json.loads(s)["step"] for s in open(hist) if '"eval"' in s}
    if step == 0 and 0 not in have_eval:
        maybe_eval()
    rng = random.Random(1000 + step)
    while step < a.steps:
        if (time.time() - t_start) / 60 > a.budget_min:
            log("time budget reached; checkpoint saved")
            break
        t0 = time.time()
        batch = rng.sample(train, a.P)
        enc = encode(tok, [e for e, _ in batch], dev)
        n_prompt = enc.input_ids.shape[1]
        model.eval()
        with torch.no_grad():
            g = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=a.max_new,
                num_return_sequences=a.G,
                pad_token_id=tok.pad_token_id,
            )
        model.train()
        comp = g[:, n_prompt:]
        cmask = completion_mask(comp, eos)
        attn = torch.cat([enc.attention_mask.repeat_interleave(a.G, 0), cmask], 1)
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        truths = [v for _, v in batch for _ in range(a.G)]
        r = torch.tensor([reward(t, v) for t, v in zip(texts, truths, strict=True)], device=dev)
        adv = rl.advantages(r.view(a.P, a.G), scale=cfg["scale"]).flatten()
        with torch.no_grad():
            lp_old = seq_logprobs(model, g, attn, n_prompt)
            lp_ref = None
            if cfg["beta"]:
                with disabled(model):
                    lp_ref = seq_logprobs(model, g, attn, n_prompt)
        stats = []
        for _ in range(a.mu):
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(g), 8):
                sl = slice(i, i + 8)
                lp = seq_logprobs(model, g[sl], attn[sl], n_prompt)
                ls, st = rl.loss(
                    lp,
                    lp_old[sl],
                    adv[sl],
                    cmask[sl],
                    None if lp_ref is None else lp_ref[sl],
                    cfg["beta"],
                    agg=cfg["agg"],
                    max_len=a.max_new,
                )
                scaler.scale(ls * len(lp) / len(g)).backward()
                stats.append(st["clip_frac"])
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        step += 1
        lens = cmask.sum(1).float()
        rec = {
            "step": step,
            "reward": float(r.mean()),
            "solved_groups": float((r.view(a.P, a.G).max(1).values > 0).float().mean()),
            "len": float(lens.mean()),
            "len_correct": float(lens[r > 0].mean()) if (r > 0).any() else None,
            "len_wrong": float(lens[r == 0].mean()) if (r == 0).any() else None,
            "clip_frac": float(np.mean(stats)),
            "grad_norm": float(gn),
            "sec": time.time() - t0,
        }
        write(rec)
        if step % 5 == 0 or step == a.steps:
            save()
        if step % a.eval_every == 0 or step == a.steps:
            maybe_eval()
            save()


if __name__ == "__main__":
    main()
