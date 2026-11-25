"""A verifiable task: multi-step integer arithmetic with an exact-match reward.

Problems are like ``(47 + 18) * 6 - 39``, generated from a seed, so the train and eval
sets are reproducible and disjoint. The model must end its reply with
``<answer>N</answer>``. Reward is 1 when N equals the true value and 0 otherwise. There is
no separate format reward, but an answer outside the tags scores 0, so the format is
rewarded implicitly.
"""

from __future__ import annotations

import random
import re

SYSTEM = (
    "Solve the arithmetic problem. Think step by step briefly, then give the final "
    "integer inside <answer></answer>."
)
_ANS = re.compile(r"<answer>\s*(-?\d[\d,]*)\s*</answer>")


def make_problem(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    c, d = rng.randint(3, 19), rng.randint(10, 199)
    form = rng.randrange(4)
    if form == 0:
        e = f"({a} + {b}) * {c} - {d}"
    elif form == 1:
        e = f"{a} * {c} + {b} * {rng.randint(3, 19)} - {d}"
    elif form == 2:
        e = f"({a} - {b}) * {c} + {d}"
    else:
        e = f"{a} * {b} - {c} * {d}"
    return e, eval(e)  # noqa: S307 - expression built above from integers only


def problems(n: int, split: str) -> list[tuple[str, int]]:
    rng = random.Random({"train": 0, "eval": 1}[split])
    seen, out = set(), []
    held = set() if split == "eval" else {e for e, _ in problems(500, "eval")}
    while len(out) < n:
        e, v = make_problem(rng)
        if e not in seen and e not in held:
            seen.add(e)
            out.append((e, v))
    return out


def prompt_messages(expr: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{expr} = ?"}]


def parse(text: str) -> int | None:
    m = _ANS.findall(text)
    return int(m[-1].replace(",", "")) if m else None


def reward(text: str, truth: int) -> float:
    return 1.0 if parse(text) == truth else 0.0
