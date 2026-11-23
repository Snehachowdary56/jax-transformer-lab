"""Character-level tasks with exact answers, so model quality is measured by exact-match accuracy.

Each example is a prompt and an answer, for example  "123+45=" -> "861." (168 written backwards).
Answers are written least-significant digit first, the order a person adds in: each digit then depends only on
the digits already written and the carry, which makes the task learnable by a small autoregressive model.
The model is trained autoregressively on the whole string; accuracy only counts the answer.
"""

from __future__ import annotations

import numpy as np

VOCAB = list("0123456789+-=.#")  # '#' is padding, '.' ends the answer
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI["#"]
END = STOI["."]


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids)


def make_example(task: str, rng: np.random.Generator, max_digits: int = 3) -> tuple[str, str]:
    a = int(rng.integers(0, 10**max_digits))
    b = int(rng.integers(0, 10**max_digits))
    if task == "add":
        return f"{a}+{b}=", f"{str(a + b)[::-1]}."
    if task == "sub":
        a, b = max(a, b), min(a, b)  # keep answers non-negative
        return f"{a}-{b}=", f"{str(a - b)[::-1]}."
    raise ValueError(f"unknown task {task}")


def seq_len(max_digits: int = 3) -> int:
    # "999+999=1998." is 2*d + 2 + (d+1) + 1 characters
    return 2 * max_digits + 2 + max_digits + 2


def batch(task: str, n: int, rng: np.random.Generator, max_digits: int = 3):
    """Returns tokens [n, T] and a loss mask [n, T-1] that is 1 only on answer positions."""
    T = seq_len(max_digits)
    tokens = np.full((n, T), PAD, dtype=np.int32)
    mask = np.zeros((n, T - 1), dtype=np.float32)
    for i in range(n):
        p, a = make_example(task, rng, max_digits)
        s = encode(p + a)
        tokens[i, : len(s)] = s
        mask[i, len(p) - 1 : len(s) - 1] = 1.0  # predict each answer character from the one before
    return tokens, mask


def mixed_batch(tasks: list[str], n: int, rng: np.random.Generator, max_digits: int = 3):
    parts = [batch(t, n // len(tasks) + (1 if i < n % len(tasks) else 0), rng, max_digits) for i, t in enumerate(tasks)]
    tokens = np.concatenate([p[0] for p in parts])
    mask = np.concatenate([p[1] for p in parts])
    perm = rng.permutation(len(tokens))
    return tokens[perm], mask[perm]
