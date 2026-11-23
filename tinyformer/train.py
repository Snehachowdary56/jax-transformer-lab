"""Training: a jit-compiled step that runs data-parallel across devices with jax.sharding.

The batch is split across a 1-D device mesh ("data" axis) and the parameters are replicated. XLA inserts the
all-reduce of gradients, so the same code runs on 1 CPU, 8 simulated CPU devices, or a GPU/TPU slice.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import data
from .model import Config, forward, loss_fn


def make_optimizer(lr: float = 1e-3, steps: int = 3000):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps=max(steps // 20, 1), decay_steps=steps, end_value=lr / 10)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))


def data_parallel_mesh(n_devices: int | None = None) -> Mesh:
    devices = np.array(jax.devices()[: n_devices or len(jax.devices())])
    return Mesh(devices, ("data",))


def make_train_step(cfg: Config, opt, mesh: Mesh | None = None):
    """Returns step(params, opt_state, tokens, mask) -> (params, opt_state, loss)."""

    def step(params, opt_state, tokens, mask):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, mask, cfg)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    if mesh is None:
        return jax.jit(step)
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P("data"))  # split the batch dimension across devices
    return jax.jit(
        step, in_shardings=(replicated, replicated, sharded, sharded), out_shardings=(replicated, replicated, replicated)
    )


def shard_batch(tokens, mask, mesh: Mesh | None):
    if mesh is None:
        return jnp.asarray(tokens), jnp.asarray(mask)
    s = NamedSharding(mesh, P("data"))
    return jax.device_put(tokens, s), jax.device_put(mask, s)


@partial(jax.jit, static_argnums=(2, 3))
def _greedy(params, prompts, cfg: Config, n_new: int):
    """Greedy decoding of n_new tokens after fixed-length prompts."""
    B, Tp = prompts.shape
    buf = jnp.full((B, cfg.seq_len), data.PAD, dtype=jnp.int32).at[:, :Tp].set(prompts)

    def body(i, buf):
        logits = forward(params, buf, cfg)
        nxt = jnp.argmax(logits[:, Tp - 1 + i], axis=-1)
        return buf.at[:, Tp + i].set(nxt)

    return jax.lax.fori_loop(0, n_new, body, buf)


def accuracy(params, cfg: Config, task: str, n: int = 1000, seed: int = 123) -> float:
    """Exact-match accuracy of greedy answers on n fresh examples (prompts are grouped by length)."""
    rng = np.random.default_rng(seed)
    examples = [data.make_example(task, rng, cfg.max_digits) for _ in range(n)]
    correct = 0
    by_len: dict[int, list[tuple[str, str]]] = {}
    for p, a in examples:
        by_len.setdefault(len(p), []).append((p, a))
    for plen, group in by_len.items():
        prompts = jnp.array([data.encode(p) for p, _ in group], dtype=jnp.int32)
        out = np.asarray(_greedy(params, prompts, cfg, cfg.seq_len - plen))
        for row, (_, a) in zip(out, group, strict=False):
            pred = data.decode(row[plen:])
            correct += pred.split(".")[0] + "." == a
    return correct / n


def train(
    cfg: Config,
    params,
    tasks: list[str],
    steps: int,
    batch_size: int,
    seed: int,
    mesh: Mesh | None = None,
    lr: float = 1e-3,
    replay: list[str] | None = None,
    replay_frac: float = 0.0,
    log_every: int = 0,
):
    """Trains on a stream of fresh batches. With replay, a fraction of each batch comes from earlier tasks."""
    opt = make_optimizer(lr, steps)
    opt_state = opt.init(params)
    step_fn = make_train_step(cfg, opt, mesh)
    rng = np.random.default_rng(seed)
    losses = []
    for i in range(steps):
        n_replay = int(batch_size * replay_frac) if replay else 0
        tok, msk = data.mixed_batch(tasks, batch_size - n_replay, rng, cfg.max_digits)
        if n_replay:
            rt, rm = data.mixed_batch(replay, n_replay, rng, cfg.max_digits)
            tok, msk = np.concatenate([tok, rt]), np.concatenate([msk, rm])
        tok, msk = shard_batch(tok, msk, mesh)
        params, opt_state, loss = step_fn(params, opt_state, tok, msk)
        losses.append(float(loss))
        if log_every and (i + 1) % log_every == 0:
            print(f"  step {i + 1:5d}  loss {np.mean(losses[-log_every:]):.4f}", flush=True)
    return params, losses
