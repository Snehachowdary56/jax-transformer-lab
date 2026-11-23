"""Runs the experiments and writes results/RESULTS.md and results/results.json.

  XLA_FLAGS=--xla_force_host_platform_device_count=8 python scripts/run_experiments.py

1. Data-parallel training on 8 devices (simulated on CPU): learn 2-digit addition, measure exact-match accuracy.
2. Continual learning on a stream of tasks (addition, then subtraction): naive fine-tuning vs experience replay.
3. XLA: the compiled data-parallel step contains the all-reduce XLA inserted for the gradients; save its StableHLO (MLIR).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tinyformer.model import Config, count_params, init_params  # noqa: E402
from tinyformer.train import accuracy, data_parallel_mesh, make_optimizer, make_train_step, train  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
STEPS = int(os.environ.get("STEPS", 1500))
BATCH = 128


def main():
    OUT.mkdir(exist_ok=True)
    cfg = Config()
    mesh = data_parallel_mesh()
    n_dev = mesh.devices.size
    print(f"devices: {n_dev} ({jax.devices()[0].platform}), params: {count_params(init_params(cfg, jax.random.PRNGKey(0))):,}")
    res = {"devices": n_dev, "steps_per_phase": STEPS, "batch": BATCH, "config": cfg.__dict__ | {"seq_len": cfg.seq_len}}

    # 1. Task A (addition), data-parallel
    t0 = time.time()
    p_add, losses_a = train(
        cfg, init_params(cfg, jax.random.PRNGKey(0)), ["add"], STEPS, BATCH, seed=1, mesh=mesh, lr=3e-3, log_every=500
    )
    res["phase_a"] = {
        "seconds": round(time.time() - t0, 1),
        "final_loss": round(sum(losses_a[-50:]) / 50, 4),
        "acc_add": accuracy(p_add, cfg, "add"),
        "acc_sub": accuracy(p_add, cfg, "sub"),
    }
    print("after addition:", res["phase_a"])

    # 2. Task B (subtraction) arrives on the stream: naive fine-tuning vs replaying 25% addition examples
    p_naive, _ = train(cfg, p_add, ["sub"], STEPS, BATCH, seed=2, mesh=mesh, lr=1e-3)
    res["naive"] = {"acc_add": accuracy(p_naive, cfg, "add"), "acc_sub": accuracy(p_naive, cfg, "sub")}
    print("naive fine-tuning:", res["naive"])
    p_replay, _ = train(cfg, p_add, ["sub"], STEPS, BATCH, seed=2, mesh=mesh, lr=1e-3, replay=["add"], replay_frac=0.25)
    res["replay"] = {"acc_add": accuracy(p_replay, cfg, "add"), "acc_sub": accuracy(p_replay, cfg, "sub")}
    print("with replay:", res["replay"])

    # 3. What XLA compiles: all-reduce in the data-parallel step, and the StableHLO (MLIR) it starts from
    opt = make_optimizer(3e-3, STEPS)
    params = init_params(cfg, jax.random.PRNGKey(0))
    state = opt.init(params)
    tok = jnp.zeros((BATCH, cfg.seq_len), jnp.int32)
    msk = jnp.zeros((BATCH, cfg.seq_len - 1), jnp.float32)
    lowered = make_train_step(cfg, opt, mesh).lower(params, state, tok, msk)
    hlo = lowered.compile().as_text()
    res["xla"] = {
        "all_reduce_ops": len(re.findall(r"\ball-reduce(?:-start)?\(", hlo)),
        "stablehlo_lines": len(lowered.as_text().splitlines()),
    }
    (OUT / "train_step.stablehlo.mlir").write_text(lowered.as_text())
    print("xla:", res["xla"])

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    md = f"""# Results

Model: decoder-only Transformer, {cfg.n_layers} layers, d_model {cfg.d_model}, {cfg.n_heads} heads,
{count_params(params):,} parameters. Task: 2-digit addition and subtraction, answers written least-significant digit first.
Accuracy is exact match of the greedy answer on 1,000 new examples. Training ran data-parallel on {n_dev} devices
(CPU devices simulated with `XLA_FLAGS=--xla_force_host_platform_device_count={n_dev}`), {STEPS} steps per phase, batch {BATCH}.

## 1. Learning addition (data-parallel training)

| | value |
|---|---|
| final training loss | {res["phase_a"]["final_loss"]} |
| addition accuracy | {pct(res["phase_a"]["acc_add"])} |
| subtraction accuracy (never seen) | {pct(res["phase_a"]["acc_sub"])} |
| time on CPU | {res["phase_a"]["seconds"]} s |

## 2. Continual learning: subtraction arrives on the stream

| strategy | addition (old task) | subtraction (new task) |
|---|---|---|
| after task A only | {pct(res["phase_a"]["acc_add"])} | {pct(res["phase_a"]["acc_sub"])} |
| naive fine-tuning on subtraction | {pct(res["naive"]["acc_add"])} | {pct(res["naive"]["acc_sub"])} |
| experience replay (25% of each batch from addition) | {pct(res["replay"]["acc_add"])} | {pct(res["replay"]["acc_sub"])} |

## 3. XLA

The compiled data-parallel train step contains **{res["xla"]["all_reduce_ops"]} all-reduce** operation(s): XLA added the
gradient averaging across devices from the sharding annotations alone. The StableHLO (MLIR) the step lowers to is in
`train_step.stablehlo.mlir` ({res["xla"]["stablehlo_lines"]:,} lines).
"""
    (OUT / "RESULTS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
