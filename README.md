# tinyformer: a Transformer in JAX, trained data-parallel, with a continual learning experiment

![CI](https://github.com/Snehachowdary56/jax-transformer-lab/actions/workflows/ci.yml/badge.svg)

A small **decoder-only (autoregressive) Transformer** written from scratch in **JAX** (no model library), trained with
**data-parallel sharding** across devices through `jax.sharding`, and used to study **continual learning on a stream of tasks**.
Everything runs on a laptop CPU; the same code runs on GPUs or TPUs.

## What's inside

| File | What it does |
|---|---|
| `tinyformer/model.py` | Pre-LayerNorm Transformer: causal multi-head self-attention, GELU MLP, learned positions, tied output embedding. Parameters are a plain JAX pytree. |
| `tinyformer/train.py` | `jit`-compiled train step (AdamW, warmup + cosine schedule, gradient clipping via Optax). With a device mesh, the batch is sharded across the `data` axis and parameters are replicated; XLA inserts the gradient all-reduce. Greedy decoding with `lax.fori_loop`. |
| `tinyformer/data.py` | Character-level arithmetic tasks with exact answers (answers written least-significant digit first) and a loss mask over answer tokens only. |
| `scripts/run_experiments.py` | Runs the experiments below and writes `results/`. |

## Results (`results/RESULTS.md`)

Model: 2 layers, d_model 128, 4 heads, 397,696 parameters. Trained data-parallel on 8 devices (CPU devices simulated with
`XLA_FLAGS=--xla_force_host_platform_device_count=8`), 1,500 steps per phase, batch 128. Accuracy = exact match of greedy answers on 1,000 new problems.

**1. Learning 2-digit addition:** 95.1% exact-match accuracy in 69 seconds on CPU.

**2. Continual learning: subtraction arrives on the stream**

| strategy | addition (old task) | subtraction (new task) |
|---|---|---|
| after addition only | 95.1% | 1.5% |
| naive fine-tuning on subtraction | **1.1%** (catastrophic forgetting) | 100.0% |
| experience replay (25% of each batch from addition) | **88.8%** | 99.9% |

**3. XLA:** the compiled data-parallel step contains the gradient **all-reduce** that XLA added from the sharding annotations;
the step's StableHLO (MLIR) is saved in `results/train_step.stablehlo.mlir`.

## Tests (8, in CI on 8 simulated devices)

- The data-parallel step gives the **same loss and parameters** as the single-device step (to 1e-5).
- The compiled sharded step contains an all-reduce; the single-device step does not.
- The **causal mask** works: changing a future token never changes earlier logits.
- Parameter count matches the architecture formula; training lowers the loss; data and loss masks are correct.

## Run it

```bash
pip install -e ".[dev]" "jax[cpu]"
export XLA_FLAGS=--xla_force_host_platform_device_count=8   # 8 CPU devices for data parallelism
pytest
python scripts/run_experiments.py                           # about 4 minutes on a laptop CPU
```

## Notes and limits
- This is a learning-scale model. On real hardware the same `jax.sharding` code would use GPUs or TPUs, and model
  (tensor) parallelism would add a second mesh axis for the weights.
- Experience replay keeps a 25% share of old-task examples; the 88.8% vs 95.1% gap shows replay reduces but does not remove forgetting.
