# Results

Model: decoder-only Transformer, 2 layers, d_model 128, 4 heads,
397,696 parameters. Task: 2-digit addition and subtraction, answers written least-significant digit first.
Accuracy is exact match of the greedy answer on 1,000 new examples. Training ran data-parallel on 8 devices
(CPU devices simulated with `XLA_FLAGS=--xla_force_host_platform_device_count=8`), 1500 steps per phase, batch 128.

## 1. Learning addition (data-parallel training)

| | value |
|---|---|
| final training loss | 0.0361 |
| addition accuracy | 95.1% |
| subtraction accuracy (never seen) | 1.5% |
| time on CPU | 69.3 s |

## 2. Continual learning: subtraction arrives on the stream

| strategy | addition (old task) | subtraction (new task) |
|---|---|---|
| after task A only | 95.1% | 1.5% |
| naive fine-tuning on subtraction | 1.1% | 100.0% |
| experience replay (25% of each batch from addition) | 88.8% | 99.9% |

## 3. XLA

The compiled data-parallel train step contains **2 all-reduce** operation(s): XLA added the
gradient averaging across devices from the sharding annotations alone. The StableHLO (MLIR) the step lowers to is in
`train_step.stablehlo.mlir` (1,865 lines).
