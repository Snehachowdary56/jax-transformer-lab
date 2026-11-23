import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinyformer import data
from tinyformer.model import Config, count_params, forward, init_params, loss_fn
from tinyformer.train import _greedy, data_parallel_mesh, make_optimizer, make_train_step, shard_batch, train

CFG = Config(d_model=64, n_heads=4, n_layers=2, d_ff=128)


def test_examples_are_correct_and_reversed():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p, a = data.make_example("add", rng, 2)
        x, y = map(int, p[:-1].split("+"))
        assert a == str(x + y)[::-1] + "."
        p, a = data.make_example("sub", rng, 2)
        x, y = map(int, p[:-1].split("-"))
        assert x >= y and a == str(x - y)[::-1] + "."
    assert data.decode(data.encode("12+7=91.")) == "12+7=91."


def test_loss_mask_covers_only_the_answer():
    tokens, mask = data.batch("add", 4, np.random.default_rng(1), 2)
    for row, m in zip(tokens, mask, strict=False):
        s = data.decode(row).rstrip("#")
        prompt_len = s.index("=") + 1
        predicted = [s[i + 1] for i in np.flatnonzero(m)]  # mask[t] scores the prediction of token t+1
        assert "".join(predicted) == s[prompt_len:]


def test_causal_mask_future_tokens_do_not_change_earlier_logits():
    params = init_params(CFG, jax.random.PRNGKey(0))
    a = jnp.array([[1, 2, 10, 3, 12, 4, 5, 13, 14, 14]])
    b = a.at[0, 6:].set(7)
    la, lb = forward(params, a, CFG), forward(params, b, CFG)
    np.testing.assert_allclose(la[0, :6], lb[0, :6], atol=1e-6)
    assert not np.allclose(la[0, 6:], lb[0, 6:])


def test_shapes_and_parameter_count():
    params = init_params(CFG, jax.random.PRNGKey(0))
    tokens, _ = data.batch("add", 3, np.random.default_rng(0), 2)
    assert forward(params, jnp.asarray(tokens), CFG).shape == (3, CFG.seq_len, CFG.vocab)
    d, f, L = CFG.d_model, CFG.d_ff, CFG.n_layers
    per_block = 4 * d + 3 * d * d + d * d + d * f + f * d
    assert count_params(params) == CFG.vocab * d + CFG.seq_len * d + 2 * d + L * per_block


def test_data_parallel_step_matches_single_device_step():
    assert len(jax.devices()) == 8
    params = init_params(CFG, jax.random.PRNGKey(1))
    opt = make_optimizer(1e-3, 10)
    state = opt.init(params)
    tokens, mask = data.batch("add", 64, np.random.default_rng(2), 2)
    p1, _, l1 = make_train_step(CFG, opt)(params, state, jnp.asarray(tokens), jnp.asarray(mask))
    mesh = data_parallel_mesh(8)
    st, sm = shard_batch(tokens, mask, mesh)
    assert len(st.addressable_shards) == 8 and st.addressable_shards[0].data.shape == (8, CFG.seq_len)
    p8, _, l8 = make_train_step(CFG, opt, mesh)(params, state, st, sm)
    np.testing.assert_allclose(l1, l8, rtol=1e-5)
    for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p8), strict=False):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_xla_inserts_all_reduce_for_gradients():
    opt = make_optimizer(1e-3, 10)
    params = init_params(CFG, jax.random.PRNGKey(0))
    tok = jnp.zeros((32, CFG.seq_len), jnp.int32)
    msk = jnp.zeros((32, CFG.seq_len - 1), jnp.float32)
    hlo = make_train_step(CFG, opt, data_parallel_mesh(8)).lower(params, opt.init(params), tok, msk).compile().as_text()
    assert re.search(r"\ball-reduce(-start)?\(", hlo)
    single = make_train_step(CFG, opt).lower(params, opt.init(params), tok, msk).compile().as_text()
    assert not re.search(r"\ball-reduce(-start)?\(", single)


def test_training_reduces_the_loss_and_decoding_fills_the_answer():
    params = init_params(CFG, jax.random.PRNGKey(0))
    tokens, mask = data.batch("add", 256, np.random.default_rng(9), 2)
    before = float(loss_fn(params, jnp.asarray(tokens), jnp.asarray(mask), CFG))
    params, losses = train(CFG, params, ["add"], 150, 64, seed=3, mesh=data_parallel_mesh(8), lr=3e-3)
    after = float(loss_fn(params, jnp.asarray(tokens), jnp.asarray(mask), CFG))
    assert after < 0.7 * before
    prompts = jnp.array([data.encode("12+34=")])
    out = _greedy(params, prompts, CFG, CFG.seq_len - 6)
    assert out.shape == (1, CFG.seq_len) and all(0 <= int(t) < CFG.vocab for t in out[0])


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        data.make_example("mul", np.random.default_rng(0))
