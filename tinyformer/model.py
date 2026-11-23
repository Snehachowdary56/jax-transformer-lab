"""Decoder-only (autoregressive) Transformer written directly in JAX: parameters are a plain pytree.

Pre-LayerNorm blocks: x + attn(ln(x)), x + mlp(ln(x)); causal multi-head self-attention; learned positions;
the output projection is tied to the token embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab: int = 15
    max_digits: int = 2  # operands have up to this many digits
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512

    @property
    def seq_len(self) -> int:
        return 2 * self.max_digits + 2 + self.max_digits + 2


def init_params(cfg: Config, key: jax.Array) -> dict:
    keys = iter(jax.random.split(key, 4 + 4 * cfg.n_layers))
    normal = lambda shape, std=0.02: std * jax.random.normal(next(keys), shape)  # noqa: E731
    params = {
        "tok": normal((cfg.vocab, cfg.d_model)),
        "pos": normal((cfg.seq_len, cfg.d_model)),
        "ln_f": {"g": jnp.ones(cfg.d_model), "b": jnp.zeros(cfg.d_model)},
        "blocks": [],
    }
    proj_std = 0.02 / jnp.sqrt(2 * cfg.n_layers)  # GPT-2 style scaling for residual projections
    for _ in range(cfg.n_layers):
        params["blocks"].append(
            {
                "ln1": {"g": jnp.ones(cfg.d_model), "b": jnp.zeros(cfg.d_model)},
                "qkv": normal((cfg.d_model, 3 * cfg.d_model)),
                "proj": normal((cfg.d_model, cfg.d_model), proj_std),
                "ln2": {"g": jnp.ones(cfg.d_model), "b": jnp.zeros(cfg.d_model)},
                "fc": normal((cfg.d_model, cfg.d_ff)),
                "out": normal((cfg.d_ff, cfg.d_model), proj_std),
            }
        )
    return params


def layer_norm(x, p, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * p["g"] + p["b"]


def attention(x, blk, n_heads):
    B, T, D = x.shape
    q, k, v = jnp.split(x @ blk["qkv"], 3, axis=-1)
    split = lambda t: t.reshape(B, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)  # noqa: E731
    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(D // n_heads)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))  # a position can only attend to itself and earlier ones
    scores = jnp.where(causal, scores, -1e9)
    out = jax.nn.softmax(scores, axis=-1) @ v
    return out.transpose(0, 2, 1, 3).reshape(B, T, D) @ blk["proj"]


def forward(params, tokens, cfg: Config):
    """tokens [B, T] -> logits [B, T, vocab]"""
    T = tokens.shape[1]
    x = params["tok"][tokens] + params["pos"][:T]
    for blk in params["blocks"]:
        x = x + attention(layer_norm(x, blk["ln1"]), blk, cfg.n_heads)
        x = x + jax.nn.gelu(layer_norm(x, blk["ln2"]) @ blk["fc"]) @ blk["out"]
    return layer_norm(x, params["ln_f"]) @ params["tok"].T


def loss_fn(params, tokens, mask, cfg: Config):
    """Next-token cross-entropy, averaged over the answer positions only."""
    logits = forward(params, tokens[:, :-1], cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
