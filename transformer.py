import math
import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def __call__(self, x: mx.array, mask=None, cache=None):
        b, t, c = x.shape

        q, k, v = self.c_attn(x).split(3, axis=2)
        k = k.reshape(b, t, self.n_head, c // self.n_head).transpose(0, 2, 1, 3)
        q = q.reshape(b, t, self.n_head, c // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_head, c // self.n_head).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(k.shape[-1]))

        if mask is not None:
            att = att + mask

        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, t, c)

        out = self.resid_dropout(self.c_proj(out))
        return out, (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x, mask=None, cache=None):
        norm = self.ln_1(x)
        att, cache = self.attn(norm, mask=mask, cache=cache)
        x = x + att
        norm = self.ln_2(x)
        mlp = self.mlp(norm)
        x = x + mlp
        return x, cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, affine=config.bias)

    def _forward_transformer_blocks(
        self, x: mx.array, pos: mx.array, mask=None, cache=None, build_cache=False
    ):
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        kv_cache = []

        if cache is not None:
            # When using cache, make sure we have the right cache length
            if len(cache) != len(self.h):
                raise ValueError(f"Cache length {len(cache)} doesn't match model layers {len(self.h)}")
                
            # For cached KV, we don't need the mask for subsequent tokens
            for i in range(len(cache)):
                x, cache[i] = self.h[i](x, mask=None, cache=cache[i])
        else:
            # Initial forward pass
            for block in self.h:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        # Return the right cache depending on whether we're building or using it
        return x, kv_cache if build_cache else cache

    def _create_causal_mask(self, length: int):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
        return mask.astype(self.wte.weight.dtype)

    def _sample_next_token(self, x, temperature):
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1 / temperature))
        return y

    def generate(self, x: mx.array, max_new_tokens=256, temperature=0.8):
        _, t = x.shape
        pos = mx.arange(0, t, 1, dtype=x.dtype)
        mask = self._create_causal_mask(t)
        x, cache = self._forward_transformer_blocks(x, pos, mask=mask, build_cache=True)
        y = self._sample_next_token(x, temperature)
        position = t
        yield y

        for _ in range(max_new_tokens):
            position += 1
            x = y[:, None]
            x, cache = self._forward_transformer_blocks(x, position, cache=cache)
            y = self._sample_next_token(x, temperature)
            yield y

    def __call__(self, x: mx.array, targets: mx.array = None):
        b, t = x.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=x.dtype)

        mask = self._create_causal_mask(t)
        x, _ = self._forward_transformer_blocks(x, pos, mask=mask)

        return x @ self.wte.weight.T

    def loss(self, x, y):
        logits = self(x)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        # mx.simplify was removed in newer versions of MLX
        return mx.mean(loss)
