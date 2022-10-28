import math
from copy import deepcopy

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .functional import multinomial, top_k_top_p_filtering


class GPTConfig:
    embed_drop = 0.1
    resid_drop = 0.1
    attn_drop = 0.1

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embed: int,
        embed_drop: float = 0.,
        resid_drop: float = 0.,
        attn_drop: float = 0.,
        n_unmasked: int = 0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_drop = embed_drop
        self.resid_drop = resid_drop
        self.attn_drop = attn_drop
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embed = n_embed
        self.n_unmasked = n_unmasked
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embed = 768


class CausalSelfAttention(M.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embed % config.n_head == 0
        self.key = M.Linear(config.n_embed, config.n_embed)
        self.query = M.Linear(config.n_embed, config.n_embed)
        self.value = M.Linear(config.n_embed, config.n_embed)

        self.attn_drop = M.Dropout(config.attn_drop)
        self.resid_drop = M.Dropout(config.resid_drop)

        self.proj = M.Linear(config.n_embed, config.n_embed)

        mask = mge.tensor(
            np.tril(np.ones((config.block_size, config.block_size))))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.mask = mask.reshape(1, 1, config.block_size, config.block_size)
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head
        self.n_embed = config.n_embed

    def forward(self, x, layer_past=None):
        B, T, _ = x.shape

        q, k, v = [m(x).reshape(B, T, self.n_head, self.head_dim).transpose(
            0, 2, 1, 3) for m in (self.query, self.key, self.value)]
        present = F.stack([k, v], axis=0)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = F.concat([past_key, k], axis=-2)
            v = F.concat([past_value, v], axis=-2)

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(k.shape[-1])

        if layer_past is None:
            mask = self.mask[:, :, :T, :T] == 0
            mask = F.repeat(mask, attn.shape[1], axis=1)
            attn = F.where(mask.astype('bool'), F.full_like(
                attn, value=-float('Inf')), attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        out = F.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_embed)
        out = self.resid_drop(self.proj(out))

        return out, present


class Block(M.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln1 = M.LayerNorm(config.n_embed)
        self.ln2 = M.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.mlp = M.Sequential(
            M.Linear(config.n_embed, 4 * config.n_embed),
            M.GELU(),
            M.Linear(4 * config.n_embed, config.n_embed),
            M.Dropout(config.resid_drop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        if return_present:
            assert not self.training

        attn, present = self.attn(self.ln1(x), layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(M.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        # input embedding stem
        self.tok_emb = M.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = mge.Parameter(
            F.zeros(shape=(1, config.block_size, config.n_embed)))
        self.drop = M.Dropout(config.embed_drop)

        # transformer
        self.blocks = M.Sequential(
            *[
                Block(config)
                for _ in range(config.n_layer)
            ]
        )

        # decoder head
        self.ln_f = M.LayerNorm(config.n_embed)
        self.head = M.Linear(config.n_embed, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.config = config

        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (M.Linear, M.Embedding)):
                M.init.normal_(m.weight, mean=0, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    M.init.zeros_(m.bias)
            elif isinstance(m, M.LayerNorm):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

        self.apply(_init_weights)

    def forward(self, idx, embed=None, targets=None):
        # forward the GPT model
        token_embed = self.tok_emb(idx)
        if embed is not None:
            token_embed = F.concat([embed, token_embed], axis=1)

        t = token_embed.shape[1]
        assert t <= self.block_size
        position_embed = self.pos_emb[:, :t, :]
        x = self.drop(token_embed + position_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = (
            None
            if targets is None else
            F.nn.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        )

        return logits, loss

    def forward_with_past(self, idx, embed=None, targets=None, past=None, past_length=None):
        # inference only
        assert not self.training

        token_embed = self.tok_emb(idx)
        if embed is not None:
            token_embed = F.concat([embed, token_embed], axis=1)

        if past is not None:
            assert past_length is not None
            past = F.concat(past, axis=-2)
            past_shape = past.shape
            expected_shape = (self.config.n_layer, 2, idx.shape[0],
                              self.config.n_head, past_length, self.config.n_embed // self.config.n_head)  # noqa: E501
            assert past_shape == expected_shape, f"expected `past_shape` is {expected_shape}, but got {past_shape}"  # noqa: E501
            position_embed = self.pos_emb[:, past_length, :]
        else:
            position_embed = self.pos_emb[:, :token_embed.shape[1], :]

        x = self.drop(token_embed + position_embed)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(
                x, layer_past=None if past is None else past[i, ...], return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = (
            None
            if targets is None else
            F.nn.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        )

        return logits, loss, F.stack(presents)


def top_k_logits(logits, k):
    v, _ = F.topk(logits, k, descending=True)
    out = deepcopy(logits)
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None):
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(
            x, past=past, past_length=(n + cond_len - 1))
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, axis=-1)
        if not sample_logits:
            _, x = F.topk(probs, k=1, descending=True)
        else:
            x = multinomial(probs, num_samples=1)
        # append to the sequence and continue
        sample = F.concat([sample, x], axis=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample
