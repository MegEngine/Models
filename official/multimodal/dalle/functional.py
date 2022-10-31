import math
from typing import Optional, Sequence

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import Tensor

from .attention import expand


def log(t, eps=1e-20):
    return F.log(t + eps)


def gumbel_noise(t):
    noise = F.zeros_like(t)
    M.init.uniform_(noise, 0, 1)
    noise = mge.random.uniform(0, 1, t.shape)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., axis=-1):
    sample = (t / temperature) + gumbel_noise(t)
    return F.argmax(sample, axis=axis)


def topk(logits: Tensor, thread: float = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thread) * num_logits), 1)
    val, ind = F.topk(logits, k, descending=True)
    probs = F.full_like(logits, -float('Inf'))
    probs = F.scatter(probs, axis=1, index=ind, source=val)
    return probs


class AxialPositionalEmbedding(M.Module):
    """
    Axial positional embedding, which is very effective.
    The input must be a 3-d Tensor.

    Args:
        dim(int): The channel of output.
        axial_shape(Sequence[int]): Numbers of each axis,
            the sequence length of input must be less than the product of `axial_shape`
        axial_dim(Optional[Sequence[int]]): Dimensions of each axis,
            the summation of `axial_dim` must equal to input's dimension.
            If not specified, the output will be summed up instead of concatenating. Default: None.

    Return:
        The axial positional embedding with the same device of input.
    """

    def __init__(
        self,
        dim: int,
        axial_shape: Sequence[int],
        axial_dim: Optional[Sequence[int]] = None
    ):
        super(AxialPositionalEmbedding, self).__init__()
        self.dim = dim
        self.shapes = axial_shape
        self.max_seq_len = np.prod(axial_shape)
        self.summed = axial_dim is None
        self.axial_dim = ((dim, ) * len(axial_shape)
                          ) if self.summed else axial_dim

        if not self.summed:
            if len(self.shapes) != len(self.axial_dim):
                raise ValueError(
                    "number of axial dimensions must equal the number of dimensions in the shape")
            if sum(self.axial_dim) != dim:
                raise ValueError(
                    f"axial dimensions must be summed up to the target dimension {dim}, but got {sum(self.axial_dim)}")  # noqa: E501

        self.weights = []
        for idx, (shape, d) in enumerate(zip(self.shapes, self.axial_dim)):
            ax_shape = [1] * len(self.shapes)
            ax_shape[idx] = shape
            self.weights.append(
                mge.Parameter(mge.random.normal(size=(1, *ax_shape, d)))
            )

    def forward(self, inputs):
        B, L, _ = inputs.shape
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length of input tensor must be less than the maximum sequence length {self.max_seq_len}, but got {L}")  # noqa: E501

        embedings = []
        for emb, dim in zip(self.weights, self.axial_dim):
            broadcast_shape = (B, *self.shapes, dim)
            embedings.append(
                F.broadcast_to(emb, broadcast_shape)
                .reshape(B, self.max_seq_len, dim)
            )

        embeding = sum(embedings) if self.summed else F.concat(
            embedings, axis=-1)
        return embeding[:, :L].to(inputs.device)


class RotaryEmbedding(M.Module):
    def __init__(
        self,
        dim: int,
        custom_freqs=None,
        freqs_type: str = 'lang',
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        is_learnable: bool = False,
    ):
        super(RotaryEmbedding, self).__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_type == 'lang':
            freqs = 1. / (theta ** (np.arange(0, dim, 2, dtype=np.float64)
                                    [:(dim // 2)] / dim))
        elif freqs_type == 'pixel':
            freqs = np.linspace(1., max_freq / 2, dim // 2,
                                dtype=np.float64) * math.pi
        elif freqs_type == 'constant':
            freqs = np.ones(num_freqs)
        else:
            raise ValueError(f'Unsupported type {freqs_type}')

        freqs = mge.tensor(freqs, dtype=np.float32)

        self.cache = dict()

        if is_learnable:
            self.freqs = mge.Parameter(freqs)
        else:
            self.freqs = freqs

    def forward(self, x, cache_key=None):
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        freqs = self.freqs
        freqs = expand(x, freqs.shape[0], 1) * expand(freqs, x.shape[0], 0)
        freqs = F.repeat(freqs, 2, axis=freqs.ndim - 1)

        if cache_key is not None:
            self.cache[cache_key] = freqs

        return freqs


class DivideMax(M.Module):
    def __init__(self, axis):
        super(DivideMax, self).__init__()
        self.axis = axis

    def forward(self, x):
        maxes = F.max(x, axis=self.axis, keepdims=True).detach()
        return x / maxes
