import random
from typing import Sequence

import megengine as mge
import megengine.functional as F
from megengine import Tensor


def sample_exponential(size: Sequence[int], lambd: float = 1., eps: float = 1e-10):
    """
        Generate random numbers from exponential distribution.
    """
    random_tensor = mge.random.uniform(0, 1, size=size)
    return -(1 / lambd) * F.log(random_tensor + eps)


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.,
    hard: bool = False,
    eps: float = 1e-10,
    axis: int = -1,
) -> Tensor:
    r"""
        Generate gumble noise, G_i = -log(-log(U_i)), U_i \in U(0, 1)
        More details see https://arxiv.org/pdf/1611.00712.pdf
    """
    gumble_noise = -F.log(sample_exponential(logits.shape, eps=eps) + eps)

    gumbels = (logits + gumble_noise) / tau
    y_soft = F.softmax(gumbels, axis=axis)

    if hard:
        index = F.argmax(y_soft, axis=axis, keepdims=True)
        y_hard = F.scatter(F.zeros_like(logits), axis=axis,
                           index=index, source=F.ones(index.shape, dtype='float32'))
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """
        Take and adapt from huggingface/transformers.
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])
        filter_indices = logits < F.topk(logits, top_k, descending=True)[
            0][..., -1, None]
        logits[filter_indices] = filter_value

    if 0.0 <= top_p <= 1.0:
        sorted_logits, sorted_indices = F.sort(logits, descending=False)

        cumulative_probs = F.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_filter = cumulative_probs <= 1 - top_p

        if min_tokens_to_keep > 1:
            sorted_indices_to_filter[..., -min_tokens_to_keep] = 0

        filter_indices = F.scatter(
            sorted_indices_to_filter, axis=1, index=sorted_indices, source=sorted_indices_to_filter)

        logits[filter_indices] = filter_value

    return logits


def multinomial(x, num_samples, repalcement=None):
    """
        Implemented by python.
    """
    if x.ndim != 2:
        raise ValueError(f"expected input has 2 dimention, but got {x.ndim}")
    if repalcement is not None:
        raise ValueError("Currently not support `replacement`")
    _, num_col = x.shape
    x = F.cumsum(x, axis=1)
    choices = []
    for t in x:
        t = t.numpy()
        ch = []
        for _ in range(num_samples):
            prob = random.random()
            for id in range(num_col):
                if t[id] > prob:
                    idx = id
                    break
            ch.append(idx)
        choices.append(ch)
    return mge.tensor(choices, dtype='int32')
