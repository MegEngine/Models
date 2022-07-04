# TODO: Remove this file if megengine supports MultiheadAttention one day
from typing import Optional

import numpy as np

import megengine as mge
import megengine.module as M
from megengine import Tensor
from megengine.functional import (
    broadcast_to,
    concat,
    dropout,
    full,
    linear,
    logical_or,
    matmul,
    repeat,
    softmax,
    split,
    where,
    zeros
)


def multi_head_attention(  # pylint: disable=too-many-branches, too-many-statements
    query: Tensor, key: Optional[Tensor], value: Optional[Tensor],
    head_dim: int,
    num_heads: int,
    attn_output_weight: Tensor,
    attn_output_bias: Optional[Tensor],
    dropout_p: float = 0,
    in_proj_weight: Optional[Tensor] = None,
    query_weight: Optional[Tensor] = None,
    key_weight: Optional[Tensor] = None,
    value_weight: Optional[Tensor] = None,
    in_proj_bias: Optional[Tensor] = None,
    query_bias: Optional[Tensor] = None,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    bias_k: Optional[Tensor] = None,
    bias_v: Optional[Tensor] = None,
    add_zero_attn: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    compute_mode: str = "default",
):
    """
    Args:
        query, key, value(Tensor): map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        head_dim(int): Total dimension of every head.
        num_heads(int): parallel attention heads.
        query_weight, query_bias, key_weight, key_bias, value_weight, value_bias(Tensor):
            input projection weight and bias.
        attn_output_weight, attn_output_bias(Tensor):output projection weight and bias.
        dropout_p(float): probability of an element to be zeroed.
        bias_k, bias_v(Tensor): bias of the key and value sequences to be added at dim=0.
        add_zero_attn(bool): add a new batch of zeros to the key and value sequences at dim=1.
        key_padding_mask(Tensor): if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -1e9.
        need_weights(bool): output attn_output_weights.
        attn_mask(Tensor): 3D mask that prevents attention to certain positions.

    Shape:
        Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length,
                N is the batch size, E is the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length,
                N is the batch size, E is the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length,
                N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size,
                S is the source sequence length.
            - attn_mask: 3D mask :math:`(N*num_heads, L, S)` where N is the batch size,
                L is the target sequence length,
            S is the source sequence length.
                attn_mask ensures that position i is allowed to attend the unmasked positions.

        Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length,
                N is the batch size, E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
                L is the target sequence length, S is the source sequence length.
    """
    tgt_len = query.shape[0]
    bsz = query.shape[1]

    # 1) Do all the linear projections in batch
    if in_proj_weight is not None:
        query, key, value = split(
            linear(query, in_proj_weight, in_proj_bias, compute_mode), 3, axis=-1)
    else:
        assert query_weight is not None and query_bias is not None and key_weight is not None
        q = query
        query = linear(q, query_weight, query_bias, compute_mode)
        key = linear(q if key is None else key, key_weight, key_bias, compute_mode)
        value = linear(q if value is None else value, value_weight, value_bias, compute_mode)
    # add bias along batch dimension
    if bias_k is not None and bias_v is not None:
        key = concat([key, repeat(bias_k, bsz, axis=1)])
        value = concat([value, repeat(bias_v, bsz, axis=1)])
        if attn_mask is not None:
            attn_mask_temp = full(
                (attn_mask.shape[0], attn_mask.shape[1], 1),
                False, dtype=bool, device=attn_mask.device
            )
            attn_mask = concat([attn_mask, attn_mask_temp], axis=2)
        if key_padding_mask is not None:
            key_padding_mask_temp = full(
                (key_padding_mask.shape[0], 1), False, dtype=bool, device=key_padding_mask.device
            )
            key_padding_mask = concat([key_padding_mask, key_padding_mask_temp], axis=1)

    query = query.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    key = key.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    value = value.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    # add zero attention along batch dimension
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        key = concat(
            [key, zeros(zero_attn_shape, dtype=key.dtype)],
            axis=1,
            device=key.device,
        )
        value = concat(
            [value, zeros(zero_attn_shape, dtype=value.dtype)],
            axis=1,
            device=value.device,
        )
        if attn_mask is not None:
            attn_mask_temp = full(
                (attn_mask.shape[0], attn_mask.shape[1], 1),
                False, dtype=bool, device=attn_mask.device
            )
            attn_mask = concat([attn_mask, attn_mask_temp], axis=2)
        if key_padding_mask is not None:
            key_padding_mask_temp = full(
                (key_padding_mask.shape[0], 1), False, dtype=bool, device=key_padding_mask.device
            )
            key_padding_mask = concat([key_padding_mask, key_padding_mask_temp], axis=1)
    # update source sequence length after adjustments
    src_len = key.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == bsz
        assert key_padding_mask.shape[1] == src_len
        key_padding_mask = key_padding_mask.reshape(bsz, 1, 1, src_len)
        key_padding_mask = broadcast_to(
            key_padding_mask, (bsz, num_heads, 1, src_len)
        ).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = Tensor(logical_or(attn_mask, key_padding_mask))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == np.bool:
        new_attn_mask = where(attn_mask, full(attn_mask.shape, -1e9), full(attn_mask.shape, 0.0))
        attn_mask = new_attn_mask

    # 2) Apply attention on all the projected vectors in batch.
    attn_output_weights = matmul(
        query, key.transpose(0, 2, 1), compute_mode=compute_mode
    ) / (head_dim ** 0.5)
    if attn_mask is not None:
        attn_output_weights = attn_output_weights + attn_mask

    attn_output_weights = attn_output_weights.reshape(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, axis=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, dropout_p)
    attn_output = matmul(attn_output_weights, value, compute_mode=compute_mode)

    # 3) "Concat" using a reshape and apply a final linear.
    attn_output = attn_output.transpose(1, 0, 2).reshape(
        tgt_len, bsz, num_heads * head_dim
    )
    attn_output_weights = (
        attn_output_weights.reshape(bsz, num_heads, tgt_len, src_len).sum(axis=1)
        / num_heads
    )
    attn_output = linear(
        attn_output, attn_output_weight, attn_output_bias, compute_mode
    )
    if need_weights:
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class MultiheadAttention(M.Module):
    """
        A simple implementation of Multi-Head Attention for CLIP.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            bias=True,
            drop_out=0.,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_out = drop_out
        self.add_zero_attn = add_zero_attn

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})")
        if self._qkv_same_embed_dim:
            self.in_proj = M.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            self.q_proj = M.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = M.Linear(embed_dim, self.kdim, bias=bias)
            self.v_proj = M.Linear(embed_dim, self.vdim, bias=bias)

        self.out_proj = M.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = mge.Parameter(mge.random.normal(size=(1, 1, embed_dim)))
            self.bias_v = mge.Parameter(mge.random.normal(size=(1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self._init_parameters()

    def _init_parameters(self):
        M.init.xavier_uniform_(self.in_proj.weight)
        M.init.zeros_(self.in_proj.bias)
        M.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False
    ):
        if self._qkv_same_embed_dim:
            return multi_head_attention(
                q, k, v,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                attn_output_weight=self.out_proj.weight,
                attn_output_bias=self.out_proj.bias,
                in_proj_weight=self.in_proj.weight,
                in_proj_bias=self.in_proj.bias,
                bias_k=self.bias_k,
                bias_v=self.bias_v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights)
        else:
            return multi_head_attention(
                q, k, v,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                query_weight=self.q_proj.weight,
                query_bias=self.q_proj.bias,
                key_weight=self.k_proj.weight,
                key_bias=self.k_proj.bias,
                value_weight=self.v_proj.weight,
                value_bias=self.v_proj.bias,
                attn_output_weight=self.out_proj.weight,
                attn_output_bias=self.out_proj.bias,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights)
