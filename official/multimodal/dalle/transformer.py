from functools import partial
from itertools import cycle, islice
from typing import Optional, Sequence, Union

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .attention import Attention, SparseAxialAttention, SparseConvolutionAttention, expand
from .functional import RotaryEmbedding


class LayerScale(M.Module):
    def __init__(self, dim: int, depth: int):
        super(LayerScale, self).__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 < depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = F.full(shape=(1, 1, dim), value=init_eps)
        self.scale = mge.Parameter(scale, dtype='float32')

    def forward(self, x):
        return x * self.scale
        # return x


class PreNorm(M.Module):
    def __init__(self, dim: int, norm_out: bool = False):
        super(PreNorm, self).__init__()
        self.norm = M.LayerNorm(dim)
        self.norm_out = M.LayerNorm(dim) if norm_out else M.Identity()

    def forward(self, x, model, **kwargs):
        x = self.norm(x)
        x = model(x, **kwargs)
        return self.norm_out(x)


class GEGLU(M.Module):
    def forward(self, x):
        x, gates = F.split(x, nsplits_or_sections=2, axis=x.ndim - 1)
        return x * F.gelu(gates)


class MLP(M.Module):
    def __init__(self, dim: int, dropout: float = 0., ratio: float = 4.):
        super(MLP, self).__init__()
        self.net = M.Sequential(
            M.Linear(dim, int(dim * ratio * 2)),
            GEGLU(),
            M.Dropout(dropout),
            M.Linear(int(dim * ratio), dim)
        )

    def forward(self, x):
        return self.net(x)


class PreShiftToken(M.Module):
    def __init__(self, image_size: int, seq_len: int):
        super(PreShiftToken, self).__init__()
        self.image_size = image_size
        self.seq_len = seq_len
        self.img_seq_len = image_size ** 2
        self.text_len = seq_len - self.img_seq_len + 1

    def forward(self, x):
        B, L, C = x.shape
        padding = self.seq_len - L + 1

        # if sequence is shorter than the text length, no image tokens to shift
        if L < self.text_len:
            return x
        x = F.nn.pad(x, pad_width=[(0, 0), (0, padding), (0, 0)])

        text_x, img_x = F.split(x, nsplits_or_sections=[self.text_len], axis=1)

        img_x = img_x.reshape(B, self.image_size, self.image_size, C)

        # shift 1 from the left for text tokens

        text_x_shift, text_x_pass = F.split(text_x, 2, axis=2)
        text_x_shift = F.nn.pad(text_x_shift, [(0, 0), (1, 0), (0, 0)])[:, :-1, :]
        text_x = F.concat([text_x_shift, text_x_pass], axis=-1)

        # shift from top, left for image tokens
        img_shift_top, img_shift_left, *img_pass = F.split(img_x, 4, axis=3)
        img_shift_left = F.nn.pad(img_shift_left, [(0, 0), (0, 0), (1, 0), (0, 0)])[:, :, :-1, :]
        img_shift_top = F.nn.pad(img_shift_top, [(0, 0), (1, 0), (0, 0), (0, 0)])[:, :-1]
        img_x = F.concat([img_shift_top, img_shift_left, *img_pass], axis=-1)

        # merge text and image sequence back together
        # b, h, w, c -> b, l, c
        img_x = F.flatten(img_x, 1, 2)
        x = F.concat([text_x, img_x], axis=1)[:, :-padding]

        return x


class SequentialSequence(M.Module):
    def __init__(
        self,
        attn_type: str,
        attn: Union[Attention, SparseAxialAttention, SparseConvolutionAttention],
        norm1: PreNorm,
        mlp: MLP,
        norm2: PreNorm,
        ls1: LayerScale,
        ls2: LayerScale,
        preshift: Optional[PreShiftToken] = None,
    ):
        super(SequentialSequence, self).__init__()
        self.__setattr__(attn_type, attn)
        self.attn_type = attn_type
        self.norm1 = norm1
        self.mlp = mlp
        self.norm2 = norm2
        self.layer_scale1 = ls1
        self.layer_scale2 = ls2
        self.shift_token = preshift if preshift is not None else M.Identity()

    def forward(self, x, **kwargs):
        attn = self.__getattribute__(self.attn_type)
        x = x + self.layer_scale1(self.norm1(self.shift_token(x), attn, **kwargs))
        x = x + self.layer_scale2(self.norm2(self.shift_token(x), self.mlp))
        return x


def _get_sliced_cycle_list(x, depth):
    if x is None:
        return list(range(depth))
    return list(islice(cycle(x), depth))


class Transformer(M.Module):

    COLUMN_ATTENTION = 'column'
    ROW_ATTENTION = 'row'
    CONVOLUTIONAL_ATTENTION = 'conv'
    FULL_ATTENTION = 'full'

    def __init__(  # pylint: disable=too-many-statements
        self,
        embed_dim: int,
        depth: int,
        seq_len: int,
        image_fmap_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        rotary_emb: bool = True,
        shift_token: bool = False,
        norm_out: bool = False,
        causal: bool = True,
        stable: bool = False,
        attention_types: Optional[Union[Sequence[str], str]] = None,
        shared_attn_ids: Optional[Sequence[int]] = None,
        shared_mlp_ids: Optional[Sequence[int]] = None,
    ):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.image_fmap_size = image_fmap_size
        self.depth = depth

        if attention_types is None:
            # standard dalle
            attn_type_list = list(
                islice(cycle(self.get_type_in_period(4)), depth - 1))
            attn_type_list.append('conv')
        elif isinstance(attention_types, (list, tuple)):
            if len(attention_types) != depth:
                attn_type_list = _get_sliced_cycle_list(attention_types, depth)
            else:
                attn_type_list = attention_types
        else:
            raise TypeError('Wrong type for `attention_types`.')
        self.attn_type_list = attn_type_list

        shared_mlp_ids = _get_sliced_cycle_list(shared_mlp_ids, depth)
        shared_attn_ids = _get_sliced_cycle_list(shared_attn_ids, depth)

        if len(shared_mlp_ids) != len(attn_type_list):
            raise ValueError(
                "Make sure `shared_mlp_ids` and `attention_types` correspond to each other")

        if len(shared_attn_ids) != len(attn_type_list):
            raise ValueError(
                "Make sure `shared_attn_ids` and `attention_types` correspond to each other")

        # The shared layers have the same key
        shared_mlp_layers = {}
        shared_attn_layers = {}

        attention_mapper = {
            'full': partial(Attention, causal=causal),
            'conv': partial(SparseConvolutionAttention, image_size=image_fmap_size),
            'row': partial(
                SparseAxialAttention,
                axial_type=SparseAxialAttention.ROW,
                image_size=image_fmap_size
            ),
            'column': partial(
                SparseAxialAttention,
                axial_type=SparseAxialAttention.COLUMN,
                image_size=image_fmap_size
            ),
        }

        self.layers = []
        preshift = PreShiftToken(image_size=image_fmap_size,
                                 seq_len=seq_len) if shift_token else None
        for idx, (attn_type, mlp_id, attn_id) in enumerate(zip(attn_type_list, shared_mlp_ids, shared_attn_ids)):  # noqa: E501
            attn, last_attn_type = shared_attn_layers.get(
                attn_id, (None, None))
            if attn is None:
                attn = attention_mapper[attn_type](
                    embed_dim=embed_dim,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    drop_out=attn_dropout,
                    stable=stable
                )
                shared_attn_layers[attn_id] = (attn, attn_type)
            elif last_attn_type != attn_type:
                raise RuntimeError(
                    "`attb_type` do not match with `shared_attn_ids`.")

            mlp = shared_mlp_layers.get(mlp_id, None)
            if mlp is None:
                mlp = MLP(embed_dim, mlp_dropout, mlp_ratio)
                shared_mlp_layers[mlp_id] = mlp

            norm1 = PreNorm(embed_dim, norm_out=norm_out)
            norm2 = PreNorm(embed_dim, norm_out=norm_out)
            ls1 = LayerScale(embed_dim, depth=idx + 1)
            ls2 = LayerScale(embed_dim, depth=idx + 1)
            self.layers.append(SequentialSequence(
                attn_type, attn, norm1, mlp, norm2, ls1, ls2, preshift))

        pos_emb = None
        self.rotary_emb = rotary_emb
        if rotary_emb:
            rotary_dim = head_dim // 3
            img_seq_len = image_fmap_size ** 2
            text_len = seq_len - img_seq_len + 1

            text_pos_embed = RotaryEmbedding(rotary_dim)
            img_aixal_pos_embed = RotaryEmbedding(
                rotary_dim, freqs_type='pixel')

            text_freqs = text_pos_embed(F.arange(text_len))
            # image is given a position far away from text
            img_to_text_freqs = text_pos_embed(F.full((img_seq_len,), 8192))
            text_freqs = F.concat((text_freqs, img_to_text_freqs), axis=0)

            img_freqs_axial = img_aixal_pos_embed(
                mge.tensor(np.linspace(-1, 1, image_fmap_size, dtype=np.float64)))

            c = img_freqs_axial.shape[0]
            img_freqs = F.concat(
                (
                    expand(img_freqs_axial, c, 1),
                    expand(img_freqs_axial, c, 0)
                ),
                axis=-1,
            )
            img_freqs = F.flatten(img_freqs, 0, 1)
            # text is given a position of -10 apart from the image axial positions,
            # which is from range [-1, 1]
            text_axial_freqs = img_aixal_pos_embed(F.full((text_len,), -10.))
            text_axial_freqs = F.concat(
                (text_axial_freqs, text_axial_freqs), axis=-1)
            img_freqs = F.concat((text_axial_freqs, img_freqs), axis=0)

            pos_emb = F.concat((text_freqs, img_freqs), axis=-1)
            pos_emb = F.expand_dims(pos_emb, axis=0)
        self.pos_emb = pos_emb

    @property
    def attention_types(self):
        return self.attn_type_list

    @staticmethod
    def is_column_attention(idx):
        return (idx - 2) % 4 == 0

    @staticmethod
    def get_type_in_period(preiod):
        return [
            Transformer.COLUMN_ATTENTION
            if Transformer.is_column_attention(x)
            else Transformer.ROW_ATTENTION
            for x in range(1, preiod + 1)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=self.pos_emb)
        return x
