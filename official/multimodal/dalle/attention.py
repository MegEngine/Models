import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import Tensor


def get_minium_value(x):
    minium = np.finfo(x.dtype).min
    return mge.tensor(minium, dtype=x.dtype)


def build_causal_attention_mask(i, j):
    mask = np.ones((i, j))
    mask = np.triu(mask, k=j - i + 1)
    return mge.tensor(mask).astype(np.bool8)


def broadcast_matmul(x, y, axis=2):
    out = F.expand_dims(x, axis=axis) @ y
    out = F.squeeze(out, axis)
    return out


def broadcast_to(x, axis, shape):
    out = F.expand_dims(x, axis=axis)
    out = F.broadcast_to(out, shape=shape)
    return out


def expand(x: Tensor, repeats, axis):
    x = F.expand_dims(x, axis=axis)
    if isinstance(axis, (list, tuple)):
        assert len(repeats) == len(axis)
        for ax, r in zip(axis, repeats):
            x = F.repeat(x, repeats=r, axis=ax)
        return x
    return F.repeat(x, repeats=repeats, axis=axis)


class StableSoftmax(M.Softmax):
    def __init__(self, axis, alpha=32 ** 2):
        super(StableSoftmax, self).__init__(axis=axis)
        self.alpha = alpha

    def forward(self, x):
        x = x / self.alpha
        x = x - F.max(x, axis=self.axis, keepdims=True).detach()
        return super().forward(x)


class BaseAttention(M.Module):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int,
        head_dim: int = 64,
        drop_out: float = 0.,
        stable: bool = False,
    ):
        super(BaseAttention, self).__init__()
        self.heads = num_heads
        self.seq_len = seq_len
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim
        self.qkv_proj = M.Linear(embed_dim, 3 * inner_dim, bias=False)

        self.out_proj = M.Sequential(
            M.Linear(inner_dim, embed_dim, bias=True),
            M.Dropout(drop_out)
        )

        self.softmax = StableSoftmax(-1) if stable else M.Softmax(-1)

    def forward(self, x, mask=None):
        raise NotImplementedError()

    def apply_pos_embed(self, pos_emb, qkv):
        n = qkv[0].shape[-2]
        pos_emb = pos_emb[..., :n, :]
        return [self.apply_rotary_embed(pos_emb, x) for x in qkv]

    def apply_rotary_embed(self, pos_emb, t, start_index=0):
        pos_dim = pos_emb.shape[-1]
        end_index = start_index + pos_dim
        assert pos_dim <= t.shape[-1]

        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * F.cos(pos_emb)) + (self.rotate_half(t) * F.sin(pos_emb))

        return F.concat([t_left, t, t_right], axis=-1)

    def rotate_half(self, x):
        x1, x2 = F.split(x, 2, axis=-1)
        x = F.stack([-x2, x1], axis=-1)
        return F.flatten(x, -2)


class Attention(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int,
        causal: bool = True,
        head_dim: int = 64,
        drop_out: float = 0.,
        static_mask: Tensor = None,
        stable: bool = False,
    ):
        super(Attention, self).__init__(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            drop_out=drop_out,
            stable=stable
        )

        self.static_mask = static_mask
        self.causal = causal

    def forward(self, x, mask=None, rotary_pos_emb=None):
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = [
            t.reshape(B, L, self.heads, -1)
            .transpose(0, 2, 1, 3)
            for t in F.split(qkv, nsplits_or_sections=3, axis=2)
        ]

        if rotary_pos_emb is not None:
            q, k, v = self.apply_pos_embed(rotary_pos_emb, qkv)
        else:
            q, k, v = qkv

        q = q * self.scale

        attn = q @ k.transpose(0, 1, 3, 2)

        mask_value = get_minium_value(attn)

        if mask is not None:
            mask = F.expand_dims(mask, axis=[1, 2])
            attn = F.where(mask, attn, mask_value)

        if self.causal:
            mask = build_causal_attention_mask(*attn.shape[2:])
            attn = F.where(mask, mask_value, attn)

        if self.static_mask is not None:
            attn = F.where(self.static_mask, attn, mask_value)

        attn = self.softmax(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class SparseConvolutionAttention(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        image_size: int = 32,
        kernel_size: int = 5,
        dilation: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        drop_out: int = 0,
        stable: bool = False,
    ):
        if kernel_size % 2 != 1:
            raise ValueError(
                f"'kernel_size' must be odd, but got {kernel_size}")
        super(SparseConvolutionAttention, self).__init__(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            drop_out=drop_out,
            stable=stable
        )

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation

    @staticmethod
    def build_attention_mask(image_size, padding, kernel_size, dilation, text_len, mask=None):
        ones = F.ones((1, 1, image_size, image_size))
        ones = F.nn.pad(ones, padding, constant_value=0.)
        # 1, 1, num, num, kernel_size, kernel_size
        ones = F.sliding_window(ones, kernel_size, dilation=dilation)
        # since the channel is 1, just merge it into num ** 2
        # 1, num ** 2, kernel_size ** 2
        ones = F.flatten(F.flatten(ones, 4), 1, 3)
        conv_padding_mask = ones == 0.
        if mask is None:
            mask = F.ones(
                (*conv_padding_mask.shape[:2], text_len)).astype(np.bool8)
        # TODO: to fit mask input
        else:
            # mask = F.expand_dims(mask[:, :text_len], axis=0)
            pass
        mask = F.concat([F.logical_not(mask), conv_padding_mask], axis=-1)
        return mask

    def conv(self, x, padding):
        b, _, c = x.shape
        # b, c, h, w
        x = x.reshape(b, self.image_size, self.image_size,
                      c).transpose(0, 3, 1, 2)
        # b, c, h', w'
        x = F.nn.pad(x, padding)
        # b, c, num, num, ks, ks
        x = F.sliding_window(x, self.kernel_size, dilation=self.dilation)
        # b, c, num, num, ks**2
        x = F.flatten(x, 4)
        # b, c, num**2, ks**2
        x = F.flatten(x, 2, 3)
        # b, num**2, ks**2, c
        x = x.transpose(0, 2, 3, 1)
        return x

    def forward(self, x, mask=None, rotary_pos_emb=None):
        B, L, _ = x.shape
        device = x.device

        img_seq_len = self.image_size ** 2
        text_len = self.seq_len + 1 - img_seq_len

        padding = self.seq_len - L + 1
        x = F.nn.pad(x, pad_width=[(0, 0), (0, padding), (0, 0)])

        qkv = self.qkv_proj(x)
        qkv = [
            t.reshape(B, L + padding, self.heads, -1)
            .transpose(0, 2, 1, 3)
            .reshape(B * self.heads, L + padding, -1)
            for t in F.split(qkv, nsplits_or_sections=3, axis=2)
        ]

        if rotary_pos_emb is not None:
            q, k, v = self.apply_pos_embed(rotary_pos_emb, qkv)
        else:
            q, k, v = qkv

        q = q * self.scale
        # split q, k, v to text and img
        split_index = L + padding - img_seq_len
        # img: B*head, img_seq_len, head_dim
        # text: B*head, L+padding, head_dim
        (q_text, q_img), (k_text, k_img), (v_text, v_img) = [
            F.split(t, [split_index], axis=1) for t in (q, k, v)]

        # text normal attention
        attn_text = q_text @ k_text.transpose(0, 2, 1)
        mask_value = get_minium_value(attn_text)

        attn_text = F.where(build_causal_attention_mask(
            *attn_text.shape[-2:]), mask_value, attn_text)

        attn_text = self.softmax(attn_text)

        text_out = attn_text @ v_text

        # image sparse Attention
        receptive_field = (self.kernel_size - 1) * self.dilation + 1
        padding = receptive_field // 2
        causal_padding = [(0, 0), (0, 0), (padding * 2, 0), (padding * 2, 0)]
        k_img = self.conv(k_img, causal_padding)
        v_img = self.conv(v_img, causal_padding)

        # let image attend to all of text

        # image_attn = F.expand_dims(q_img, axis=2) @ k_img.transpose(0, 1, 3, 2)
        # image_attn = F.squeeze(image_attn)
        # B*head, img_seq_len, num_of_window
        image_attn = broadcast_matmul(q_img, k_img.transpose(0, 1, 3, 2))
        imgae_to_text_attn = q_img @ k_text.transpose(0, 2, 1)

        # build conv padding mask together with text casual mask
        mask = self.build_attention_mask(
            self.image_size, causal_padding, self.kernel_size, self.dilation, text_len, mask)
        mask = mask.to(device)
        mask = F.repeat(mask, repeats=B * self.heads, axis=0)

        # get image attention of all image and text tokens
        attn = F.concat([imgae_to_text_attn, image_attn], axis=-1)
        attn = F.where(mask, mask_value, attn)
        attn = self.softmax(attn)

        # aggregate, calculate the output separately
        image_to_text_attn, image_attn = F.split(
            attn, nsplits_or_sections=[text_len], axis=2)

        image_to_image_out = broadcast_matmul(image_attn, v_img)
        image_to_text_out = image_to_text_attn @ v_text

        image_out = image_to_image_out + image_to_text_out

        # combination
        out = F.concat([text_out, image_out], axis=1)
        # b, h, l, c -> b, l, h, c
        out = out.reshape(B, self.heads, *out.shape[-2:]).transpose(0, 2, 1, 3)
        out = F.flatten(out, 2)
        # return self.out_proj(out)[:, :L]
        out = self.out_proj(out)[:, :L]
        return out


class SparseAxialAttention(BaseAttention):
    ROW = 0
    COLUMN = 1

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        axial_type: int,
        image_size: int = 32,
        num_heads: int = 8,
        head_dim: int = 64,
        drop_out: float = 0.,
        stable: bool = False,
    ):
        super(SparseAxialAttention, self).__init__(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            drop_out=drop_out,
            stable=stable
        )
        if axial_type not in [self.ROW, self.COLUMN]:
            raise ValueError("axial type must be `ROW` or `COLUMN`")
        self.image_size = image_size
        self.axial_type = axial_type

    @staticmethod
    def build_attention_mask(shape, image_size, text_len, mask=None):
        b, x, i, _ = shape
        causal_mask = build_causal_attention_mask(i, image_size)
        # causal_mask = broadcast_to(causal_mask, axis=[0, 1], shape=[
        #                            b, x, i, image_size])

        if mask is None:
            mask = F.ones((i, text_len)).astype(np.bool8)
        # TODO: to fit mask input
        else:
            # mask = F.expand_dims(mask[:, :text_len], axis=0)
            pass

        mask = F.concat([F.logical_not(mask), causal_mask], axis=-1)
        mask = expand(mask, repeats=[b, x], axis=[0, 1])
        return mask

    def split(self, x):
        B, _, C = x.shape
        x = x.reshape(B, self.image_size, self.image_size, C)
        if self.axial_type == self.COLUMN:
            x = x.transpose(0, 2, 1, 3)
        return x

    def merge(self, x):
        if self.axial_type == self.COLUMN:
            x = x.transpose(0, 2, 1, 3)
        x = F.flatten(x, 1, 2)
        return x

    def forward(self, x, mask=None, rotary_pos_emb=None):
        B, L, _ = x.shape

        img_seq_len = self.image_size ** 2
        text_len = self.seq_len + 1 - img_seq_len

        padding = self.seq_len - L + 1
        x = F.nn.pad(x, pad_width=[(0, 0), (0, padding), (0, 0)])

        qkv = self.qkv_proj(x)
        qkv = [
            t.reshape(B, L + padding, self.heads, -1)
            .transpose(0, 2, 1, 3)
            .reshape(B * self.heads, L + padding, -1)
            for t in F.split(qkv, nsplits_or_sections=3, axis=2)
        ]

        if rotary_pos_emb is not None:
            q, k, v = self.apply_pos_embed(rotary_pos_emb, qkv)
        else:
            q, k, v = qkv

        q = q * self.scale

        # split q, k, v to text and img
        split_index = L + padding - img_seq_len
        (q_text, q_img), (k_text, k_img), (v_text, v_img) = [
            F.split(t, [split_index], axis=1) for t in (q, k, v)]

        # text normal attention
        attn_text = q_text @ k_text.transpose(0, 2, 1)
        mask_value = get_minium_value(attn_text)

        attn_text = F.where(build_causal_attention_mask(
            *attn_text.shape[-2:]), mask_value, attn_text)

        attn_text = self.softmax(attn_text)

        text_out = attn_text @ v_text

        # image sparse attention

        q_img, k_img, v_img = [self.split(t) for t in (q_img, k_img, v_img)]

        image_attn = q_img @ k_img.transpose(0, 1, 3, 2)
        image_to_text_attn = q_img @ expand(
            k_text, repeats=self.image_size, axis=1).transpose(0, 1, 3, 2)

        attn = F.concat([image_to_text_attn, image_attn], axis=-1)

        # build mask
        mask = self.build_attention_mask(
            attn.shape, self.image_size, text_len, mask)

        attn = F.where(mask, mask_value, attn)
        attn = self.softmax(attn)

        # aggregate
        image_to_text_attn, image_attn = F.split(
            attn, nsplits_or_sections=[text_len], axis=3)
        image_to_image_out = image_attn @ v_img
        image_to_text_out = image_to_text_attn @ expand(
            v_text, repeats=self.image_size, axis=1)

        image_out = image_to_image_out + image_to_text_out

        # merge
        image_out = self.merge(image_out)

        # combination
        out = F.concat([text_out, image_out], axis=1)

        out = out.reshape(B, self.heads, *out.shape[-2:])
        out = out.transpose(0, 2, 1, 3)
        out = F.flatten(out, 2)
        out = self.out_proj(out)
        return out[:, :L]
