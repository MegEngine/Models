from typing import Optional, Sequence, Union

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import Tensor

from ..clip import CLIP
from .functional import AxialPositionalEmbedding, DivideMax, gumbel_sample, topk
from .tokenizer import SimpleTokenizer, YttmTokenizer
from .transformer import Transformer
from .vae import OpenAIDiscreteVAE, VQGanVAE


class DALLE(M.Module):
    r"""
    The dalle model.

    Args:
        embed_dim: The embedded dimonsion of input tokens.
        vae: The vae model, which support OpenAIDiscreteVAE and VQGanVAE for now.
        num_text_tokens: The length of text tokens.
        text_seq_len: The length of text sequence.
        depths: How many blocks in Transformer. Default: 64.
        num_heads: Number of head in multi-head atteniton. Default: 8
        head_dim: Dimonsion for each head. Default: 64.
        mlp_ratio: The expand ratio in mlp of transformer block. Default: 4.0 .
        mlp_dropout: Dropout ratio in mlp. Default: 0.0 .
        attn_dropout: Dropout ratio in attentions. Default: 0.0 .
        stable: Whether to use stable ways to process the transformer output. Default: False.
        norm_out: Whether to use an additional normlize in transformer block. Default: False.
        rotary_emb: Whether to use rotary embedding in transformer. Default: True.
        loss_img_weight: THe weight of image when training. Default: None.
        attention_types: Attention types for each layers, It's unnecessary
            to have the same length of depths. Dafault: None.
        shared_attn_ids: The ids of shared attention layer. Default: None.
        shared_mlp_ids: The ids of shared mlp layers. Default: None.
    """

    def __init__(
        self,
        embed_dim: int,
        # TODO: add more vae model, such as DVAE, VQVAE
        vae: Union[OpenAIDiscreteVAE, VQGanVAE],
        num_text_tokens: int,
        text_seq_len: int,
        depths: int = 64,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        stable: bool = False,
        norm_out: bool = False,
        rotary_emb: bool = True,
        loss_img_weight: Optional[float] = None,
        attention_types: Optional[Union[Sequence[str], str]] = None,
        shared_attn_ids: Optional[Sequence[int]] = None,
        shared_mlp_ids: Optional[Sequence[int]] = None,
    ):
        super(DALLE, self).__init__()
        if not isinstance(vae, (OpenAIDiscreteVAE, VQGanVAE)):
            raise TypeError(
                "The vae must be an instance of `OpenAIDiscreteVAE` and `VQGanVAE`")
        vae.eval()
        self.vae = vae

        image_size = vae.image_size
        imgae_fmap_size = image_size // (2 ** vae.num_layers)

        self.stable = stable
        self.text_seq_len = text_seq_len
        self.num_image_tokens = vae.num_tokens
        self.image_seq_len = imgae_fmap_size ** 2
        self.num_text_tokens = num_text_tokens + text_seq_len
        self.total_tokens = self.num_text_tokens + self.num_image_tokens
        self.seq_len = text_seq_len + self.image_seq_len

        # +1 for <bos>
        self.text_pos_emb = M.Embedding(
            text_seq_len + 1, embed_dim) if not rotary_emb else None
        self.image_pos_emb = AxialPositionalEmbedding(
            embed_dim, axial_shape=(imgae_fmap_size, imgae_fmap_size)) if not rotary_emb else None

        self.transformer = Transformer(
            embed_dim=embed_dim,
            depth=depths,
            seq_len=self.seq_len,
            image_fmap_size=imgae_fmap_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dropout=mlp_dropout,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            norm_out=norm_out,
            stable=stable,
            rotary_emb=rotary_emb,
            attention_types=attention_types,
            shared_attn_ids=shared_attn_ids,
            shared_mlp_ids=shared_mlp_ids,
        )

        self.to_logits = M.Sequential(
            M.LayerNorm(embed_dim),
            M.Linear(embed_dim, self.total_tokens)
        )

        self.norm_by_max = DivideMax(axis=-1) if stable else M.Identity()

        self.text_emb = M.Embedding(self.num_text_tokens, embed_dim)
        self.image_emb = M.Embedding(self.num_image_tokens, embed_dim)

        self.logits_mask = self._build_logits_mask(text_seq_len, num_text_tokens)

        self.loss_img_weight = loss_img_weight

    def _build_logits_mask(self, text_seq_len: int, num_text_tokens: int):
        seq_range = F.arange(self.seq_len)
        logits_range = F.arange(self.total_tokens)
        seq_range = F.expand_dims(seq_range, axis=[0, 2])
        logits_range = F.expand_dims(logits_range, axis=[0, 1])

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens))
            | ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )
        return logits_mask

    def generate_texts(
        self,
        tokenizer: Union[SimpleTokenizer, YttmTokenizer],
        text: Optional[str] = None,
        filter_thread: float = 0.5,
        temperature: float = 1.0
    ):
        if text is None or text == "":
            text_tokens = mge.tensor([[0]])
        else:
            text_tokens = mge.tensor([tokenizer.encode(text)])

        for _ in range(text_tokens.shape[1], self.text_seq_len):
            tokens = self.text_emb(text_tokens)
            if self.text_pos_emb is not None:
                text_pos_emb = self.text_pos_emb(  # pylint: disable=not-callable
                    F.arange(text_tokens.shape[1], dtype='int32'))
                tokens += text_pos_emb

            seq_len = tokens.shape[1]

            trans_out = self.transformer(tokens)
            trans_out = self.norm_by_max(trans_out)

            logits = self.to_logits(trans_out)

            # mask and filter
            logits_mask = self.logits_mask[:, :seq_len]
            inf = F.full_like(logits, -float('Inf'))
            logits = F.where(logits_mask, inf, logits)
            logits = logits[:, -1, :]

            filtered_logits = topk(logits, thread=filter_thread)

            # sample
            sample = gumbel_sample(
                filtered_logits, temperature=temperature, axis=-1)

            text_tokens = F.concat([text_tokens, sample[:, None]], axis=-1)

        padding_tokens = set(np.arange(self.text_seq_len)
                             + (self.num_text_tokens - self.text_seq_len))
        texts = []
        filter_tokens = set((49406, 40407, 0)) | padding_tokens
        for text_token in text_tokens:
            text_token = text_token.tolist()
            text_token = [token for token in text_token if token not in filter_tokens]
            texts.append(tokenizer.decode(text_token))
        return text_tokens, texts

    def generate_images(
        self,
        text: Tensor,
        clip: Optional[CLIP] = None,
        filter_thread: float = 0.5,
        temperature: float = 1.0,
        image: Optional[Tensor] = None,
        num_init_img_tokens: Optional[float] = None,
        cond_scale: float = 1.,
    ):
        text = text[:, :self.text_seq_len]
        out = text

        if image is not None:
            image_size = self.vae.image_size
            if image.shape[1:] != (3, image_size, image_size):
                raise ValueError(
                    f"expect shape of input image to be (3, {image_size}, {image_size}), but got {image.shape[1:]}")  # noqa: E501

            indices = self.vae.get_codebook_indices(image)
            # OpenAI used 14 * 32 initial tokens to prime, 0.4375 = 14 / 35
            num_img_tokens = (
                int(0.4375 * self.image_seq_len)
                if num_init_img_tokens is None else
                num_init_img_tokens
            )
            if num_img_tokens >= self.image_seq_len:
                raise ValueError(
                    f"expect number of initial image tokens for priming to be less than the total image token sequence length {self.image_seq_len}, but got {num_img_tokens}")  # noqa: E501
            indices = indices[:, :num_img_tokens]
            out = F.concat([out, indices], axis=-1)

        for cur_len in range(out.shape[1], self.seq_len):
            is_image = cur_len >= self.text_seq_len

            text, image = F.split(out, nsplits_or_sections=[
                                  self.text_seq_len], axis=1)

            logits = self.forward_with_cond_scale(
                text, image, cond_scale=cond_scale)
            logits = logits[:, -1, :]

            filtered_logits = topk(logits, thread=filter_thread)
            sample = gumbel_sample(
                filtered_logits, temperature=temperature, axis=-1)

            # offset sampled token if it is an image token,
            # since logit space is composed of text and then image tokens
            if is_image:
                sample -= self.num_text_tokens

            out = F.concat([out, sample[:, None]], axis=-1)

        text_seq = out[:, :self.text_seq_len]
        img_seq = out[:, -self.image_seq_len:]
        img_seq[img_seq < 0] = 0

        images = self.vae.decode(img_seq)

        if clip is not None:
            scores = clip(text_seq, images)[0]
            return images, scores
        return images

    def forward(
        self,
        text: Tensor,
        image: Tensor = None,
        return_loss: bool = False,
        null_cond_prob: float = 0.
    ):
        if text.shape[-1] != self.text_seq_len:
            raise ValueError(
                f"expected length of input token be equal to {self.text_seq_len}, but got {text.shape[-1]}")  # noqa: E501
        if self.training and image is None:
            raise RuntimeError("`image` must be supplied when training.")
        # randomly remove text condition with <null_cond_prob> probability
        if self.training and null_cond_prob > 0.:
            null_mask = mge.random.uniform(
                0, 1, (text.shape[0], )) >= null_cond_prob
            text *= F.expand_dims(null_mask, axis=1)

        # make sure padding in text tokens get unique padding token id
        text_range = F.arange(self.text_seq_len) + \
            (self.num_text_tokens - self.text_seq_len)
        text = F.where(text == 0, text_range, text).astype('int32')

        # add <bos>

        text = F.pad(text, [*[(0, 0) for _ in range(text.ndim - 1)], (1, 0)])

        tokens = self.text_emb(text)
        if self.text_pos_emb is not None:
            tokens += self.text_pos_emb(F.arange(text.shape[-1],  # pylint: disable=not-callable
                                        dtype='int32'))

        seq_len = tokens.shape[1]

        if image is not None:
            is_raw_image = image.ndim == 4

            if is_raw_image:
                image_size = self.vae.image_size
                channels = self.vae.channels
                source_shape = image.shape[1:]
                target_shape = (channels, image_size, image_size)
                if source_shape != target_shape:
                    raise ValueError(
                        f"Invalid image size, expected {target_shape} but got {source_shape}")

                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            if self.image_pos_emb is not None:
                image_emb += self.image_pos_emb(image_emb)  # pylint: disable=not-callable

            tokens = F.concat([tokens, image_emb], axis=1)

            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > self.seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        out = self.transformer(tokens)
        out = self.norm_by_max(out)

        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        logits_mask = self.logits_mask[:, :seq_len]
        inf = F.full_like(logits, value=-float('Inf'))
        logits = F.where(logits_mask, inf, logits)

        if not return_loss or self.loss_img_weight is None:
            return logits

        offsetted_image = image + self.num_text_tokens
        labels = F.concat([text[:, 1:], offsetted_image], axis=1)

        logits = logits.transpose(0, 2, 1)

        loss_text = F.cross_entropy(
            logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(
            logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

        loss = (loss_text + self.loss_img_weight
                * loss_img) / (self.loss_img_weight + 1)
        return loss

    def forward_with_cond_scale(self, *args, cond_scale: float = 1., **kwargs):
        if cond_scale == 1.:
            return self(*args, **kwargs)

        logits = self(*args, **kwargs)

        null_cond_logits = self(*args, null_cond_prob=1, **kwargs)
        return null_cond_logits + (logits - null_cond_logits) * cond_scale
