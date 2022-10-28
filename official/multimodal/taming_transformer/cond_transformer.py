from typing import Union

import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine import Tensor, hub

from .mingpt import GPT, GPTConfig, multinomial
from .vqgan import GumbelVQ, VQModel, _vqgan


class PremuterIdentity(M.Module):
    def forward(self, x, reverse=True):  # pylint: disable=unused-argument
        return x


class SOSProvider(M.Module):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = F.ones((x.shape[0], 1)) * self.sos_token
        c = c.astype('int64')
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c

    def forward(self, x):
        raise NotImplementedError()


class CoordStage():
    def __init__(self, num_embeddings, down_factor):
        self.num_embeddings = num_embeddings
        self.down_factor = down_factor

    def eval(self):
        return self

    def encode(self, c):
        """fake vqmodel interface"""
        assert c.min() >= 0.0 and c.max() <= 1.0
        ch = c.shape[1]
        assert ch == 1

        c = F.nn.interpolate(c, scale_factor=1 / self.down_factor, mode="area")
        c = F.clip(c, 0.0, 1.0)
        c = self.num_embeddings * c
        c_quant = F.round(c)
        c_ind = c_quant.astype('int64')

        info = None, None, c_ind
        return c_quant, None, info

    def decode(self, c):
        c = c / self.num_embeddings
        c = F.nn.interpolate(c, scale_factor=self.down_factor, mode="nearest")
        return c


class Net2NetTransformer(M.Module):
    def __init__(
        self,
        transformer: GPT,
        first_stage_model: Union[VQModel, GumbelVQ],
        cond_stage_model: Union[VQModel, GumbelVQ, SOSProvider],
        downsample_cond_size: int = -1,
        keep_prob: float = 1.0,
        sos_token: float = 0.0,
        conditional: bool = True,
    ):
        super(Net2NetTransformer, self).__init__()
        self.conditional = conditional
        self.sos_token = sos_token

        self.transformer = transformer
        self.first_stage_model = first_stage_model
        self.cond_stage_model = cond_stage_model
        self.premuter = PremuterIdentity()

        self.downsample_cond_size = downsample_cond_size
        self.keep_prob = keep_prob
        self.vocab_size = self.transformer.config.vocab_size

    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].reshape(quant_z.shape[0], -1)
        indices = self.premuter(indices)
        return quant_z, indices

    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.nn.interpolate(c, size=2 * (self.downsample_cond_size, ))
        quant_c, _, [_, _, indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.reshape(c.shape[0], -1)
        return quant_c, indices

    def decode_to_img(self, index, zshape):
        index = self.premuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    def forward(self, x):
        z_indices = self.encode_to_z(x)[1]
        c_indices = self.encode_to_c(x)[1]

        if self.training and self.keep_prob < 1.0:
            mask = F.dropout(F.ones_like(z_indices),
                             drop_prob=1 - self.keep_prob).astype('int64')
            r_indices = Tensor(np.random.randint(
                low=0, high=self.vocab_size, size=z_indices.shape, dtype='int32'))
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices

        cz_indices = F.concat([c_indices, a_indices], axis=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]

        return logits, target

    def top_k_logits(self, logits, k):
        values, _ = F.topk(logits, k, descending=True)
        out = F.where(
            logits < values[..., [-1]], F.full_like(logits, value=-float('Inf')), logits)
        return out

    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None):
        x = F.concat([c, x], axis=1)
        if self.transformer.training:
            raise RuntimeError("set transformer to eval before sample")

        if self.keep_prob <= 0.0:
            # one pass suffices since input is pure noise anyway
            if len(x.shape) != 2:
                raise ValueError("`x` must be 2 dimension")
            noise = c[:, x.shape[1] - c.shape[1]: -1]
            x = F.concat([x, noise], axis=1)
            # take all logits for now and scale by temp
            logits = self.transformer(x)[0] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, axis=-1)
            # sample from the distribution or take the most likely
            if sample:
                ori_shape = probs.shape
                probs = F.flatten(probs, 1)
                indices = multinomial(probs, num_samples=1)
                probs = probs.reshape(*ori_shape)
                indices = indices.reshape(ori_shape[:2])
            else:
                indices = F.argmax(probs, axis=-1)
            # cut off conditioning
            x = indices[:, c.shape[1] - 1:]
        else:
            block_size = self.transformer.get_block_size()
            for _ in range(steps):
                if x.shape[1] > block_size:
                    raise ValueError(
                        'The first dimension of `x` must be less than block size of transformer.')

                x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
                # pluck the logits at the final step and scale by temperature
                logits = self.transformer(x_cond)[0] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, axis=-1)
                # sample from the distribution or take the most likely
                if sample:
                    indices = multinomial(probs, num_samples=1)
                else:
                    indices = F.argmax(probs, axis=-1)

                x = F.concat([x, indices], axis=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/taming_s_flckr.pkl"
)
def s_flckr_transformer():
    gpt_config = GPTConfig(
        vocab_size=1024,
        block_size=512,
        n_layer=24,
        n_head=16,
        n_embed=1024,
    )
    first_stage_model = _vqgan(1024, 256)
    cond_stage_model = _vqgan(
        1024, 256, in_channel=182, out_channel=182, task_type='segmentation')
    gpt = GPT(gpt_config)
    model = Net2NetTransformer(
        gpt,
        first_stage_model,
        cond_stage_model,
        conditional=True,
    )
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/taming_celebahq.pkl"
)
def celebahq_transformer():
    gpt_config = GPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=24,
        n_head=16,
        n_embed=1664,
    )
    first_stage_model = _vqgan(1024, 256)
    cond_stage_model = SOSProvider(0)
    cond_stage_model.task_type = first_stage_model.task_type
    gpt = GPT(gpt_config)
    model = Net2NetTransformer(
        gpt,
        first_stage_model,
        cond_stage_model,
        conditional=False,
    )
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/taming_drin.pkl"
)
def drin_transformer():
    gpt_config = GPTConfig(
        vocab_size=1024,
        block_size=512,
        n_layer=24,
        n_head=16,
        n_embed=1024,
    )
    first_stage_model = _vqgan(1024, 256)
    cond_stage_model = _vqgan(1024, 256, in_channel=1, out_channel=1, task_type='depth')
    gpt = GPT(gpt_config)
    model = Net2NetTransformer(
        gpt,
        first_stage_model,
        cond_stage_model,
        conditional=True,
    )
    return model


def ffhq_transformer():
    gpt_config = GPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=24,
        n_head=16,
        n_embed=1664,
    )
    first_stage_model = _vqgan(1024, 256)
    cond_stage_model = SOSProvider(0)
    cond_stage_model.task_type = first_stage_model.task_type
    gpt = GPT(gpt_config)
    model = Net2NetTransformer(
        gpt,
        first_stage_model,
        cond_stage_model,
        conditional=False,
    )
    return model


def ade20k_transformer():
    gpt_config = GPTConfig(
        vocab_size=4096,
        block_size=512,
        n_layer=28,
        n_head=16,
        n_embed=1024,
        resid_drop=0.1,
        attn_drop=0.1,
    )
    first_stage_model = _vqgan(4096, 256)
    cond_stage_model = _vqgan(
        1024, 256, image_key='segmentation', in_channel=151, out_channel=151)
    gpt = GPT(gpt_config)
    model = Net2NetTransformer(
        gpt,
        first_stage_model,
        cond_stage_model,
        conditional=True,
    )
    return model


def coco_transformer():
    gpt_config = GPTConfig(
        vocab_size=8192,
        block_size=512,
        n_layer=32,
        n_head=16,
        n_embed=1280,
    )
    first_stage_model = _vqgan(8192, 256)
    cond_stage_model = _vqgan(
        1024, 256, image_key='segmentation', in_channel=183, out_channel=183)
    gpt = GPT(gpt_config)
    model = Net2NetTransformer(
        gpt,
        first_stage_model,
        cond_stage_model,
        conditional=True,
    )
    return model
