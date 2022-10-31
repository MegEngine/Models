from typing import Optional

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub

from .diffusion_modules import Decoder, Encoder
from .quantize import GumbelQuantizer, VectorQuantizer


class BaseVQModel(M.Module):
    def __init__(
        self,
        diffusion_config: dict,
        embed_dim: int,
        colorize_nlabels: Optional[int] = None,
        task_type: str = "image",
    ):

        super(BaseVQModel, self).__init__()
        self.task_type = task_type

        self.encoder = Encoder(**diffusion_config)
        self.decoder = Decoder(**diffusion_config)

        self.in_resolution = diffusion_config['in_resolution']
        self.attn_resolution = diffusion_config['attention_resolutions']
        self.in_channel = diffusion_config['in_channel']

        if colorize_nlabels is not None or task_type == 'segmentation':
            colorize_nlabels = diffusion_config['out_channel']
            self.colorize = mge.random.normal(size=(3, colorize_nlabels, 1, 1))

        self.quant_conv = M.Conv2d(
            diffusion_config['z_channel'], embed_dim, 1)
        self.post_quant_conv = M.Conv2d(
            embed_dim, diffusion_config['z_channel'], 1)

        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)  # pylint: disable=no-member
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)  # pylint: disable=no-member
        dec = self.decode(quant_b)
        return dec

    def forward(self, x):
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def to_rgb(self, x):
        assert self.task_type == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.colorize = mge.random.normal(
                size=(3, self.colorize_nlabels, 1, 1))  # pylint: disable=no-member
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - F.min(x)) / (F.max(x) - F.min(x)) - 1.0
        return x


class VQModel(BaseVQModel):
    def __init__(
        self,
        diffusion_config: dict,
        num_embeddings: int,
        embed_dim: int,
        remap=None,
        sane_index_shape: bool = False,
        colorize_nlabels: Optional[int] = None,
        task_type: str = "image",
    ):
        super(VQModel, self).__init__(
            diffusion_config,
            embed_dim,
            colorize_nlabels,
            task_type,
        )

        self.quantize = VectorQuantizer(
            num_embeddings,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape
        )


class GumbelVQ(BaseVQModel):
    def __init__(
        self,
        diffusion_config: dict,
        num_embeddings: int,
        embed_dim: int,
        kl_weight: float = 1e-8,
        remap=None,
        colorize_nlabels: Optional[int] = None,
        task_type: str = "image",
    ):
        z_channel = diffusion_config["z_channel"]
        super(GumbelVQ, self).__init__(
            diffusion_config,
            embed_dim,
            colorize_nlabels,
            task_type,
        )

        self.vocab_size = num_embeddings

        self.quantize = GumbelQuantizer(
            hidden_dim=z_channel,
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            kl_weight=kl_weight,
            temperature=1.0,
            remap=remap
        )

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError("Do not call `decode_code` method!")


def _vqgan(num_embeddings, embed_dim, model_type='vqmodel', **kwargs):
    diffusion_config = dict(
        in_resolution=256,
        in_channel=3,
        z_channel=256,
        out_channel=3,
        base_channel=128,
        channel_multiplier=[1., 1., 2., 2., 4.],
        attention_resolutions=[16],
        num_res_blocks=2,
        double_z=False,
        dropout=0.0,
    )
    for k, v in kwargs.items():
        if k in diffusion_config:
            diffusion_config[k] = v
    if model_type == 'vqmodel':
        model = VQModel(
            diffusion_config=diffusion_config,
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
        )
    else:
        model = GumbelVQ(
            diffusion_config=diffusion_config,
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
        )
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/vqgan_imagenet_f16_1024.pkl"
)
def vqgan_imagenet_f16_1024(**kwargs):
    return _vqgan(1024, 256, **kwargs)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/vqgan_imagenet_f16_16384.pkl"
)
def vqgan_imagenet_f16_16384(**kwargs):
    return _vqgan(16384, 256, **kwargs)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/vqgan_gumbel_f8.pkl"
)
def vqgan_gumbel_f8(**kwargs):
    kwargs['channel_multiplier'] = [1., 1., 2., 4.]
    kwargs['attention_resolutions'] = [32]
    return _vqgan(8192, 256, model_type='gumbel', **kwargs)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/vqgan_openimages_f8_256.pkl"
)
def vqgan_openimages_f8_256(**kwargs):
    kwargs['channel_multiplier'] = [1., 2., 2., 4.]
    kwargs['attention_resolutions'] = [32]
    kwargs['z_channel'] = 4
    return _vqgan(256, 4, **kwargs)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/vqgan_openimages_f8_16384.pkl"
)
def vqgan_openimages_f8_16384(**kwargs):
    kwargs['channel_multiplier'] = [1., 2., 2., 4.]
    kwargs['attention_resolutions'] = [32]
    kwargs['z_channel'] = 4
    return _vqgan(16384, 4, **kwargs)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/vqgan_gumbel_openimages_f8.pkl"
)
def vqgan_gumbel_openimages_f8(**kwargs):
    kwargs['channel_multiplier'] = [1., 1., 2., 4.]
    kwargs['attention_resolutions'] = [32]

    return _vqgan(8192, 256, model_type='gumbel', **kwargs)
