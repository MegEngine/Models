from math import log, sqrt
from typing import Union

import megengine.functional as F

from ...taming_transformer.vqgan import GumbelVQ, VQModel, vqgan_imagenet_f16_1024
from .base_vae import BaseVAE


class VQGanVAE(BaseVAE):
    def __init__(self, model: Union[VQModel, GumbelVQ]):
        image_size = model.in_resolution
        num_layers = int(log(image_size / model.attn_resolution[0]) / log(2))
        channels = model.in_channel
        num_tokens = model.quantize.num_embeddings

        super(VQGanVAE, self).__init__(
            num_layers,
            num_tokens,
            image_size,
            channels
        )
        self.model = model

        self.is_gumbel = isinstance(model, GumbelVQ)

    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return F.flatten(indices, 1)
        return indices.reshape(b, -1)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes=self.num_tokens).astype('float32')
        z = one_hot_indices @ self.model.quantize.embedding.weight

        c = z.shape[-1]
        z = z.reshape(b, int(sqrt(n)), -1, c).transpose(0, 3, 1, 2)
        img = self.model.decode(z)

        img = (F.clip(img, -1., 1.) + 1) * 0.5
        return img

    def forward(self):
        raise NotImplementedError()


def vqgan_vae_1024(pretrained=True):
    vae = vqgan_imagenet_f16_1024(pretrained=pretrained)
    model = VQGanVAE(vae)
    return model
