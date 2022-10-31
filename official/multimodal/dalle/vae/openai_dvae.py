import math

import megengine.functional as F

from .base_vae import BaseVAE
from .openaidvae import openai_discrete_VAE_decoder, openai_discrete_VAE_encoder
from .openaidvae.utils import map_pixels, unmap_pixels


class DiscreteVAE(BaseVAE):
    def __init__(
        self,
        pretrained: bool = True
    ):
        super(DiscreteVAE, self).__init__(
            num_layers=3,
            num_tokens=8192,
            image_size=256,
        )

        self.encoder = openai_discrete_VAE_encoder(pretrained=pretrained)
        self.decoder = openai_discrete_VAE_decoder(pretrained=pretrained)

    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.encoder.blocks(img)
        z = F.argmax(z_logits, axis=1)
        z = F.flatten(z, 1)
        return z

    def decode(self, img_seq):
        b, n, = img_seq.shape
        L = int(math.sqrt(n))
        img_seq = img_seq.reshape(b, L, L)

        z = F.one_hot(img_seq, num_classes=self.num_tokens)

        z = z.transpose(0, 3, 1, 2).astype('float32')
        x_stats = self.decoder(z).astype('float32')
        x_rec = unmap_pixels(F.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, inputs):
        raise NotImplementedError("Do not call forward method!")
