from .dalle import DALLE
from .generate import Generator
from .pretrained import coco_512_16_16d_16h_80tsl
from .vae import (
    OpenAIDiscreteVAE,
    OpenAIDiscreteVAEDecoder,
    OpenAIDiscreteVAEEncoder,
    VQGanVAE,
    openai_discrete_VAE_decoder,
    openai_discrete_VAE_encoder
)
from .vae.vqgan_vae import vqgan_vae_1024
