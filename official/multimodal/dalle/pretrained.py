from megengine import hub

from .dalle import DALLE
from .vae.vqgan_vae import vqgan_vae_1024


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/dalle_coco_512_16_16d_16h_80tsl.pkl"
)
def coco_512_16_16d_16h_80tsl():
    vae = vqgan_vae_1024(False)
    model = DALLE(
        num_text_tokens=8192,
        text_seq_len=80,
        embed_dim=512,
        vae=vae,
        num_heads=16,
        head_dim=64,
        stable=False,
        depths=16,
        attention_types=['row', 'row', 'column', 'row', 'row', 'row', 'column', 'full']
    )
    return model
