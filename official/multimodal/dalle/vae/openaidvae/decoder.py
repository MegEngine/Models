from collections import OrderedDict
from functools import partial

import megengine.module as M
from megengine import hub

from .utils import Upsample


class DecoderBlock(M.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        layers,
    ) -> None:
        super(DecoderBlock, self).__init__()
        assert out_channels % 4 == 0, "The output channel must be devided into 4"
        self.post_gain = 1 / (layers ** 2)
        hid_ch = out_channels // 4
        self.id_path = M.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else M.Identity()
        self.res_path = M.Sequential(OrderedDict([
            ("relu1", M.ReLU()),
            ('conv_1', M.Conv2d(in_channels, hid_ch, 1)),
            ("relu2", M.ReLU()),
            ('conv_2', M.Conv2d(hid_ch, hid_ch, 3, padding=1)),
            ("relu3", M.ReLU()),
            ('conv_3', M.Conv2d(hid_ch, hid_ch, 3, padding=1)),
            ("relu4", M.ReLU()),
            ('conv_4', M.Conv2d(hid_ch, out_channels, 3, padding=1)),
        ]))

    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Decoder(M.Module):
    def __init__(self, n_init=128, n_hid=256, n_blk_per_group=2, out_ch=3, vocab_size=8192):
        super(Decoder, self).__init__()
        group_count = 4
        n_layers = group_count * n_blk_per_group
        blk_range = range(n_blk_per_group)
        make_blk = partial(DecoderBlock, layers=n_layers)
        self.vocab_size = vocab_size
        self.blocks = M.Sequential(OrderedDict([
            ('input', M.Conv2d(vocab_size, n_init, 1)),
            ('group_1', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(n_init if i == 0 else 8
                                              * n_hid, 8 * n_hid)) for i in blk_range],
                ('upsample', Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_2', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(8 * n_hid if i
                                              == 0 else 4 * n_hid, 4 * n_hid)) for i in blk_range],
                ('upsample', Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_3', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * n_hid if i
                                              == 0 else 2 * n_hid, 2 * n_hid)) for i in blk_range],
                ('upsample', Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_4', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * n_hid if i
                                              == 0 else 1 * n_hid, 1 * n_hid)) for i in blk_range],
            ]))),
            ('output', M.Sequential(OrderedDict([
                ('relu', M.ReLU()),
                ('conv', M.Conv2d(1 * n_hid, 2 * out_ch, 1)),
            ]))),
        ]))

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("The input must be 4-dim")
        if x.shape[1] != self.vocab_size:
            raise ValueError(
                "The input must be the same shape as the vocab")
        # if x.dtype != "float32":
        #     raise ValueError("The input must be float32")
        return self.blocks(x)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/dalle_openai_dvae_decoder.pkl"
)
def openai_discrete_VAE_decoder(**kwargs):
    return Decoder(**kwargs)
