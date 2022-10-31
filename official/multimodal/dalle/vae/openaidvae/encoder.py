from collections import OrderedDict
from functools import partial

import megengine.module as M
from megengine import hub


class EncoderBlock(M.Module):
    def __init__(self, n_in, n_out, layers):
        super(EncoderBlock, self).__init__()
        n_hid = n_out // 4
        self.pre_gain = 1 / (layers ** 2)
        self.id_path = M.Conv2d(
            n_in, n_out, 1) if n_in != n_out else M.Identity()
        self.res_path = M.Sequential(OrderedDict([
            ("relu1", M.ReLU()),
            ('conv_1', M.Conv2d(n_in, n_hid, 3, padding=1)),
            ("relu2", M.ReLU()),
            ('conv_2', M.Conv2d(n_hid, n_hid, 3, padding=1)),
            ("relu3", M.ReLU()),
            ('conv_3', M.Conv2d(n_hid, n_hid, 3, padding=1)),
            ("relu4", M.ReLU()),
            ('conv_4', M.Conv2d(n_hid, n_out, 1)),
        ]))

    def forward(self, x):
        return self.id_path(x) + self.pre_gain * self.res_path(x)


class Encoder(M.Module):
    def __init__(self, input_channel=3, n_hid=256, n_blk_per_group=2, vocab_size=8192):
        super(Encoder, self).__init__()
        group_count = 4
        n_layers = group_count * n_blk_per_group
        blk_range = range(n_blk_per_group)
        make_blk = partial(EncoderBlock, layers=n_layers)
        self.input_channel = input_channel
        self.vocab_size = vocab_size
        self.blocks = M.Sequential(OrderedDict([
            ('input', M.Conv2d(input_channel, n_hid, 7, padding=3)),
            ('group_1', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(n_hid, n_hid))
                  for i in blk_range],
                ('pool', M.MaxPool2d(kernel_size=2, stride=2)),
            ]))),
            ('group_2', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(n_hid if i
                                              == 0 else 2 * n_hid, 2 * n_hid)) for i in blk_range],
                ('pool', M.MaxPool2d(kernel_size=2, stride=2)),
            ]))),
            ('group_3', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * n_hid if i
                                              == 0 else 4 * n_hid, 4 * n_hid)) for i in blk_range],
                ('pool', M.MaxPool2d(kernel_size=2, stride=2)),
            ]))),
            ('group_4', M.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * n_hid if i
                   == 0 else 8 * n_hid, 8 * n_hid)) for i in blk_range],
            ]))),
            ('output', M.Sequential(OrderedDict([
                ('relu', M.ReLU()),
                ('conv', M.Conv2d(8 * n_hid, self.vocab_size, 1)),
            ]))),
        ]))

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Input must be 4D tensor")
        if x.shape[1] != self.input_channel:
            raise ValueError(
                f"Input channel must be {self.input_channel}")
        return self.blocks(x)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/dalle_openai_dvae_encoder.pkl"
)
def openai_discrete_VAE_encoder(**kwargs):
    return Encoder(**kwargs)
