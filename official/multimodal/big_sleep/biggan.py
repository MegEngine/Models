import copy
import json
import math
from functools import partial
from typing import Optional, Sequence, Union

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub

from .spectral_norm import spectral_norm


def broadcat_to_batch(x):
    idx = len(x.shape) - 1
    axis = (0, 2, 3)
    return F.expand_dims(x, axis=axis[idx:])


class BigGANConfig():

    def __init__(
        self,
        layers: Optional[Sequence[Sequence[Union[int, bool]]]] = None,
        output_dim: int = 128,
        z_dim: int = 128,
        class_embed_dim: int = 128,
        base_channel: int = 128,
        num_classes: int = 1000,
        attention_layer_position: int = 8,
        eps: float = 1e-4,
        n_stats: int = 51
    ):
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.base_channel = base_channel
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SelfAttention(M.Module):

    def __init__(
        self,
        in_channels: int,
        reduction: float = 8.0,
        out_reduction: float = 2.0,
        eps: float = 1e-12
    ):
        super(SelfAttention, self).__init__()
        inner_chan = int(in_channels // reduction)
        out_chan = int(in_channels // out_reduction)
        sn = partial(spectral_norm, eps=eps)

        self.theta = sn(M.Conv2d(in_channels, inner_chan, 1, bias=False))
        self.phi = sn(M.Conv2d(in_channels, inner_chan, 1, bias=False))
        self.g = sn(M.Conv2d(in_channels, out_chan, 1, bias=False))
        self.out = sn(M.Conv2d(out_chan, in_channels, 1, bias=False))
        self.maxpool = M.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.softmax = M.Softmax(axis=-1)
        self.gamma = mge.Parameter(F.zeros((1)))

        self.inner_chan = in_channels
        self.out_chan = out_chan

    @staticmethod
    def flatten_spatial(x):
        return F.flatten(x, 2)

    def forward(self, x):
        B, _, H, W = x.shape

        theta = self.theta(x)
        theta = self.flatten_spatial(theta)

        phi = self.phi(x)
        phi = self.maxpool(phi)
        phi = self.flatten_spatial(phi)

        attn = theta.transpose(0, 2, 1) @ phi
        attn = self.softmax(attn)

        g = self.g(x)
        g = self.maxpool(g)
        g = self.flatten_spatial(g)
        attn_g = g @ attn.transpose(0, 2, 1)
        attn_g = attn_g.reshape(B, self.out_chan, H, W)
        attn_g = self.out(attn_g)

        out = x + self.gamma * attn_g
        return out


class BigGANBatchNorm(M.Module):

    def __init__(
        self,
        num_features: int,
        condition_vector_dim: Optional[int] = None,
        n_stats: int = 51,
        eps: float = 1e-4,
        conditional: bool = True,
    ):
        super(BigGANBatchNorm, self).__init__()

        self.eps = eps
        self.conditional = conditional

        sn = partial(spectral_norm, eps=eps)

        self.running_means = F.zeros((n_stats, num_features))
        self.running_vars = F.ones((n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)

        if conditional:
            if not isinstance(condition_vector_dim, int):
                raise TypeError(
                    '`condition_vector_dim` must be interger when specifing `conditional`, but got type {}'.format(  # noqa: E501
                        type(condition_vector_dim)
                    ))
            self.scale = sn(M.Linear(condition_vector_dim,
                            num_features, bias=False))
            self.offset = sn(
                M.Linear(condition_vector_dim, num_features, bias=False))
        else:
            self.weight = mge.Parameter(F.ones((num_features)))
            self.bias = mge.Parameter(F.zeros((num_features)))

    def forward(self, x, truncation, condition_vector=None):
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:  # Interpolate
            running_mean = self.running_means[start_idx] * \
                coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + \
                self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]

        running_mean = broadcat_to_batch(running_mean)
        running_var = broadcat_to_batch(running_var)

        if self.conditional:

            weight = 1 + broadcat_to_batch(self.scale(condition_vector))
            bias = broadcat_to_batch(self.offset(condition_vector))

            out = (x - running_mean) / \
                F.sqrt(running_var + self.eps) * weight + bias
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias,
                               training=False, momentum=1.0, eps=self.eps)

        return out


class GeneratorBlock(M.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_vector_dim: int,
        reduction: float = 4.0,
        up_sample: bool = False,
        n_stats: int = 51,
        eps: float = 1e-12,
    ):
        super(GeneratorBlock, self).__init__()
        self.drop_channels = (in_channels != out_channels)

        sn = partial(spectral_norm, eps=eps)
        BatchNorm = partial(BigGANBatchNorm, condition_vector_dim=condition_vector_dim,
                            n_stats=n_stats, eps=eps, conditional=True)

        inner_chan = int(in_channels // reduction)

        self.up_sample = partial(
            F.nn.interpolate, scale_factor=2, mode='nearest') if up_sample else M.Identity()

        self.bn_0 = BatchNorm(in_channels)
        self.conv_0 = sn(M.Conv2d(in_channels, inner_chan, 1))

        self.bn_1 = BatchNorm(inner_chan)
        self.conv_1 = sn(M.Conv2d(inner_chan, inner_chan, 3, padding=1))

        self.bn_2 = BatchNorm(inner_chan)
        self.conv_2 = sn(M.Conv2d(inner_chan, inner_chan, 3, padding=1))

        self.bn_3 = BatchNorm(inner_chan)
        self.conv_3 = sn(M.Conv2d(inner_chan, out_channels, 1))

        self.relu = M.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)

        x = self.up_sample(x)

        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]

        x0 = self.up_sample(x0)

        out = x + x0
        return out


class Generator(M.Module):
    def __init__(self, config: BigGANConfig):
        super(Generator, self).__init__()
        self.config = config
        ch = config.base_channel
        condition_vector_dim = config.z_dim * 2

        sn = partial(spectral_norm, eps=config.eps)

        self.gen_z = sn(M.Linear(condition_vector_dim, 4 * 4 * 16 * ch))

        self.layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                self.layers.append(SelfAttention(ch * layer[1], eps=config.eps))
            self.layers.append(
                GeneratorBlock(
                    ch * layer[1],
                    ch * layer[2],
                    condition_vector_dim,
                    up_sample=layer[0],
                    n_stats=config.n_stats,
                    eps=config.eps
                )
            )

        self.bn = BigGANBatchNorm(
            ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        self.relu = M.ReLU()
        self.conv_to_rgb = sn(M.Conv2d(ch, ch, 3, padding=1))

    def forward(self, cond_vector, truncation):
        z = self.gen_z(F.expand_dims(cond_vector[0], 0))
        # TODO: don't use following conversion
        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.reshape(-1, 4, 4, 16 * self.config.base_channel)
        z = z.transpose(0, 3, 1, 2)

        next_available_latent_index = 1
        for layer in self.layers:
            if isinstance(layer, GeneratorBlock):
                z = layer(
                    z, F.expand_dims(cond_vector[next_available_latent_index], axis=0), truncation)
                next_available_latent_index += 1
            else:
                z = layer(z)

        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = F.tanh(z)
        return z


class BigGAN(M.Module):

    @classmethod
    def from_pretrained(cls, image_size):
        if image_size not in [128, 256, 512]:
            raise ValueError("`image size` must be one of 128, 256, or 512")
        model = _MODELS[f'biggan_{image_size}'](pretrained=True)
        model.eval()
        return model

    def __init__(self, config: BigGANConfig):
        super(BigGAN, self).__init__()
        self.config = config
        self.embeddings = M.Linear(
            config.num_classes, config.z_dim, bias=False)
        self.generator = Generator(config)

    def forward(self, z, class_label, truncation):
        assert 0 < truncation <= 1

        embed = self.embeddings(class_label)
        cond_vector = F.concat([z, embed], axis=1)

        z = self.generator(cond_vector, truncation)
        return z


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/biggan128.pkl"
)
def biggan_128():
    config = BigGANConfig(
        attention_layer_position=8,
        base_channel=128,
        class_embed_dim=128,
        eps=0.0001,
        layers=[
            [False, 16, 16],
            [True, 16, 16],
            [False, 16, 16],
            [True, 16, 8],
            [False, 8, 8],
            [True, 8, 4],
            [False, 4, 4],
            [True, 4, 2],
            [False, 2, 2],
            [True, 2, 1],
        ],
        n_stats=51,
        num_classes=1000,
        output_dim=128,
        z_dim=128,
    )
    return BigGAN(config)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/biggan256.pkl"
)
def biggan_256():
    config = BigGANConfig(
        attention_layer_position=8,
        base_channel=128,
        class_embed_dim=128,
        eps=0.0001,
        layers=[
            [False, 16, 16],
            [True, 16, 16],
            [False, 16, 16],
            [True, 16, 8],
            [False, 8, 8],
            [True, 8, 8],
            [False, 8, 8],
            [True, 8, 4],
            [False, 4, 4],
            [True, 4, 2],
            [False, 2, 2],
            [True, 2, 1],
        ],
        n_stats=51,
        num_classes=1000,
        output_dim=256,
        z_dim=128,
    )
    return BigGAN(config)


@hub.pretrained(
    "https://data.megengine.org.cn/research/multimodality/biggan512.pkl"
)
def biggan_512():
    config = BigGANConfig(
        attention_layer_position=8,
        base_channel=128,
        class_embed_dim=128,
        eps=0.0001,
        layers=[
            [False, 16, 16],
            [True, 16, 16],
            [False, 16, 16],
            [True, 16, 8],
            [False, 8, 8],
            [True, 8, 8],
            [False, 8, 8],
            [True, 8, 4],
            [False, 4, 4],
            [True, 4, 2],
            [False, 2, 2],
            [True, 2, 1],
            [False, 1, 1],
            [True, 1, 1],
        ],
        n_stats=51,
        num_classes=1000,
        output_dim=256,
        z_dim=128,
    )
    return BigGAN(config)


_MODELS = {
    'biggan_128': biggan_128,
    'biggan_256': biggan_256,
    'biggan_512': biggan_512,
}
