import megengine.functional as F
import megengine.module as M

logit_laplace_eps: float = 0.1


def map_pixels(x):
    if x.ndim != 4:
        raise ValueError('input must be 4D')
    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x):
    if x.ndim != 4:
        raise ValueError('input must be 4D')
    return F.clip((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)


class Upsample(M.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        return F.nn.interpolate(inputs, scale_factor=self.scale_factor, mode=self.mode)
