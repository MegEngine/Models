import math
from functools import update_wrapper

import numpy as np

import megengine as mge
import megengine.functional as F


def sinc(x):
    return F.where(x != 0, F.sin(math.pi * x) / (math.pi * x), F.ones_like(x))


def lanczos(x, a):
    cond = F.logical_and(-a < x, x < a)
    out = F.where(cond, sinc(x) * sinc(x / a), F.zeros_like(x))
    return out / F.sum(out)


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = np.zeros(n)
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    out = np.concatenate([np.flip(-out[1:], axis=0), out])[1:-1]
    return mge.tensor(out, dtype='float32')


def odd(fn):
    return update_wrapper(lambda x: F.sin(x) * fn(F.abs(x)), fn)


def _to_linear_srgb(input):
    cond = input <= 0.04045
    a = input / 12.92
    b = ((input + 0.055) / 1.055)**2.4
    return F.where(cond, a, b)


def _to_nonlinear_srgb(input):
    cond = input <= 0.0031308
    a = 12.92 * input
    b = 1.055 * input**(1 / 2.4) - 0.055
    return F.where(cond, a, b)


to_linear_srgb = odd(_to_linear_srgb)
to_nonlinear_srgb = odd(_to_nonlinear_srgb)


def resample(input, size, align_corners=True, is_srgb=False):  # pylint: disable=unused-argument
    n, c, h, w = input.shape
    dh, dw = size

    if is_srgb:
        input = to_linear_srgb(input)

    input = input.reshape(n * c, 1, h, w)

    if dh < h:
        kernel_h = lanczos(
            ramp(dh / h, 3), 3).to(input.device).astype(input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(
            input, [(0, 0), (0, 0), (pad_h, pad_h), (0, 0)], 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(
            ramp(dw / w, 3), 3).to(input.device).astype(input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, [(0, 0), (0, 0), (0, 0),
                      (pad_w, pad_w)], 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape(n, c, h, w)
    # NOTE: can not set align_corners when specify mode with `bicubic` in megengine
    input = F.nn.interpolate(input, size, mode='bicubic',
                             align_corners=None)

    if is_srgb:
        input = to_nonlinear_srgb(input)

    return input
