import copy
from typing import Optional

import megengine as mge
import megengine.functional as F
from megengine import Tensor
from megengine.functional import normalize
from megengine.module import ConvTranspose2d, ConvTranspose3d, Module


class SpectralNorm():

    def __init__(
        self,
        name: str = 'weight',
        n_power_iterations: int = 1,
        axis: int = 0,
        eps: float = 1e-12
    ) -> None:
        self.name = name
        self.axis = axis
        if n_power_iterations <= 0:
            raise ValueError(
                "`n_power_iterations` must be positive, but got {}".format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        weight_mat = weight
        if self.axis != 0:
            weight_mat = weight_mat.transpose(
                self.axis, *[d for d in range(weight_mat.dim()) if d != self.axis])
        height = weight_mat.shape[0]
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> Tensor:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            for _ in range(self.n_power_iterations):
                v.set_value(normalize(F.matmul(
                    weight_mat, u, transpose_a=True, transpose_b=False), axis=0, eps=self.eps))
                u.set_value(
                    normalize(F.matmul(weight_mat, v), axis=0, eps=self.eps))
            if self.n_power_iterations > 0:
                u = copy.deepcopy(u)
                v = copy.deepcopy(v)
        sigma = F.dot(u, F.matmul(weight_mat, v))
        weight = weight / sigma
        return weight

    def __call__(self, module: Module, inputs: Tensor) -> None:
        setattr(module, self.name, self.compute_weight(
            module, do_power_iteration=module.training))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, axis: int, eps: float):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, axis, eps)
        weight = getattr(module, name)

        weight_mat = fn.reshape_weight_to_matrix(weight)
        h, w = weight_mat.shape
        u = mge.Parameter(mge.random.normal(0., 1., size=[h]))
        v = mge.Parameter(mge.random.normal(0., 1., size=[w]))
        u = normalize(u, axis=0, eps=fn.eps)
        v = normalize(v, axis=0, eps=fn.eps)

        module.__delattr__(fn.name)

        module.__setattr__(fn.name + '_orig', copy.deepcopy(weight))
        # just del it, do not assgin back for now
        # module.__setattr__(fn.name, weight * 1.0)
        module.__setattr__(fn.name + '_u', u.detach())
        module.__setattr__(fn.name + '_v', v.detach())
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(
    module: Module,
    name: str = 'weight',
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    axis: Optional[int] = None
) -> Module:
    if axis is None:
        if isinstance(module, (ConvTranspose2d, ConvTranspose3d)):
            axis = 1
        else:
            axis = 0
    SpectralNorm.apply(module, name, n_power_iterations, axis, eps)
    return module


def remove_spectral_norm(module: Module, name: str = 'weight') -> Module:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("spectral_norm of '{}' not found in {}".format(
            name, module))

    return module
