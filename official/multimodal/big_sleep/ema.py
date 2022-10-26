# Exponential Moving Average (from https://gist.github.com/crowsonkb/76b94d5238272722290734bf4725d204)  # noqa: E501
from copy import deepcopy

import megengine as mge
import megengine.functional as F
import megengine.module as M


class EMA(M.Module):
    def __init__(self, model: M.Module, decay: float):
        super(EMA, self).__init__()
        self.model = model
        self.decay = decay
        self.accum = mge.tensor(1.)

        self._biased = deepcopy(model)
        self.average = deepcopy(model)
        for param in self._biased.parameters():
            param.set_value(param.detach() * 0)
        for param in self.average.parameters():
            param.set_value(param.detach() * 0)
        self.update()

    def update(self):
        if not self.training:
            raise RuntimeError('Update should only be called during training')

        self.accum *= self.decay

        model_params = dict(self.model.named_parameters())
        biased_params = dict(self._biased.named_parameters())
        average_params = dict(self.average.named_parameters())
        assert model_params.keys() == biased_params.keys() == average_params.keys(
        ), 'Model parameter keys incompatible with EMA stored parameter keys'

        for name, param in model_params.items():
            biased_params[name].set_value(
                F.mul(biased_params[name], self.decay))
            biased_params[name].set_value(
                F.add(biased_params[name], (1 - self.decay) * param))
            average_params[name].set_value(biased_params[name])
            average_params[name].set_value(
                F.div(average_params[name], 1 - self.accum))

        model_buffers = dict(self.model.named_buffers())
        biased_buffers = dict(self._biased.named_buffers())
        average_buffers = dict(self.average.named_buffers())
        assert model_buffers.keys() == biased_buffers.keys() == average_buffers.keys()

        for name, buffer in model_buffers.items():
            biased_buffers[name].set_value(buffer)
            average_buffers[name].set_value(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.average(*args, **kwargs)
