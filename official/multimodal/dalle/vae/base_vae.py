from abc import abstractmethod

import megengine.module as M


class BaseVAE(M.Module):
    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        image_size: int,
        channels: int = 3,
    ):
        super(BaseVAE, self).__init__()

        self.channels = channels
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.image_size = image_size

    @abstractmethod
    def get_codebook_indices(self, inputs):
        pass

    @abstractmethod
    def decode(self, inputs):
        pass

    def forward(self, inputs):
        raise NotImplementedError()
