from typing import Union

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .functional import gumbel_softmax


class BaseQuantizer(M.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        remap=None,
        unknown_index: Union[str, int] = "random",
    ):
        super(BaseQuantizer, self).__init__()

        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.embedding = M.Embedding(num_embeddings, embed_dim)

        self.remap = remap
        if remap is not None:
            self.used = mge.tensor(np.load(remap))
            self.remap_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.remap_embed
                self.remap_embed = self.remap_embed + 1
            print(f"Remapping {num_embeddings} indices to {self.re_emremap_embedbed} indices. "  # pylint: disable=no-member  # noqa: E501
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.remap_embed = num_embeddings

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        match = (inds[:, :, None] == self.used[None, None, ...]).long()
        new = F.argmax(match, axis=-1)
        unknown = F.sum(match, axis=2) < 1
        if self.unknown_index == "random":
            new[unknown] = mge.tensor(np.random.randint(
                0, self.remap_embed, size=new[unknown].shape))
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        if self.remap_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = F.gather(self.used[None, :][inds.shape[0]
                        * [0], :], axis=1, index=inds)
        return back.reshape(ishape)

    def forward(self, inputs):
        raise NotImplementedError()


class VectorQuantizer(BaseQuantizer):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py  # noqa: E501
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: Union[str, int] = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super(VectorQuantizer, self).__init__(
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            remap=remap,
            unknown_index=unknown_index)

        self.beta = beta
        self.legacy = legacy

        M.init.uniform_(self.embedding.weight, -1.0
                        / num_embeddings, 1.0 / num_embeddings)

        self.sane_index_shape = sane_index_shape

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert not rescale_logits, "Only for interface compatible with Gumbel"
        assert not return_logits, "Only for interface compatible with Gumbel"
        # B, H, W, C
        z = z.transpose(0, 2, 3, 1)
        # z_flattend = F.flatten(z, 1)
        z_flattend = z.reshape(-1, self.embed_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = F.sum(z_flattend ** 2, axis=1, keepdims=True) + \
            F.sum(self.embedding.weight ** 2, axis=1) - 2 * \
            z_flattend @ self.embedding.weight.transpose(1, 0)

        # find closest encodings
        min_encoding_indices = F.argmin(d, axis=1)

        # get quantized latent vectors
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)

        # compute loss for embedding
        if self.legacy:
            loss = F.mean((z_q.detach() - z)**2) + self.beta * \
                F.mean((z_q - z.detach())**2)
        else:
            loss = self.beta * F.mean((z_q.detach() - z)**2) + \
                F.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape to original shape B, C, H, W
        z_q = z_q.transpose(0, 3, 1, 2)

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            # flatten
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = F.flatten(indices)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)
        print(z_q.shape)
        if shape is not None:
            z_q = z_q.reshape(shape)
            # reshape back to match original input shape
            z_q = z_q.transpose(0, 3, 1, 2)

        return z_q


class GumbelQuantizer(BaseQuantizer):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(
        self,
        hidden_dim: int,
        num_embeddings: int,
        embed_dim: int,
        straight_through: bool = False,
        kl_weight: float = 5e-4,
        temperature: float = 1.0,
        use_vqinterface=True,
        remap=None,
        unknown_index: Union[str, int] = "random",
    ):
        super(GumbelQuantizer, self).__init__(
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            remap=remap,
            unknown_index=unknown_index)

        self.straight_through = straight_through
        self.use_vqinterface = use_vqinterface
        self.temperature = temperature
        self.kl_weight = kl_weight

        self.proj = M.Conv2d(hidden_dim, num_embeddings, 1)

    def get_codebook_entry(self, indices, shape):
        b, h, w, _ = shape
        assert b * h * w == indices.shape[0]
        indices = indices.reshape(b, h, w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).transpose(
            0, 3, 1, 2).astype('float32')
        z_q = (one_hot.transpose(0, 2, 3, 1) @
               self.embedding.weight).transpose(0, 3, 1, 2)
        return z_q

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize.
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        logits = self.proj(z)

        if self.remap is not None:
            # continue only with used logits
            full_zeros = F.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = gumbel_softmax(logits, tau=temp, axis=1, hard=hard)

        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = (soft_one_hot.transpose(0, 2, 3, 1) @
               self.embedding.weight).transpose(0, 3, 1, 2)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, axis=1)
        diff = self.kl_weight * \
            F.sum(qy * F.log(qy * self.num_embeddings + 1e-10), axis=1).mean()

        ind = F.argmax(soft_one_hot, axis=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind
