import datetime
import os
import random
from copy import deepcopy
from typing import Callable, Optional, Sequence, Union
from tqdm import tqdm

import cv2
import numpy as np

import megengine as mge
import megengine.data.transform as T
import megengine.functional as F
from megengine import Tensor

from .cond_transformer import Net2NetTransformer
from .functional import multinomial
from .mingpt import sample_with_past
from .vqgan import GumbelVQ, VQModel


def scale(x):
    return 2.0 * x - 1.0


def rescale(x):
    return (x + 1.) / 2.


def convert_tensor_to_image(x, restore: Callable = rescale):
    x = F.clip(x, -1.0, 1.0)
    x = restore(x)
    x = x.transpose(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    return x


def convert_segmentation_to_image(seg):
    seg = seg.numpy().transpose(1, 2, 0)[:, :, None, :]
    colorize = np.random.RandomState(1).randn(1, 1, seg.shape[-1], 3)
    colorize = colorize / colorize.sum(axis=2, keepdims=True)
    seg = seg @ colorize
    seg = seg[..., 0, :]
    seg = ((seg + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return seg


def convert_rgba_to_depth(x):
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[2] == 4
    y = x.copy()
    y.dtype = np.float32
    y = y.reshape(x.shape[:2])
    return np.ascontiguousarray(y)


def convert_image_to_tensor(x):
    if x.ndim == 3:
        x = x.transpose(2, 0, 1)
    elif x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    else:
        raise ValueError('Input must have 2 or 3 dimensions')
    x = F.expand_dims(Tensor(x, dtype='float32'), axis=0)
    return x


def preprocess_image(path, callback=None):
    image = cv2.imread(path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if callback is not None:
        image = callback(image)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return convert_image_to_tensor(image)


def preprocess_segmetation(path, n_labels, callback=None):
    x = cv2.imread(path, 0)
    seg = np.eye(n_labels)[x]
    if callback is not None:
        seg = callback(seg)
    return convert_image_to_tensor(seg)


def preprocess_depth(path, callback=None):
    x = cv2.imread(path, -1)
    x = cv2.cvtColor(x, cv2.COLOR_BGRA2RGBA)
    depth = convert_rgba_to_depth(x)
    depth = np.expand_dims(depth, axis=-1)
    if callback is not None:
        depth = callback(depth)
    depth = (depth - depth.min()) / max(1e-8, depth.max() - depth.min())
    depth = scale(depth)
    return convert_image_to_tensor(depth)


class Reconstruction():
    r"""
    Used to reconstruct image.
    Args:
        model: The VQGAN model, which must be one of VQModel or GumbelVQ.
    """

    def __init__(self, model: Union[VQModel, GumbelVQ]):
        model.eval()
        self.model = model
        self.size = model.in_resolution
        self.transform = T.CenterCrop(output_size=2 * (self.size, ), order=['image'])

    @property
    def image_size(self):
        return self.size

    def preprocess(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        s = min(image.shape[:-1])

        if s < self.size:
            raise ValueError(f'min dim for image {s} < {self.size}')

        r = self.size / s
        s = (round(r * image.shape[1]), round(r * image.shape[0]))
        image = cv2.resize(image, dsize=s, interpolation=cv2.INTER_LANCZOS4)
        image = self.transform.apply(image)
        image = image.astype(np.float32) / 255
        image = convert_image_to_tensor(image)
        return scale(image)

    def save_image(self, x, name, func=rescale):
        cv2.imwrite(
            name,
            cv2.cvtColor(
                convert_tensor_to_image(x, func),
                cv2.COLOR_RGB2BGR,
            ),
        )

    def __call__(self, image, file_name=None):
        image = self.preprocess(image)
        # could also use model(x) for reconstruction but use explicit encoding and decoding here
        z = self.model.encode(image)[0]
        print(
            f"VQGAN --- {self.model.__class__.__name__}: latent shape: {z.shape[2:]}")
        xrec = self.model.decode(z)
        if file_name is not None:
            self.save_image(xrec[0], file_name)
        return xrec


class ConditionalSampler():
    r"""
    The class for sliding Window conditional sample.

    Args:
        model: Net2NetTransformer, whose task_type must be one of segmentation or depth.
        temperature: Control the smooth of logits. Default: 1.0 .
        top_k: Get top k of logits when sampling. Default: 100.
        scale_factor: Where to interpolate the input image(s). Default: 1.0 .
        image_size: Only used when task_type is depth, input image will beed to image_size.
            Default: 256.
        animate: Whether to save the sampling process into a mp4 file. Default: True.
        root: Root directory to save files. Default: None.
        seed: Fix random seed. Default: None.
        kernel_size: Size of sliding window when sampling,
            whose square value must be less than half of transformer's block_size. Default: 16.
        fps: FPS of the saved mp4. Default: 15.
        segmentation_save: Only for task_type is segmentation. Whether use a specific method
            to ensure the saved segmentaion images have the same color map. Default: True.
    """

    def __init__(
        self,
        model: Net2NetTransformer,
        temperature: float = 1.0,
        top_k: Optional[int] = 100,
        update_every: int = 50,
        scale_factor: float = 1.0,
        image_size: int = 256,
        animate: bool = False,
        root: Optional[str] = None,
        seed: Optional[int] = None,
        kernel_size: int = 16,
        fps: int = 15,
        segmentation_save: bool = True
    ):
        model.eval()
        self.model = model
        task_type = model.cond_stage_model.task_type
        self.task_type = task_type
        self.temperature = temperature
        self.top_k = top_k
        self.animate = animate
        self.update_every = update_every
        if root is None:
            root = task_type
        self.root = root
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.fps = fps
        self.segmentation_save = segmentation_save
        self.block_size = self.model.transformer.block_size
        if kernel_size ** 2 > self.block_size:
            raise ValueError(
                f"The square value of 'kernel_size' must be less than block_size of transformer {self.block_size}")  # noqa: E501
        if task_type == 'depth':
            self.image_size = image_size
            self.transform = T.CenterCrop((image_size, ) * 2, order=['image'])
        else:
            self.n_label = self.model.cond_stage_model.decoder.out_channel
        os.makedirs(self.root, exist_ok=True)

        if seed is not None:
            np.random.seed(seed)
            mge.random.seed(seed)
            random.seed(seed)

    def interpolate(self, x):
        if self.scale_factor != 1.0:
            return F.nn.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
        return x

    def save_image(self, x, name, func=rescale):
        cv2.imwrite(
            os.path.join(self.root, name),
            cv2.cvtColor(
                convert_tensor_to_image(x, func),
                cv2.COLOR_RGB2BGR,
            ),
        )

    def save_segmentation(self, x, name):
        cv2.imwrite(
            os.path.join(self.root, name),
            cv2.cvtColor(
                convert_segmentation_to_image(x),
                cv2.COLOR_RGB2BGR,
            )
        )

    def save_depth(self, x, name):
        cv2.imwrite(
            os.path.join(self.root, name),
            convert_tensor_to_image(x),
        )

    def preprocess(self, c, x):
        if isinstance(c, str):
            if self.task_type == 'segmentation':
                c = preprocess_segmetation(c, self.n_label)
            else:
                c = preprocess_depth(c, callback=self.transform.apply)

        if isinstance(x, str):
            if self.task_type == 'depth':
                x = preprocess_image(x, callback=self.transform.apply)
            else:
                x = preprocess_image(x)

        return c, x

    def __call__(
        self,
        c: Union[Tensor, str],
        x: Optional[Union[Tensor, str]] = None,
        name: Optional[str] = None,
    ):
        """
            'c' stands for conditional, such as segmentation or depth image.
            'x' means natural image.
        """
        if self.task_type not in ['segmentation', 'depth']:
            raise RuntimeError(
                f"Only supported when task_type is segmentation or depth, but got {self.task_type}.")  # noqa: E501
        if name is None:
            name = str(datetime.datetime.now())
        self.name = name

        c, x = self.preprocess(c, x)

        c = self.interpolate(c)
        quant_c, c_indices = self.model.encode_to_c(c)
        quant_c_shape = quant_c.shape

        if self.task_type == 'segmentation':
            if self.segmentation_save:
                self.save_segmentation(c[0], name + '_segmentation.png')
            else:
                num_classes = c.shape[1]
                c = F.argmax(c, axis=1, keepdims=True)
                c = F.one_hot(c, num_classes)
                c = F.squeeze(c, axis=1).transpose(0, 3, 1, 2).astype('float32')
                c = self.model.cond_stage_model.to_rgb(c)[0]

                self.save_image(c, name + '_segmentation.png')
        else:
            self.save_depth(c[0], name + '_depth.png')

        if x is None:
            codebook_size = self.model.first_stage_model.embed_dim
            z_indices_shape = c_indices.shape
            quant_z_shape = quant_c.shape

            z_indices = Tensor(np.random.randint(
                0, codebook_size, z_indices_shape))
            x_sample = self.model.decode_to_img(z_indices, quant_z_shape)

            start_h = start_w = 0
        else:
            x = self.interpolate(x)
            quant_z, z_indices = self.model.encode_to_z(x)
            quant_z_shape = quant_z.shape

            xrec = self.model.first_stage_model.decode(quant_z)
            self.save_image(xrec[0], name + '_reconstruction.png')

            start = 0
            z_indices[:, start:] = 0

            z_indices = z_indices.reshape(
                quant_z_shape[0], quant_z_shape[2], quant_z_shape[3])
            start_h = start // quant_z_shape[3]
            start_w = start % quant_z_shape[3]

            if quant_z.shape == quant_c.shape:
                z_indices = deepcopy(c_indices).reshape(
                    quant_z_shape[0], quant_z_shape[2], quant_z_shape[3])

            x_sample = self.model.decode_to_img(
                z_indices[:, :quant_z_shape[2], :quant_z_shape[3]], quant_z_shape)

        self.save_image(x_sample[0], name + '_first_sample.png')

        self.sample(
            start_h,
            start_w,
            z_indices.reshape(
                quant_z_shape[0], quant_z_shape[2], quant_z_shape[3]),
            c_indices.reshape(
                quant_c_shape[0], quant_c_shape[2], quant_c_shape[3]),
            quant_z_shape,
        )

    def get_local_index(self, i, length):
        half_size = self.kernel_size // 2
        if i <= half_size:
            center = i
        elif length - i < half_size:
            center = self.kernel_size - length + i
        else:
            center = half_size
        start = i - center
        end = start + self.kernel_size
        return slice(start, end), center

    def sample(self, start_w, start_h, idx, c_idx, z_shape):
        B, _, H, W = z_shape
        total_step = (H - start_h) * (W - start_h) - 1
        last_step = start_h * W + start_w
        if self.animate:
            import imageio  # pylint: disable=import-outside-toplevel
            writer = imageio.get_writer(os.path.join(
                self.root, f"{self.name}_sampling.mp4"), fps=self.fps)

        with tqdm(total=total_step, desc='step') as pbar:
            for h in range(start_h, H):
                range_h, local_h = self.get_local_index(h, H)
                for w in range(start_w, W):
                    range_w, local_w = self.get_local_index(w, W)

                    # get patch
                    patch = F.flatten(idx[:, range_h, range_w], 1)
                    c_patch = F.flatten(c_idx[:, range_h, range_w], 1)
                    patch = F.concat([c_patch, patch], axis=1)

                    logits, _ = self.model.transformer(patch[:, :-1])
                    logits = logits[:, -(self.kernel_size ** 2):, :]

                    logits = logits.reshape(B, self.kernel_size, self.kernel_size, -1)

                    logits = logits[:, local_h, local_w, :]

                    logits /= self.temperature

                    if self.top_k is not None:
                        logits = self.model.top_k_logits(logits, self.top_k)

                    probs = F.softmax(logits, axis=-1)

                    idx[:, h, w] = multinomial(probs, num_samples=1)[0]

                    step = h * W + w
                    pbar.update(step - last_step)
                    last_step = step
                    pbar.set_description(
                        f"Step: ({w},{h}) | Local: ({local_w},{local_h}) | Crop: ({range_h.start}:{range_h.stop},{range_w.start}:{range_w.stop})")  # noqa: E501
                    if step % self.update_every == 0 or step == total_step:
                        x_sample = self.model.decode_to_img(idx, z_shape)
                        self.save_image(
                            x_sample[0], f'{self.name}_sample_{step}.png', rescale)
                        if self.animate:
                            writer.append_data(
                                convert_tensor_to_image(x_sample[0], rescale))
            if self.animate:
                writer.close()


class FastSampler():
    r"""
    The class for unconditional or class conditional sample.

    Args:
        model: Net2NetTransformer, which must be unconditonal or class conditional.
        batch_size: batch_size when sampling.
        temperature: Control the smooth of logits. Default: 1.0 .
        top_k: Get top k of logits when sampling. Default: 100.
        top_p: Default: 1.0 .
        num_samples: How many images to save. Default: 50000,
        class_labels: The lables. for Imagenet dataset, it's [i for i in range(1000)].
        steps: How many steps for sampling, which must be equal to h * w.
        update_every: Save interval. Default: 50.
        root: Root directory to save files. Default: None.
        seed: Fix random seed. Default: None.
    """

    def __init__(
        self,
        model: Net2NetTransformer,
        batch_size: int,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 1.0,
        num_samples: int = 50000,
        class_labels: Optional[Sequence[int]] = None,
        steps: int = 256,
        update_every: int = 50,
        root: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        model.eval()
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_samples = num_samples
        self.labels = class_labels
        self.steps = steps
        self.update_every = update_every

        if root is None:
            root = model.cond_stage_model.task_type
        if class_labels is not None:
            root = os.path.join(root, 'class_label')
        else:
            root = os.path.join(root, 'unconditional')

        os.makedirs(root, exist_ok=True)

        self.root = root

        if seed is not None:
            np.random.seed(seed)
            mge.random.seed(seed)
            random.seed(seed)

    @staticmethod
    def batch_iter(batch_size, total_size):
        steps = total_size // batch_size
        total_step = steps + int((total_size % batch_size) > 0)
        for i in range(total_step):
            if i < steps:
                yield batch_size
            else:
                yield total_size % batch_size

    def save_image(self, x, name, func=rescale):
        cv2.imwrite(
            os.path.join(self.root, name),
            cv2.cvtColor(
                convert_tensor_to_image(x, func),
                cv2.COLOR_RGB2BGR,
            ),
        )

    def sample_class_conditional(self, dim_z=256, h=16, w=16, name=None):
        if not self.model.conditional:
            raise RuntimeError(
                "Expect a class-conditional Net2NetTransformer.")

        if name is None:
            name = str(datetime.datetime.now())

        for class_label in tqdm(self.labels, desc="Classes"):
            batches = iter(FastSampler.batch_iter(self.batch_size, self.num_samples))
            for idx, bs in tqdm(enumerate(batches), desc="Sampling Class"):
                shape = [bs, dim_z, h, w]

                c_indices = F.repeat(mge.tensor(
                    [[class_label]]), repeats=bs, axis=0)
                index_sample = sample_with_past(
                    c_indices,
                    self.model.transformer,
                    self.steps,
                    temperature=self.temperature,
                    sample_logits=True,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )

                x_sample = self.model.decode_to_img(index_sample, shape)
                for i, x in x_sample:
                    count = bs * idx + i
                    self.save_image(x, f"{name}_{count:06}.png")

    def sample_unconditional(self, dim_z=256, h=16, w=16, name=None):
        if self.model.conditional:
            raise RuntimeError(
                "Expect a unconditional Net2NetTransformer.")

        if name is None:
            name = str(datetime.datetime.now())

        sos_token = self.model.cond_stage_model.sos_token

        batches = iter(FastSampler.batch_iter(self.batch_size, self.num_samples))
        for idx, bs in tqdm(enumerate(batches), desc="Sampling Unconditonal"):
            shape = [bs, dim_z, h, w]

            c_indices = F.repeat(mge.tensor(
                [[sos_token]]), repeats=bs, axis=0)
            index_sample = sample_with_past(
                c_indices,
                self.model.transformer,
                self.steps,
                temperature=self.temperature,
                sample_logits=True,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            x_sample = self.model.decode_to_img(index_sample, shape)
            for i, x in enumerate(x_sample):
                count = bs * idx + i
                self.save_image(x, f"{name}_{count:06}.png")

    def __call__(self, dim_z=256, h=16, w=16, name=None):
        if self.steps != h * w:
            raise ValueError('`step` must be equal to h times w.')
        if self.model.conditional:
            self.sample_class_conditional(dim_z, h, w, name)
        else:
            self.sample_unconditional(dim_z, h, w, name)
