import datetime
import math
import os
import random
from typing import Optional, Sequence, Union
from tqdm import tqdm, trange

import cv2
import numpy as np

import megengine as mge
import megengine.data.transform as T
import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.autodiff import GradManager

from ..clip import CLIP, tokenize
from .biggan import BigGAN
from .ema import EMA
from .resample import resample


def convert_tensor_to_image(image):
    # add 0.5 to round to nearest integer
    image = image * 255 + 0.5
    image = F.clip(image, 0, 255).transpose(1, 2, 0).astype('uint8').numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def save_images(images, filename, nrow=8, padding=2):
    if images.ndim == 4 and images.shape[0] != 1:
        total_num, C, h, w = images.shape
    # save images in one picture
        num_x = min(nrow, total_num)
        num_y = int(math.ceil(float(total_num) / num_x))
        H, W = int(h + padding), int(w + padding)
        grid = F.full((C, H * num_y + padding, W * num_x + padding), 0.)
        cur_num = 0
        for y in range(num_y):
            for x in range(num_x):
                if cur_num >= total_num:
                    break
                grid[:, y * H + padding: (y + 1) * H, x * W + padding: (x + 1)
                     * W] = images[cur_num, ...]
                cur_num += 1
    elif images.ndim == 3:
        grid = images
    elif images.shape[0] == 1:
        grid = images[0]
    else:
        raise ValueError('The input must have 3 or 4 dimension.')
    image = convert_tensor_to_image(grid)
    cv2.imwrite(filename, image)


def create_text_path(text, img, save_date_time):
    text_path = ''
    if text is not None:
        text_path += text
    if img is not None:
        if isinstance(img, str):
            # only take name
            img_name = "".join(os.path.splitext(img)[-1].split('.')[:-1])
        else:
            img_name = "INPUT_IMG"
        text_path += '_' + img_name
    text_path = text_path.replace(
        "-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:255]
    if save_date_time:
        text_path = datetime.datetime.now().strftime("%y%m%d-%H%M%S-") + text_path
    return text_path


def cosine_similarity(x, y, axis=1, eps=1e-8):
    if x.ndim != y.ndim:
        raise ValueError('The inputs must have same dimension.')
    if axis < 0:
        axis = x.ndim + axis
    if axis >= x.ndim:
        raise ValueError('Wrong axis was given.')
    t = F.norm(x, ord=2., axis=axis, keepdims=False) * \
        F.norm(y, ord=2., axis=axis, keepdims=False)
    t = F.maximum(t, eps)
    return F.sum(x * y, axis=axis) / t


def differentiable_topk(x, k, temperature=1.0):
    n, dim = x.shape
    topks = []
    for i in range(k):
        prob = F.softmax(x / temperature, axis=-1)
        values, indices = F.topk(prob, 1, descending=True)
        topk = F.scatter(F.zeros_like(x), axis=-1,
                         index=indices, source=values)
        topks.append(topk)
        if not i == k - 1:
            x = F.scatter(x, axis=-1, index=indices,
                          source=F.full(indices.shape, value=float('-inf')))
    topks = F.concat(topks, axis=-1)
    return F.sum(topks.reshape(n, k, dim), axis=1)


def rand_cutout(image, size, center_bias=False, center_focus=2):
    width = image.shape[-1]
    min_offset = 0
    max_offset = width - size
    if center_bias:
        # sample around image center
        center = max_offset / 2
        std = center / center_focus
        offset_x = int(random.gauss(mu=center, sigma=std))
        offset_y = int(random.gauss(mu=center, sigma=std))
        # resample uniformly if over boundaries
        offset_x = random.randint(min_offset, max_offset) if (
            offset_x > max_offset or offset_x < min_offset) else offset_x
        offset_y = random.randint(min_offset, max_offset) if (
            offset_y > max_offset or offset_y < min_offset) else offset_y
    else:
        offset_x = random.randint(min_offset, max_offset)
        offset_y = random.randint(min_offset, max_offset)
    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
    return cutout


class Latents(M.Module):
    def __init__(
        self,
        num_latents: int = 15,
        num_classes: int = 1000,
        z_dim: int = 128,
        max_classes: Optional[int] = None,
        class_temperature: float = 2.
    ):
        super(Latents, self).__init__()
        self.normu = mge.Parameter(F.zeros((num_latents, z_dim)))
        M.init.normal_(self.normu)
        self.cls = mge.Parameter(F.zeros((num_latents, num_classes)))
        M.init.normal_(self.cls, mean=-3.9, std=0.3)
        self.thresh_lat = mge.tensor(1)

        if max_classes is not None:
            if max_classes <= 0 or max_classes > num_classes:
                raise ValueError("`max_classes` must be between 0 and {}, but got {}".format(
                    num_classes, max_classes))

        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if self.max_classes is not None:
            classes = differentiable_topk(
                self.cls, self.max_classes, temperature=self.class_temperature)
        else:
            classes = F.sigmoid(self.cls)

        return self.normu, classes


class BigSleep(M.Module):
    def __init__(
        self,
        num_cutouts: int = 128,
        loss_coef: int = 100,
        image_size: int = 512,
        bilinear: bool = False,
        max_classes: Optional[int] = None,
        class_temperature: float = 2.,
        experimental_resample: bool = False,
        ema_decay: float = 0.99,
        center_bias: bool = False,
        clip_type: str = 'rn50',
    ):
        super(BigSleep, self).__init__()
        if image_size not in [128, 256, 512]:
            raise ValueError("`image size` must be one of 128, 256, or 512")
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        self.experimental_resample = experimental_resample
        self.center_bias = center_bias
        self.biggan = BigGAN.from_pretrained(image_size)
        self.max_classes = max_classes
        self.class_temperature = class_temperature
        self.ema_decay = ema_decay

        self.clip = CLIP.from_pretrained(clip_type, 'float32')

        self.interpolation_settings = {
            'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        self.clip_image_resolution = self.clip.image_resolution
        self.clip_mean = mge.tensor(
            [0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        self.clip_std = mge.tensor(
            [0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)

        self.init_latents()

    def init_latents(self):
        latents = Latents(
            num_latents=len(self.biggan.config.layers) + 1,
            num_classes=self.biggan.config.num_classes,
            z_dim=self.biggan.config.z_dim,
            max_classes=self.max_classes,
            class_temperature=self.class_temperature
        )
        self.latents = EMA(latents, self.ema_decay)

    def normalize_image(self, x):
        return (x - self.clip_mean) / self.clip_std

    def reset(self):
        self.init_latents()

    def get_similarity(self, text, img, text_type='max'):
        sign = -1 if text_type == 'max' else 1
        if img.ndim == 4:
            img = img[0, 0]
        if text.ndim == 4:
            text = text[0, 0]
        return sign * self.loss_coef * F.mean(cosine_similarity(text, img, axis=-1))

    def forward_model(self):
        out = self.biggan(*self.latents(), 1)
        out = (out + 1) / 2
        return out

    def forward(self, texts, text_min=None, return_loss=True):
        out = self.forward_model()

        if not return_loss:
            return out

        pieces = []
        for _ in range(self.num_cutouts):
            # sample cutout size
            size = int(
                self.image_size * np.clip(np.random.normal(loc=0.8, scale=0.3), 0.5, 0.95))
            # get cutout
            apper = rand_cutout(out, size, center_bias=self.center_bias)
            if self.experimental_resample:
                apper = resample(apper, (self.clip_image_resolution,) * 2)
            else:
                apper = F.nn.interpolate(
                    apper, (self.clip_image_resolution,) * 2, **self.interpolation_settings)
            pieces.append(apper)

        image = F.concat(pieces)
        image = self.normalize_image(image)

        image_embed = self.clip.encode_image(image)

        latents, soft_one_hot_classes = self.latents()
        num_latents = latents.shape[0]
        latent_thres = self.latents.model.thresh_lat

        latent_loss = F.abs(1 - F.std(latents, axis=1)).mean() + F.abs(
            F.mean(latents, axis=1)).mean() + 4 * F.maximum(F.square(latents).mean(), latent_thres)

        for latent in latents:
            diffs = latent - latent.mean()
            var = F.mean(diffs ** 2.0)
            std = var ** 0.5
            zscores = diffs / std
            skews = F.mean(zscores ** 3.0)
            kurtoses = F.mean(zscores ** 4.0) - 3.0
            latent_loss = latent_loss + \
                F.abs(kurtoses) / num_latents + F.abs(skews) / num_latents

        cls_loss = ((50 * F.topk(soft_one_hot_classes, k=999)[0] ** 2)).mean()

        result = []
        for text in texts:
            result.append(self.get_similarity(text, image_embed, 'max'))
        if text_min is not None:
            for text in text_min:
                result.append(self.get_similarity(text, image_embed, 'min'))
        similarity_loss = sum(result).mean()

        return out, (latent_loss, cls_loss, similarity_loss)


class Imagine(M.Module):
    r"""
    Imagine class for generating pictures using BigGAN and CLIP.

    Args:
        text: The text used for generating picture(s), which can be str or a sequence of str.
        img: Optional, The reference image will drive process of generating slightly.
        text_min: Optional, Penalizing word(s) in text. Default: None.
        lr: learning rate. Default: 0.07.
        image_size: The image size of generated picture(s), which must be one of (512, 256, 128).
            Default: 512.
        gradient_accumulate_every: How many steps for granient accumulate when training. Default: 1.
        save_every: Save interval. Default: 50.
        epochs: Traning epochs. Default: 20.
        iterations: Iterations in each epoch. Default: 1050.
        save_progress: Whether to save image in each save interval point as a single picture.
            If not specified, the image will cover the last saved picture. Default: False.
        animate: Whether to save all the images in save interval point into a mp4 file.
            Default: True.
        fps: FPS of the saved mp4. Default: 15.
        bilinear: Whether to use bilinear upsample in BigSleep. If False,
            it will use nearest mode for upsample. Default: False.
        seed: Fix random seed. Default: None.
        max_classes: Constraint the maximum number of classification for Latens in BigSleep.
            Default: None.
        class_temperature: Used when max_classes is specified, just for differentiable topk.
            Default: 2.0.
        save_date_time: Whether to add datetime to the name of saved files. Default: False.
        save_best: Whether to save the best score image. Default: True.
        experimental_resample: Whether to use experimental_resample for upsample in BigSleep.
            Default: False.
        ema_decay: Decay in EMA. Default: 0.99.
        num_cutouts: How many times cut out image in BigSleep. Default: 128.
        center_bias: Center bias when cut out image in BigSleep. Default: False.
        clip_type: The architecture of clip, which must be one of ("RN50", "RN101", 'RN50x4',
            'RN50x16', 'RN50x64', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336px').
        root: Root directory to save files.
    """

    def __init__(
        self,
        *,
        text: Union[str, Sequence[str]],
        img: Union[str, Tensor] = None,
        text_min: Optional[Sequence[str]] = None,
        lr: float = .07,
        image_size: int = 512,
        gradient_accumulate_every: int = 1,
        save_every: int = 50,
        epochs: int = 20,
        iterations: int = 1050,
        save_progress: bool = True,
        animate: bool = False,
        fps: int = 15,
        bilinear: bool = False,
        seed: Optional[int] = None,
        max_classes: Optional[int] = None,
        class_temperature: float = 2.,
        save_date_time: bool = False,
        save_best: bool = True,
        experimental_resample: bool = False,
        ema_decay: float = 0.99,
        num_cutouts: int = 128,
        center_bias: bool = False,
        clip_type: str = 'RN50',
        root: str = 'BigSleep',
    ):
        super(Imagine, self).__init__()

        self.seed = seed

        if seed is not None:
            mge.random.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            root = os.path.join(root, str(seed))

        self.epochs = epochs
        self.iterations = iterations
        self.animate = animate
        self.fps = fps

        self.model = BigSleep(
            image_size=image_size,
            bilinear=bilinear,
            max_classes=max_classes,
            class_temperature=class_temperature,
            experimental_resample=experimental_resample,
            ema_decay=ema_decay,
            num_cutouts=num_cutouts,
            center_bias=center_bias,
            clip_type=clip_type,
        )

        self.lr = lr
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.total_image_updates = (
            self.epochs * self.iterations) / self.save_every

        os.makedirs(root, exist_ok=True)
        self.root = root

        self.encoded_texts = {
            "max": [],
            "min": []
        }

        self.clip_transform = T.Compose([
            T.Resize(self.model.clip_image_resolution),
            T.CenterCrop((self.model.clip_image_resolution, ) * 2),
        ])

        self.get_clip_encoding(text=text, img=img, text_min=text_min)
        self.reset()

    def reset(self):
        self.model.reset()
        self.optimizer = mge.optimizer.Adam(self.model.latents.model.parameters(), self.lr)
        self.gm = GradManager().attach(self.model.latents.model.parameters())

    def apply_clip_transform(self, x):
        x = self.clip_transform.apply(x)
        x = mge.tensor(x / 255, dtype='float32')
        x = F.expand_dims(x.transpose(2, 0, 1), axis=0)
        x = self.model.normalize_image(x)
        return x

    def encode_text(self, text):
        if text is None:
            return 0, 0
        tokenized_text = tokenize(text)
        text_encoding = self.model.clip.encode_text(tokenized_text).detach()
        return text_encoding, 1

    def encode_image(self, img):
        if img is None:
            return 0, 0
        if isinstance(img, str):
            img = cv2.imread(img)[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        normed_img = self.apply_clip_transform(img)
        img_encoding = self.model.clip.encode_image(normed_img).detach()
        return img_encoding, 1

    def clip_encode(self, *, text=None, img=None):
        if text is None and img is None:
            raise ValueError(
                "Must give one of text and img as least, but got all None")
        text_encoding, text_flag = self.encode_text(text)
        img_encoding, img_flag = self.encode_image(img)
        return (text_encoding + img_encoding) / (text_flag + img_flag)

    def encode(self, *, text, img=None, text_min=None):
        if '|' in text:
            text = text.split('|')
        else:
            text = [text]
        self.encoded_texts['max'] = [self.clip_encode(text=t, img=img) for t in text]

        if text_min is not None and text_min != "":
            if '|' in text_min:
                text_min = text_min.split('|')
            else:
                text_min = [text_min]
            self.encoded_texts['min'] = [
                self.clip_encode(text=t, img=img) for t in text_min]

    def get_clip_encoding(self, *, text=None, img=None, text_min=None):
        self.best_score = 0
        self.text = text
        self.text_min = text_min
        if isinstance(text, list):
            text = '|'.join(text)
        if text_min is not None and len(text_min) > 0:
            if isinstance(text_min, list):
                text_min = '|'.join(text_min)
            text = (text + '_wout_' + text_min[:255]
                    ) if text is not None else ('wout_' + text_min[:255])

        self.text_path = create_text_path(text, img, self.save_date_time)

        self.filename = os.path.join(self.root, f'./{self.text_path}.png')

        self.encode(text=text, img=img, text_min=text_min)

    def train_step(self, epoch, i, pbar=None, writer=None):
        total_loss = 0
        with self.gm:
            for _ in range(self.gradient_accumulate_every):
                out, losses = self.model(
                    self.encoded_texts["max"], self.encoded_texts["min"])
                loss = sum(losses) / self.gradient_accumulate_every
                total_loss += loss
                self.gm.backward(loss)

            self.optimizer.step().clear_grad()
            self.model.latents.update()

        if (i + 1) % self.save_every == 0:
            self.model.latents.model.eval()
            out, losses = self.model(
                self.encoded_texts["max"], self.encoded_texts["min"])

            top_score, best = F.topk(F.expand_dims(losses[2], 0), k=1)
            image = self.model.forward_model()[best[0]]
            self.model.latents.model.train()
            save_images(image, str(self.filename))
            if pbar is not None:
                pbar.update(1)
            else:
                print(f'image updated at "./{str(self.filename)}"')

            if self.save_progress is not None:
                total_iterations = epoch * self.iterations + i
                num = total_iterations // self.save_every
                save_images(image, os.path.join(self.root, f'{self.text_path}_{num}.png'))

            if writer is not None:
                writer.append_data(convert_tensor_to_image(image))

            if self.save_best and top_score.item() < self.best_score:
                self.best_score = top_score.item()
                save_images(
                    image, os.path.join(self.root, f'{self.text_path}_best.png'))

        return out, total_loss

    def forward(self):
        penalizing = ""
        if self.text_min is not None and len(self.text_min) > 0:
            penalizing = f'penalizing "{self.text_min}"'
        print(f'Imagining "{self.text_path}" {penalizing}...')

        writer = None
        if self.animate:
            import imageio  # pylint: disable=import-outside-toplevel
            writer = imageio.get_writer(
                os.path.join(self.root, f"{self.text_path}_{penalizing}.mp4"),
                fps=self.fps,
            )

        image_pbar = tqdm(total=self.total_image_updates,
                          desc='image update', position=2, leave=True)
        epoch_pbar = trange(self.epochs, desc='      epochs',
                            position=0, leave=True)
        for epoch in (ep for ep in epoch_pbar):
            pbar = trange(self.iterations, desc='   iteration',
                          position=1, leave=True)
            image_pbar.update(0)
            for i in (it for it in pbar):
                _, loss = self.train_step(epoch, i, image_pbar, writer)
                pbar.set_description(f'loss: {loss.item():04.2f}')
        if writer is not None:
            writer.close()
