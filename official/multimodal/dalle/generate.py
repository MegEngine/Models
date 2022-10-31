import math
import os
from typing import Optional, Sequence, Union
from tqdm import tqdm

import megengine.functional as F

from ..big_sleep.big_sleep import save_images
from .dalle import DALLE
from .tokenizer import SimpleTokenizer, YttmTokenizer


def normlize_image(image, low, high):
    image = F.clip(image, low, high)
    image = (image - low) / (F.maximum(high - low, 1e-5))
    return image


def get_split_indices(length, splits_size):
    nums = length // splits_size
    left_size = length % splits_size
    out = [splits_size * i for i in range(1, nums + 1)]
    if left_size == 0:
        return out[:-1]
    return out


class Generator():
    r"""
    Generate images from texts.

    Args:
        dalle: The DALLE model.
        texts: The input texts, which can be a single string or a squance of string.
        top_k: Sampling parameter.
        num_images: How many images to generate for each text.
        temperature: Sampling parameter.
        root: Root directory to save files. Default: None.
    """

    def __init__(
        self,
        dalle: DALLE,
        texts: Union[str, Sequence[str]],
        top_k: float = 0.9,
        num_images: int = 64,
        batch_size: int = 4,
        generate_texts: bool = False,
        bpe_path: Optional[str] = None,
        temperature: float = 1.0,
        root: Optional[str] = None
    ):
        dalle.eval()
        self.dalle = dalle
        self.set_text(texts)
        if bpe_path is None:
            self.tokenizer = SimpleTokenizer()
        else:
            self.tokenizer = YttmTokenizer(bpe_path)
        if top_k <= 0 or top_k > 1:
            raise ValueError("`top_k` must be between 0 and 1.")
        self.generate_texts = generate_texts
        self.topk = top_k
        self.text_seq_len = dalle.text_seq_len
        self.num_images = num_images
        self.batch_size = batch_size
        self.temperature = temperature

        if root is None or root == "":
            root = './dalle'
        os.makedirs(root, exist_ok=True)
        self.root = root

    def set_text(self, texts: Union[str, Sequence[str]]):
        if not isinstance(texts, (list, tuple)):
            raise ValueError()
        elif isinstance(texts, str):
            texts = [texts]

        self.texts = texts

    def __call__(self):
        global_pbar = tqdm(self.texts, desc='Global process', total=(
            math.ceil(self.num_images / self.batch_size)) * len(self.texts))
        for text in global_pbar:
            if self.generate_texts:
                text_tokens, gen_texts = self.dalle.generate_texts(
                    self.tokenizer,
                    text=text,
                    filter_thread=self.topk,
                    temperature=self.temperature,
                )
                text = gen_texts[0]
            else:
                text_tokens = self.tokenizer.tokenize([text], self.text_seq_len)

            texts = F.repeat(text_tokens, repeats=self.num_images, axis=0)

            outputs_dir = os.path.join(self.root, text.replace(' ', '_')[:100])
            os.makedirs(outputs_dir, exist_ok=True)
            pbar = tqdm(
                F.split(texts, get_split_indices(self.num_images, self.batch_size)),
                desc=f'Generating images for - {text}'
            )

            for idx, text_chunk in enumerate(pbar):
                image = self.dalle.generate_images(text_chunk, filter_thread=self.topk)
                max_value = F.max(image)
                min_value = F.min(image)
                image = normlize_image(image, min_value, max_value)
                save_images(image, os.path.join(outputs_dir, f'{idx}.png'))
                global_pbar.update(1)
