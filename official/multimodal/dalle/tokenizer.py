import os
import youtokentome as yttm

import megengine.functional as F
from megengine import Tensor

from ..clip.simple_tokenizer import SimpleTokenizer  # pylint: disable=unused-import  # noqa: F401


class YttmTokenizer:
    def __init__(self, bpe_path: str):
        if not os.path.exists(bpe_path):
            raise ValueError(f'BPE json path {bpe_path} does not exist')

        tokenizer = yttm.BPE(model=bpe_path)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()

    def decode(self, tokens, pad_tokens=(0, )):
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()

        return self.tokenizer.decode(tokens, ignore_ids=pad_tokens)

    def encode(self, texts):
        encoded = self.tokenizer.encode(texts, output_type=yttm.OutputType.ID)
        return list(map(Tensor, encoded))

    def tokenize(self, texts, context_length=256, truncate_text=False):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = self.encode(texts)

        result = F.zeros((len(all_tokens), context_length), dtype='int32')
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = Tensor(tokens)

        return result
