# -*- coding: utf-8 -*-
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# ---------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
# ----------------------------------------------------------------------
"""Megengine BERT model."""

import copy
import json
import math
import os
import urllib
import urllib.request
from io import open

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
from megengine import Parameter
from megengine.functional.loss import cross_entropy
from megengine.module import Dropout, Embedding, Linear, Module, Sequential
from megengine.module.activation import Softmax


def transpose(inp, a, b):
    cur_shape = list(range(0, inp.ndim))
    cur_shape[a], cur_shape[b] = cur_shape[b], cur_shape[a]
    return inp.transpose(cur_shape)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        x * 0.5 * (1.0 + F.tanh((F.sqrt(2 / math.pi) * (x + 0.044715 * (x **  3)))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + F.tanh(F.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))))


ACT2FN = {"gelu": gelu, "relu": F.relu}


class BertConfig:
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class BertLayerNorm(Module):
    """Construct a layernorm module in the TF style (epsilon inside the square root).
    """

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = Parameter(np.ones(hidden_size).astype(np.float32))
        self.bias = Parameter(np.zeros(hidden_size).astype(np.float32))
        self.variance_epsilon = eps

    def forward(self, x):
        u = F.mean(x, len(x.shape) - 1, True)
        s = F.mean((x - u) ** 2, len(x.shape) - 1, True)
        x = (x - u) / ((s + self.variance_epsilon) ** 0.5)
        return self.weight * x + self.bias


class BertEmbeddings(Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name
        # and be able to load any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.shape[1]

        if token_type_ids is None:
            token_type_ids = F.zeros_like(input_ids)

        position_ids = F.linspace(0, seq_length - 1, seq_length).astype(np.int32)
        position_ids = F.broadcast_to(F.expand_dims(position_ids, 0), input_ids.shape)
        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # using symbolic shapes to make trace happy
        x_shape = mge.tensor(x.shape)
        new_x_shape = F.concat(
            [x_shape[:-1], (self.num_attention_heads, self.attention_head_size)]
        )
        x = x.reshape(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = F.matmul(query_layer, transpose(key_layer, -1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = Softmax(len(attention_scores.shape) - 1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = F.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        # using symbolic shapes to make trace happy
        context_shape = mge.tensor(context_layer.shape)
        new_context_layer_shape = F.concat([context_shape[:-2], self.all_head_size])
        context_layer = context_layer.reshape(new_context_layer_shape)
        return context_layer


class BertSelfOutput(Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(Module):
    def __init__(self, config):
        super().__init__()
        self.layer = Sequential(
            *[BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # self.layer = ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.activation = F.tanh

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary
            (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape
            [batch_size, sequence_length] with the token types indices selected in [0, 1].
            Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, 1]. It's a mask to be used if the input sequence length
            is smaller than the max input sequence length in the current batch.
            It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers`
            output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of
                encoded-hidden-states at the end of each attention block
                (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size
                [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of
                hidden-states corresponding to the last attention block of shape
                [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size]
            which is the output of classifier pretrained on top of the hidden state
            associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
    ):
        if attention_mask is None:
            attention_mask = F.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = F.zeros_like(input_ids)
        # print('input_ids', input_ids.sum())
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        # print('attention_mask', attention_mask.sum())
        extended_attention_mask = F.expand_dims(attention_mask, (1, 2))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.astype(
            next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForSequenceClassification(Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary.
            Items in the batch should begin with the special "CLS" token.
            (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, 1]. It's a mask to be used if the input sequence length
            is smaller than the max input sequence length in the current batch. It's the mask
            that we typically use for attention when a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels, bert=None):
        super().__init__()
        if bert is None:
            self.bert = BertModel(config)
        else:
            self.bert = bert
        self.num_labels = num_labels
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = cross_entropy(
                logits.reshape(-1, self.num_labels), labels.reshape(-1)
            )
            return logits, loss
        else:
            return logits, None


DATA_URL = "https://data.megengine.org.cn/models/weights/bert"
CONFIG_NAME = "bert_config.json"
VOCAB_NAME = "vocab.txt"
MODEL_NAME = {
    "wwm_cased_L-24_H-1024_A-16": "wwm_cased_L_24_H_1024_A_16",
    "wwm_uncased_L-24_H-1024_A-16": "wwm_uncased_L_24_H_1024_A_16",
    "cased_L-12_H-768_A-12": "cased_L_12_H_768_A_12",
    "cased_L-24_H-1024_A-16": "cased_L_24_H_1024_A_16",
    "uncased_L-12_H-768_A-12": "uncased_L_12_H_768_A_12",
    "uncased_L-24_H-1024_A-16": "uncased_L_24_H_1024_A_16",
    "chinese_L-12_H-768_A-12": "chinese_L_12_H_768_A_12",
    "multi_cased_L-12_H-768_A-12": "multi_cased_L_12_H_768_A_12",
}


def download_file(url, filename):
    # urllib.URLopener().retrieve(url, filename)
    urllib.request.urlretrieve(url, filename)


def create_hub_bert(model_name, pretrained):
    assert model_name in MODEL_NAME, "{} not in the valid models {}".format(
        model_name, MODEL_NAME
    )
    data_dir = "./{}".format(model_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    vocab_url = "{}/{}/{}".format(DATA_URL, model_name, VOCAB_NAME)
    config_url = "{}/{}/{}".format(DATA_URL, model_name, CONFIG_NAME)

    vocab_file = "./{}/{}".format(model_name, VOCAB_NAME)
    config_file = "./{}/{}".format(model_name, CONFIG_NAME)

    download_file(vocab_url, vocab_file)
    download_file(config_url, config_file)

    config = BertConfig(config_file)

    model = hub.load("megengine/models", MODEL_NAME[model_name], pretrained=pretrained)

    return model, config, vocab_file


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "uncased_L-12_H-768_A-12/bert_4f2157f7_uncased_L-12_H-768_A-12.pkl"
)
def uncased_L_12_H_768_A_12():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    }
    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "cased_L-12_H-768_A-12/bert_b9727c2f_cased_L-12_H-768_A-12.pkl"
)
def cased_L_12_H_768_A_12():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 28996,
    }
    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "uncased_L-24_H-1024_A-16/bert_222f5012_uncased_L-24_H-1024_A-16.pkl"
)
def uncased_L_24_H_1024_A_16():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    }

    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "cased_L-24_H-1024_A-16/bert_01f2a65f_cased_L-24_H-1024_A-16.pkl"
)
def cased_L_24_H_1024_A_16():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 28996,
    }

    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "chinese_L-12_H-768_A-12/bert_ee91be1a_chinese_L-12_H-768_A-12.pkl"
)
def chinese_L_12_H_768_A_12():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 21128,
    }
    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "multi_cased_L-12_H-768_A-12/bert_283ceec5_multi_cased_L-12_H-768_A-12.pkl"
)
def multi_cased_L_12_H_768_A_12():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 119547,
    }

    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "wwm_uncased_L-24_H-1024_A-16/bert_e2780a6a_wwm_uncased_L-24_H-1024_A-16.pkl"
)
def wwm_uncased_L_24_H_1024_A_16():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    }
    config = BertConfig.from_dict(config_dict)
    return BertModel(config)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/bert/"
    "wwm_cased_L-24_H-1024_A-16/bert_0a8f1389_wwm_cased_L-24_H-1024_A-16.pkl"
)
def wwm_cased_L_24_H_1024_A_16():
    config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 28996,
    }
    config = BertConfig.from_dict(config_dict)
    return BertModel(config)
