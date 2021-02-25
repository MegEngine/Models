# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from tqdm import tqdm

import megengine as mge
import megengine.functional as F

from official.nlp.bert.config_args import get_args
from official.nlp.bert.model import BertForSequenceClassification, create_hub_bert
from official.nlp.bert.mrpc_dataset import MRPCDataset

args = get_args()
logger = mge.get_logger(__name__)


def net_eval(input_ids, segment_ids, input_mask, label_ids, net=None):
    net.eval()
    results = net(input_ids, segment_ids, input_mask, label_ids)
    logits, loss = results
    return loss, logits, label_ids


def eval(dataloader, net):
    logger.info("***** Running evaluation *****")
    logger.info("batch size = %d", args.eval_batch_size)

    sum_loss, sum_accuracy, total_steps, total_examples = 0, 0, 0, 0

    for _, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        input_ids, input_mask, segment_ids, label_ids = tuple(
            mge.tensor(t) for t in batch
        )
        batch_size = input_ids.shape[0]
        loss, logits, label_ids = net_eval(
            input_ids, segment_ids, input_mask, label_ids, net=net
        )
        sum_loss += loss.mean().item()
        sum_accuracy += F.topk_accuracy(logits, label_ids) * batch_size
        total_examples += batch_size
        total_steps += 1

    result = {
        "eval_loss": sum_loss / total_steps,
        "eval_accuracy": sum_accuracy / total_examples,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("%s = %s", key, str(result[key]))


if __name__ == "__main__":
    bert, config, vocab_file = create_hub_bert(args.pretrained_bert, pretrained=False)
    args.vocab_file = vocab_file
    model = BertForSequenceClassification(config, num_labels=2, bert=bert)
    mrpc_dataset = MRPCDataset(args)
    model.load_state_dict(mge.load(args.load_model_path))
    mrpc_dataset = MRPCDataset(args)
    eval_dataloader, eval_size = mrpc_dataset.get_eval_dataloader()
    eval(eval_dataloader, model)
