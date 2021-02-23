# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files)"
        " for the task.",
    )

    parser.add_argument(
        "--pretrained_bert", required=True, type=str, help="pretrained bert name"
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--load_model_path",
        default="./check_point_last.pkl",
        type=str,
        help="the initial model",
    )

    parser.add_argument(
        "--save_model_path",
        default="./check_point_last.pkl",
        type=str,
        help="the path to save model",
    )

    return parser.parse_args()
