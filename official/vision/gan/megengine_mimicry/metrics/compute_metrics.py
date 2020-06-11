# Copyright (c) 2020 Kwot Sin Lee
# This code is licensed under MIT license
# (https://github.com/kwotsin/mimicry/blob/master/LICENSE)
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
"""
Computes different GAN metrics for a generator.
"""
import os

import numpy as np

from . import compute_fid, compute_is, compute_kid
from ..utils import common


def evaluate(metric,
             netG,
             log_dir,
             evaluate_range=None,
             evaluate_step=None,
             num_runs=3,
             start_seed=0,
             overwrite=False,
             write_to_json=True,
             device=None,
             **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    if metric == 'kid':
        if 'num_subsets' not in kwargs or 'subset_size' not in kwargs:
            raise ValueError(
                "num_subsets and subset_size must be provided for KID computation.")

    elif metric == 'fid':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for FID computation.")

    elif metric == 'inception_score':
        if 'num_samples' not in kwargs:
            raise ValueError("num_samples must be provided for IS computation.")

    else:
        choices = ['fid', 'kid', 'inception_score']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(metric, choices))

    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')

    if not os.path.exists(ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                ckpt_dir))

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
        'inception_score': 'Inception Score',
        'kid': 'KID',
    }

    # Set output file and restore if available.
    if metric == 'fid':
        output_file = os.path.join(
            log_dir,
            'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                      kwargs['num_fake_samples'] // 1000))

    elif metric == 'inception_score':
        output_file = os.path.join(
            log_dir,
            'inception_score_{}k.json'.format(kwargs['num_samples'] // 1000))

    elif metric == 'kid':
        output_file = os.path.join(
            log_dir, 'kid_{}k_{}_subsets.json'.format(
                kwargs['num_subsets'] * kwargs['subset_size'] // 1000,
                kwargs['num_subsets']))

    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        if step in scores_dict and write_to_json and not overwrite:
            print("INFO: {} at step {} has been computed. Skipping...".format(
                names_dict[metric], step))
            continue

        # Load and restore the model checkpoint
        ckpt_file = os.path.join(ckpt_dir, 'netG_{}_steps.pth'.format(step))
        if not os.path.exists(ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=ckpt_file, optimizer=None)

        # Compute score for each seed
        scores = []
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "fid":
                score = compute_fid.fid_score(netG=netG,
                                              seed=seed,
                                              device=device,
                                              log_dir=log_dir,
                                              **kwargs)

            elif metric == "inception_score":
                score, _ = compute_is.inception_score(netG=netG,
                                                      seed=seed,
                                                      device=device,
                                                      log_dir=log_dir,
                                                      **kwargs)

            elif metric == "kid":
                score, _ = compute_kid.kid_score(netG=netG,
                                                 device=device,
                                                 seed=seed,
                                                 log_dir=log_dir,
                                                 **kwargs)

            scores.append(score)
            print("INFO: {} (step {}) [seed {}]: {}".format(
                names_dict[metric], step, seed, score))

        scores_dict[step] = scores

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            scores = scores_dict[step]
            mean = np.mean(scores)
            std = np.std(scores)

            print("INFO: {} (step {}): {} (± {}) ".format(
                names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict
