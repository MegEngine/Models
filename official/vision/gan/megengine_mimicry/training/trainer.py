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
Implementation of Trainer object for training GANs.
"""
import os
import re
import time

import megengine

from ..utils import common
from . import logger, metric_log, scheduler


class Trainer:
    """
    Trainer object for constructing the GAN training pipeline.

    Attributes:
        netD (Module): Torch discriminator model.
        netG (Module): Torch generator model.
        optD (Optimizer): Torch optimizer object for discriminator.
        optG (Optimizer): Torch optimizer object for generator.
        dataloader (DataLoader): Torch object for loading data from a dataset object.
        num_steps (int): The number of training iterations.
        n_dis (int): Number of discriminator update steps per generator training step.
        lr_decay (str): The learning rate decay policy to use.
        log_dir (str): The path to storing logging information and checkpoints.
        logger (Logger): Logger object for visualising training information.
        scheduler (LRScheduler): GAN training specific learning rate scheduler object.
        params (dict): Dictionary of training hyperparameters.
        netD_ckpt_file (str): Custom checkpoint file to restore discriminator from.
        netG_ckpt_file (str): Custom checkpoint file to restore generator from.
        print_steps (int): Number of training steps before printing training info to stdout.
        vis_steps (int): Number of training steps before visualising images with TensorBoard.
        flush_secs (int): Number of seconds before flushing summaries to disk.
        log_steps (int): Number of training steps before writing summaries to TensorBoard.
        save_steps (int): Number of training steps bfeore checkpointing.
        save_when_end (bool): If True, saves final checkpoint when training concludes.
    """
    def __init__(self,
                 netD,
                 netG,
                 optD,
                 optG,
                 dataloader,
                 num_steps,
                 log_dir='./log',
                 n_dis=1,
                 netG_ckpt_file=None,
                 netD_ckpt_file=None,
                 lr_decay=None,
                 **kwargs):
        self.netD = netD
        self.netG = netG
        self.optD = optD
        self.optG = optG
        self.n_dis = n_dis
        self.lr_decay = lr_decay
        self.dataloader = dataloader
        self.num_steps = num_steps

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Obtain custom or latest checkpoint files
        if netG_ckpt_file:
            self.netG_ckpt_dir = os.path.dirname(netG_ckpt_file)
            self.netG_ckpt_file = netG_ckpt_file
        else:
            self.netG_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netG')
            self.netG_ckpt_file = self._get_latest_checkpoint(
                self.netG_ckpt_dir)  # can be None

        if netD_ckpt_file:
            self.netD_ckpt_dir = os.path.dirname(netD_ckpt_file)
            self.netD_ckpt_file = netD_ckpt_file
        else:
            self.netD_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netD')
            self.netD_ckpt_file = self._get_latest_checkpoint(
                self.netD_ckpt_dir)  # can be None

        # Default parameters, unless provided by kwargs
        default_params = {
            'print_steps': kwargs.get('print_steps', 1),
            'vis_steps': kwargs.get('vis_steps', 500),
            'flush_secs': kwargs.get('flush_secs', 30),
            'log_steps': kwargs.get('log_steps', 50),
            'save_steps': kwargs.get('save_steps', 5000),
            'save_when_end': kwargs.get('save_when_end', True),
        }
        for param in default_params:
            self.__dict__[param] = default_params[param]

        # Hyperparameters for logging experiments
        self.params = {
            'log_dir': self.log_dir,
            'num_steps': self.num_steps,
            # 'batch_size': self.dataloader.sampler.batch_size,
            'n_dis': self.n_dis,
            'lr_decay': self.lr_decay,
            # 'optD': optD.__repr__(),
            # 'optG': optG.__repr__(),
        }
        self.params.update(default_params)

        # Log training hyperparmaeters
        self._log_params(self.params)

        # Training helper objects
        self.logger = logger.Logger(log_dir=self.log_dir,
                                    num_steps=self.num_steps,
                                    dataset_size=len(self.dataloader.dataset),
                                    flush_secs=self.flush_secs)

        self.scheduler = scheduler.LRScheduler(lr_decay=self.lr_decay,
                                               optD=self.optD,
                                               optG=self.optG,
                                               num_steps=self.num_steps)

    def _log_params(self, params):
        """
        Takes the argument options to save into a json file.
        """
        params_file = os.path.join(self.log_dir, 'params.json')

        # Check for discrepancy with previous training config.
        if os.path.exists(params_file):
            check = common.load_from_json(params_file)

            if params != check:
                diffs = []
                for k in params:
                    if k in check and params[k] != check[k]:
                        diffs.append('{}: Expected {} but got {}.'.format(
                            k, check[k], params[k]))

                diff_string = '\n'.join(diffs)
                raise ValueError(
                    "Current hyperparameter configuration is different from previously:\n{}"
                    .format(diff_string))

        common.write_to_json(params, params_file)

    def _get_latest_checkpoint(self, ckpt_dir):
        """
        Given a checkpoint dir, finds the checkpoint with the latest training step.
        """
        def _get_step_number(k):
            """
            Helper function to get step number from checkpoint files.
            """
            search = re.search(r'(\d+)_steps', k)

            if search:
                return int(search.groups()[0])
            else:
                return -float('inf')

        if not os.path.exists(ckpt_dir):
            return None

        files = os.listdir(ckpt_dir)
        if len(files) == 0:
            return None

        ckpt_file = max(files, key=lambda x: _get_step_number(x))

        return os.path.join(ckpt_dir, ckpt_file)

    def _fetch_data(self, iter_dataloader):
        """
        Fetches the next set of data and refresh the iterator when it is exhausted.
        Follows python EAFP, so no iterator.hasNext() is used.
        """
        real_batch = next(iter_dataloader)

        if isinstance(real_batch, (tuple, list)):  # (image, label)
            real_batch = real_batch[0]

        return iter_dataloader, real_batch

    def _restore_models_and_step(self):
        """
        Restores model and optimizer checkpoints and ensures global step is in sync.
        """
        global_step_D = global_step_G = 0

        if self.netD_ckpt_file and os.path.exists(self.netD_ckpt_file):
            print("INFO: Restoring checkpoint for D...")
            global_step_D = self.netD.restore_checkpoint(
                ckpt_file=self.netD_ckpt_file, optimizer=self.optD)

        if self.netG_ckpt_file and os.path.exists(self.netG_ckpt_file):
            print("INFO: Restoring checkpoint for G...")
            global_step_G = self.netG.restore_checkpoint(
                ckpt_file=self.netG_ckpt_file, optimizer=self.optG)

        if global_step_G != global_step_D:
            raise ValueError('G and D Networks are out of sync.')
        else:
            global_step = global_step_G  # Restores global step

        return global_step

    def train(self):
        """
        Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        global_step = self._restore_models_and_step()
        print("INFO: Starting training from global step {}...".format(
            global_step))

        try:
            start_time = time.time()

            # Iterate through data
            iter_dataloader = iter(self.dataloader)
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                # -------------------------
                #   One Training Step
                # -------------------------
                # Update n_dis times for D
                for i in range(self.n_dis):
                    iter_dataloader, real_batch = self._fetch_data(
                        iter_dataloader=iter_dataloader)

                    # -----------------------
                    #   Update G Network
                    # -----------------------
                    # Update G, but only once.
                    if i == 0:
                        errG = self.netG.train_step(
                            real_batch,
                            netD=self.netD,
                            optG=self.optG)
                        log_data.add_metric("errG", errG.item(), group="loss")

                    # ------------------------
                    #   Update D Network
                    # -----------------------
                    errD, D_x, D_Gz = self.netD.train_step(real_batch,
                                                           netG=self.netG,
                                                           optD=self.optD)
                    log_data.add_metric("errD", errD.item(), group="loss")
                    log_data.add_metric("D_x", D_x.item(), group="prob")
                    log_data.add_metric("D_Gz", D_Gz.item(), group="prob")

                # --------------------------------
                #   Update Training Variables
                # -------------------------------
                global_step += 1

                log_data = self.scheduler.step(log_data=log_data,
                                               global_step=global_step)

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                          self.print_steps)
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    self.logger.vis_images(netG=self.netG,
                                           global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                              global_step=global_step,
                                              optimizer=self.optG)

                    self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                              global_step=global_step,
                                              optimizer=self.optD)

            # Save models at the very end of training
            if self.save_when_end:
                print("INFO: Saving final checkpoints...")
                self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                          global_step=global_step,
                                          optimizer=self.optG)

                self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                          global_step=global_step,
                                          optimizer=self.optD)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                      global_step=global_step,
                                      optimizer=self.optG)

            self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                      global_step=global_step,
                                      optimizer=self.optD)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")
