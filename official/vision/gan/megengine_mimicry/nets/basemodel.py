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
import os
from abc import abstractmethod

import megengine
import megengine.jit as jit
import megengine.module as M
import numpy as np


class BaseModel(M.Module):
    def __init__(self):
        super().__init__()
        self.train_step = self._reset_jit_graph(self._train_step_implementation)
        self.infer_step = self._reset_jit_graph(self._infer_step_implementation)

    def _reset_jit_graph(self, impl: callable):
        """create a `jit.trace` object based on abstract graph implementation"""
        return jit.trace(impl)

    @abstractmethod
    def _train_step_implementation(self, *args, **kwargs):
        """Abstract train step function, traced at the beginning of training.

        A typical implementation for a classifier could be
        ```
        class Classifier(BaseModel):

            def _train_step_implementation(
                self,
                image: Tensor,
                label: Tensor,
                opt: Optimizer = None
            ):
                logits = self.forward(image)
                loss = F.cross_entropy_with_softmax(logits, label)
                if opt is not None:
                    opt.zero_grad()
                    opt.backward(loss)
                    opt.step()
        ```

        This implementation is wrapped in a `megengine.jit.trace` object, which equals to
        something like
        ```
        @jit.trace
        def train_step(image, label, opt=None):
            return _train_step_implemenation(image, label, opt=opt)
        ```

        And we call `model.train_step(np_image, np_label, opt=sgd_optimizer)` to
        perform the wrapped training step.
        """
        raise NotImplementedError

    @abstractmethod
    def _infer_step_implementation(self, *args, **kwargs):
        """Abstract infer step function, traced at the beginning of inference.

        See document of `_train_step_implementation`.
        """
        raise NotImplementedError

    def train(self, mode: bool = True):
        # when switching mode, graph should be reset
        self.train_step = self._reset_jit_graph(self._train_step_implementation)
        self.infer_step = self._reset_jit_graph(self._infer_step_implementation)
        super().train(mode=mode)

    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(np.prod(p.shape) for p in self.parameters())
        num_trainable_params = sum(np.prod(p.shape) for p in self.parameters(requires_grad=True))

        return num_total_params, num_trainable_params

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        r"""
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file to be restored.")

        ckpt_dict = megengine.load(ckpt_file)

        # Restore model weights
        self.load_state_dict(ckpt_dict['model_state_dict'])

        # Restore optimizer status if existing. Evaluation doesn't need this
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # Return global step
        return ckpt_dict['global_step']

    def save_checkpoint(self,
                        directory,
                        global_step,
                        optimizer=None,
                        name=None):
        r"""
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            global_step (int): The global step variable during training.
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':
            self.state_dict(),
            'optimizer_state_dict':
            optimizer.state_dict() if optimizer is not None else None,
            'global_step':
            global_step
        }

        # Save the file with specific name
        if name is None:
            name = "{}_{}_steps.pth".format(
                os.path.basename(directory),  # netD or netG
                global_step)

        megengine.save(ckpt_dict, os.path.join(directory, name))
