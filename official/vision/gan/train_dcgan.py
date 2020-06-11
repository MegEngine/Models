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
import megengine.data as data
import megengine.data.transform as T
import megengine.optimizer as optim

import megengine_mimicry as mmc
import megengine_mimicry.nets.dcgan.dcgan_cifar as dcgan

dataset = mmc.datasets.load_dataset(root=None, name='cifar10')
dataloader = data.DataLoader(
    dataset,
    sampler=data.Infinite(data.RandomSampler(dataset, batch_size=64, drop_last=True)),
    transform=T.Compose([T.Normalize(std=255), T.ToMode("CHW")]),
    num_workers=4
)

netG = dcgan.DCGANGeneratorCIFAR()
netD = dcgan.DCGANDiscriminatorCIFAR()
optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

LOG_DIR = "./log/dcgan_example"

trainer = mmc.training.Trainer(
    netD=netD,
    netG=netG,
    optD=optD,
    optG=optG,
    n_dis=5,
    num_steps=100000,
    lr_decay="linear",
    dataloader=dataloader,
    log_dir=LOG_DIR,
    device=0)

trainer.train()

mmc.metrics.compute_metrics.evaluate(
    metric="fid",
    netG=netG,
    log_dir=LOG_DIR,
    evaluate_step=100000,
    num_runs=1,
    device=0,
    num_real_samples=50000,
    num_fake_samples=50000,
    dataset_name="cifar10",
)

mmc.metrics.compute_metrics.evaluate(
    metric="inception_score",
    netG=netG,
    log_dir=LOG_DIR,
    evaluate_step=100000,
    num_runs=1,
    device=0,
    num_samples=50000,
)

mmc.metrics.compute_metrics.evaluate(
    metric="kid",
    netG=netG,
    log_dir=LOG_DIR,
    evaluate_step=100000,
    num_runs=1,
    device=0,
    num_subsets=50,
    subset_size=1000,
    dataset_name="cifar10",
)

