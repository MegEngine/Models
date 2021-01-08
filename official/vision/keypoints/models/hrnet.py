import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

import official.vision.classification.resnet.model as resnet


class MakeOneBranch(M.Module):
    def __init__(self, block, in_channels, channels, num_blocks):
        super().__init__()
        _convs = [block(in_channels, channels)]
        for i in range(1, num_blocks):
            _convs.append(block(channels * block.expansion, channels))
        self.branch_convs = M.Sequential(*_convs)

    def forward(self, x):
        x = self.branch_convs(x)
        return x


class MakeBranches(M.Module):
    def __init__(self, block, base_channels, num_branches, num_blocks):
        super().__init__()

        self.num_branches = num_branches
        branch_convs = []
        for i in range(num_branches):
            branch_convs.append(
                MakeOneBranch(
                    block,
                    base_channels * 2 ** i,
                    base_channels * 2 ** i // block.expansion,
                    num_blocks,
                )
            )

        self.branch_convs = branch_convs

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branch_convs[i](x[i])
        return x


class FuseLayers(M.Module):
    def __init__(self, base_channels, num_branches, multi_scale_output=True):
        super().__init__()

        self.num_branches = num_branches
        self.num_outs = num_branches if multi_scale_output else 1

        branch_convs = {}
        for i in range(self.num_outs):
            for j in range(num_branches):
                if i < j:
                    convs = [
                        M.Conv2d(
                            base_channels * 2 ** j, base_channels * 2 ** i, 1, 1, 0
                        ),
                        M.BatchNorm2d(base_channels * 2 ** i),
                    ]
                    branch_convs["{}_to_{}".format(j, i)] = M.Sequential(*convs)
                elif i > j:
                    convs = []
                    for k in range(i - j - 1):
                        convs += [
                            M.Conv2d(
                                base_channels * 2 ** j, base_channels * 2 ** j, 3, 2, 1
                            ),
                            M.BatchNorm2d(base_channels * 2 ** j),
                            M.ReLU(),
                        ]
                    convs += [
                        M.Conv2d(
                            base_channels * 2 ** j, base_channels * 2 ** i, 3, 2, 1
                        ),
                        M.BatchNorm2d(base_channels * 2 ** i),
                    ]
                    branch_convs["{}_to_{}".format(j, i)] = M.Sequential(*convs)

        self.branch_convs = branch_convs

    def forward(self, x):

        x_fused = []
        for i in range(self.num_outs):
            for j in range(self.num_branches):
                if i < j:
                    up = self.branch_convs["{}_to_{}".format(j, i)](x[j])
                    up = F.nn.interpolate(up, x[i].shape[2:], align_corners=True)
                    if j == 0:
                        x_fused.append(up)
                    else:
                        x_fused[i] += up
                elif i == j:
                    if j == 0:
                        x_fused.append(x[j])
                    else:
                        x_fused[i] += x[j]
                elif i > j:
                    down = self.branch_convs["{}_to_{}".format(j, i)](x[j])
                    if j == 0:
                        x_fused.append(down)
                    else:
                        x_fused[i] += down

        return x_fused


class HRModule(M.Module):
    def __init__(
        self, block, base_channels, num_branches, num_blocks, multi_scale_output=True
    ):
        super().__init__()

        self._make_branches = MakeBranches(
            block, base_channels, num_branches, num_blocks
        )

        self.fuse_layers = FuseLayers(base_channels, num_branches, multi_scale_output)

    def forward(self, x):
        x = self._make_branches(x)
        x = self.fuse_layers(x)
        return x


class HRStage(M.Module):
    def __init__(
        self,
        block,
        base_channels,
        num_modules,
        num_branches,
        num_blocks,
        multi_scale_output=True,
    ):
        super(HRStage, self).__init__()

        self.num_branches = num_branches

        convs = []
        for i in range(num_modules - 1):
            convs.append(HRModule(block, base_channels, num_branches, num_blocks, True))
        convs.append(
            HRModule(block, base_channels, num_branches, num_blocks, multi_scale_output)
        )
        self.stage_convs = M.Sequential(*convs)

    def forward(self, x):
        assert len(x) == self.num_branches, "Expect {} branches, but given {}".format(
            self.num_branches, len(x)
        )
        x = self.stage_convs(x)
        return x


class HRNet(M.Module):
    def __init__(self, base_channels, cfg):
        super(HRNet, self).__init__()

        self.cfg = cfg
        self.keypoint_num = cfg.keypoint_num

        self.head = M.Sequential(
            M.Conv2d(3, 64, 3, 2, 1),
            M.BatchNorm2d(64),
            M.Conv2d(64, 64, 3, 2, 1),
            M.BatchNorm2d(64),
            M.ReLU(),
        )

        block = getattr(resnet, cfg.block_type["stage1"])
        self.branch1 = MakeOneBranch(block, 64, 64, cfg.num_blocks["stage1"])

        num_channels = 64 * block.expansion

        self.transition1 = M.Sequential(
            M.Conv2d(num_channels, base_channels, 3, 1, 1),
            M.BatchNorm2d(base_channels),
            M.ReLU(),
        )

        self.transition2 = M.Sequential(
            M.Conv2d(num_channels, base_channels * 2, 3, 2, 1),
            M.BatchNorm2d(base_channels * 2),
            M.ReLU(),
        )

        block = getattr(resnet, cfg.block_type["stage2"])
        self.stage2 = HRStage(
            block, base_channels, cfg.num_modules["stage2"], 2, cfg.num_blocks["stage2"]
        )

        self.transition3 = M.Sequential(
            M.Conv2d(base_channels * 2, base_channels * 2 ** 2, 3, 2, 1),
            M.BatchNorm2d(base_channels * 2 ** 2),
            M.ReLU(),
        )

        block = getattr(resnet, cfg.block_type["stage3"])
        self.stage3 = HRStage(
            block, base_channels, cfg.num_modules["stage3"], 3, cfg.num_blocks["stage3"]
        )

        self.transition4 = M.Sequential(
            M.Conv2d(base_channels * 2 ** 2, base_channels * 2 ** 3, 3, 2, 1),
            M.BatchNorm2d(base_channels * 2 ** 3),
            M.ReLU(),
        )

        block = getattr(resnet, cfg.block_type["stage4"])
        self.stage4 = HRStage(
            block,
            base_channels,
            cfg.num_modules["stage4"],
            4,
            cfg.num_blocks["stage4"],
            multi_scale_output=False,
        )

        self.last_layer = M.Conv2d(base_channels, self.keypoint_num, 1, 1, 0)

        self.initialize_weights()

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        M.init.zeros_(m.bias)
            if isinstance(m, M.ConvTranspose2d):
                M.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        M.init.zeros_(m.bias)
            if isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

    def calc_loss(self, images, heatmaps, heat_valid):
        out = self(images)
        valid = F.expand_dims((heat_valid > 0.1), [2, 3])
        label = heatmaps[:, -1]
        loss = F.nn.square_loss(out * valid, label * valid)
        return loss

    def predict(self, images):
        return self(images)

    def extract_features(self, x):
        x = self.head(x)
        x = self.branch1(x)

        x1 = self.transition1(x)
        x2 = self.transition2(x)

        x = [x1, x2]
        x = self.stage2(x)

        x3 = self.transition3(x[-1])
        x.append(x3)
        x = self.stage3(x)

        x4 = self.transition4(x[-1])
        x.append(x4)
        x = self.stage4(x)
        return x

    def forward(self, x):

        x = self.extract_features(x)

        out = self.last_layer(x[0])

        return out


class HRNet_Config:
    block_type = {
        "stage1": "Bottleneck",
        "stage2": "BasicBlock",
        "stage3": "BasicBlock",
        "stage4": "BasicBlock",
    }

    num_blocks = {"stage1": 4, "stage2": 4, "stage3": 4, "stage4": 4}

    num_modules = {
        "stage1": 1,
        "stage2": 1,
        "stage3": 4,
        "stage4": 3,
    }

    keypoint_num = 17


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/hrnet_w32_256x192_74_0.pkl"
)
def hrnet_w32(**kwargs):

    cfg = HRNet_Config()
    model = HRNet(base_channels=32, cfg=cfg, **kwargs)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/hrnet_w48_256x192_74_5.pkl"
)
def hrnet_w48(**kwargs):

    cfg = HRNet_Config()
    model = HRNet(base_channels=48, cfg=cfg, **kwargs)
    return model
