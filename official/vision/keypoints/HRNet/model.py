import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import official.vision.classification.resnet.model as B

class Make_One_Branch(M.Module):
    def __init__(self, block, in_channels, channels, num_blocks):
        super(Make_One_Branch, self).__init__()
        _convs = [block(in_channels, channels)]
        for i in range(1, num_blocks):
            _convs.append(
                block(channels*block.expansion, channels)
            )
        self.branch_convs = M.Sequential(*_convs)

    def forward(self, x):
        x = self.branch_convs(x)
        return x


class Make_Branches(M.Module):
    def __init__(self, block, base_channels, num_branches, num_blocks):
        super(Make_Branches, self).__init__()

        self.num_branches = num_branches
        branch_convs = []
        for i in range(num_branches):
            branch_convs.append(
                Make_One_Branch(block, base_channels*2**i, base_channels*2**i, num_blocks)
            )

        self.branch_convs = branch_convs

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branch_convs[i](x[i])
        return x      
        

class Fuse_Layers(M.Module):
    def __init__(self, base_channels, num_branches):
        super(Fuse_Layers, self).__init__()

        self.num_branches = num_branches
        
        branch_convs = {}
        for i in range(num_branches):
            for j in range(num_branches):
                if i < j:
                    _convs = [
                        M.Conv2d(base_channels*2**j, base_channels*2**i, 1, 1, 0),
                        M.BatchNorm2d(base_channels*2**j)
                    ]
                    branch_convs['{}_to_{}'.format(j, i)] = M.Sequential(*_convs)
                elif i > j:
                    _convs = []
                    for k in range(i - j -1):
                        _convs += [
                            M.Conv2d(base_channels*2**j, base_channels*2**j, 3, 2, 1),
                            M.BatchNorm2d(base_channels*2**j),
                            M.ReLU()
                        ]
                    _convs += [
                        M.Conv2d(base_channels*2**j, base_channels*2**i, 3, 2, 1),
                        M.BatchNorm2d(base_channels*2**j)
                        ]
                    branch_convs['{}_to_{}'.format(j, i)] = M.Sequential(*_convs))

        self.branch_convs = branch_convs

    def forward(self, x):

        x_fused = []
        for i in range(self.num_branches):
            for j in range(self.num_branches):
                if j == 0:
                    if i == j:
                        x_fused.append(x[j])
                    else:
                        x_fused.append(
                            self.branch_convs['{}_to_{}'.format(j, i)](x[j])
                        )
                else:
                    if i < j:
                        up = self.branch_convs['{}_to_{}'.format(j, i)](x[j])
                        up = F.interpolate(up, (x[i].sizeof(2), x[i].sizeof(3)))
                        x_fused[i] += up
                    elif i == j:
                        x_fused[i] += x[j]
                    
                    elif i > j:
                        down = self.branch_convs['{}_to_{}'.format(j, i)](x[j])
                        x_fused[i] += down

        return x_fused


class HRModule(M.Module):
    def __init__(self, block, base_channels, num_branches, num_blocks):
        super(HRModule, self).__init__()

        self._make_branches = Make_Branches(block, base_channels, num_branches, num_blocks)

        self._fuse_layers = Fuse_Layers(base_channels, num_branches)

    def forward(self, x):
        x = self._make_branches(x)
        x = self._fuse_layers(x)
        return x

class HRStage(M.Module):
    def __init__(self, block, base_channels, num_modules, num_branches, num_blocks):

        _convs = []
        for i in range(num_modules):
            _convs.append(
                HRModule(block, base_channels, num_branches, num_blocks)
            )
        self.stage_convs = M.Sequential(*_convs)

    def forward(self, x):
        assert len(x) == self.num_branches "Expect {} branches, but given {}".format(self.num_branches, len(x))
        x = self.satge_convs(x)
        return x

class HRNet(M.Module):
    def __init__(self, cfg):

        self.cfg = args
        base_channels = cfg.base_channels

        self.head = M.Sequential(
            M.Conv2d(3, 64, 3, 2, 1),
            M.BatchNorm2d(64),
            M.Conv2d(3, 64, 3, 2, 1),
            M.BatchNorm2d(64),
            M.ReLU()
        )

        block = getattr(B, cfg.block_type['stage1'])
        self.branch1 = Make_One_Branch(B, 64, 64, cfg.num_blocks['stage1'])

        num_channels = 64 * block.expansion

        self.transistion1 = M.Sequential(
            M.Conv2d(num_channels, base_channels, 3, 1, 1),
            M.BatchNorm2d(base_channels),
            M.ReLU()
        )

        self.transistion2 = M.Sequential(
            M.Conv2d(num_channels, base_channels*2, 3, 2, 1),
            M.BatchNorm2d(base_channels*2),
            M.ReLU()
        )

        block = getattr(B, cfg.block_type['stage2'])
        self.stage2 = HRStage(block, base_channels, cfg.num_modules['stage2'], 2, cfg.num_blocks['stage2'])

        self.transistion3 = M.Sequential(
            M.Conv2d(base_channels*2, base_channels*2**2, 3, 2, 1),
            M.BatchNorm2d(base_channels*2**2),
            M.ReLU()
        )

        block = getattr(B, cfg.block_type['stage3'])
        self.stage2 = HRStage(block, base_channels, cfg.num_modules['stage3'], 3, cfg.num_blocks['stage3'])

        self.transistion4 = M.Sequential(
            M.Conv2d(base_channels*2**2, base_channels*2**3, 3, 2, 1),
            M.BatchNorm2d(base_channels*2**3),
            M.ReLU()
        )

        block = getattr(B, cfg.block_type['stage4'])
        self.stage2 = HRStage(block, base_channels, cfg.num_modules['stage4'], 4, cfg.num_blocks['stage4'])

        last_layer_channels = base_channels * 2**(0 + 1 + 2 + 3)
        self.last_layer = M.Sequential(
            M.Conv2d(last_layer_channels, last_layer_channels, 1, 1, 0),
            M.BatchNorm2d(last_layer_channels),
            M.ReLU(),
            M.Conv2d(last_layer_channels, cfg.num_classes, 3, 1, 1)
        )

        def forward(self, x):

            x = self.head(x)
            x = self.branch1(x)

            x1 = self.transistion1(x)
            x2 = self.transistion2(x)

            x = [x1, x2]
            x = self.stage2(x)

            x3 = self.transistion3(x[-1])
            x.append(x3)
            x = self.stage3(x)

            x4 = self.transistion4(x[-1])
            x.append(x4)
            x = self.stage4(x)

            for ind, xx in enumerate(x):
                if ind == 0:
                    shape = (x[ind].sizeof(2), x[ind].sizeof(3))
                else:
                    x[ind] = F.interpolate(xx, shape)

            x = F.concat(x, 1)
            x = self.last_layer(x)
            
            return x

     


        



    
