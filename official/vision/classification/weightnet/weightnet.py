import megengine.functional as F
import megengine.module as M

class WeightNet(M.Module):
    r"""Applies WeightNet to a standard convolution.

    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.

    M/G control the amount of parameters.
    """

    def __init__(self, inp, oup, ksize, stride):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = max(16, inp//16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = M.Conv2d(inp_gap, self.M*oup, 1, 1, 0, groups=1, bias=True)
        self.sigmoid = M.Sigmoid()
        self.wn_fc2 = M.Conv2d(self.M*oup, oup*inp*ksize*ksize, 1, 1, 0, groups=self.G*oup, bias=False)


    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, self.oup, self.inp, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=x_w.shape[0])
        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])
        return x

class WeightNet_DW(M.Module):
    r""" Here we show a grouping manner when we apply WeightNet to a depthwise convolution.

    The grouped fc layer directly generates the convolutional kernel, has fewer parameters while achieving comparable results.
    This layer has M/G*inp inputs, inp groups and inp*ksize*ksize outputs.

    """
    def __init__(self, inp, ksize, stride):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = max(16, inp//16)
        self.inp = inp
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = M.Conv2d(inp_gap, self.M//self.G*inp, 1, 1, 0, groups=1, bias=True)
        self.sigmoid = M.Sigmoid()
        self.wn_fc2 = M.Conv2d(self.M//self.G*inp, inp*ksize*ksize, 1, 1, 0, groups=inp, bias=False)


    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, 1, 1, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=x_w.shape[0])
        x = x.reshape(-1, self.inp, x.shape[2], x.shape[3])
        return x
