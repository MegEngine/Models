import megengine.functional as F
import megengine.module as M

class FReLU(M.Module):
    r""" FReLU applied to light-weight CNN.

    Normally we have a funnel size of kxk.
    Here we show FReLU with a funnel size of 1x3+3x1 to save FLOPs.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu1 = M.Conv2d(in_channels, in_channels, (1,3), 1, (0,1), groups=in_channels, bias=False)
        self.conv_frelu2 = M.Conv2d(in_channels, in_channels, (3,1), 1, (1,0), groups=in_channels, bias=False)
        self.bn1 = M.BatchNorm2d(in_channels)
        self.bn2 = M.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu1(x)
        x1 = self.bn1(x1)
        x2 = self.conv_frelu2(x)
        x2 = self.bn2(x2)
        x = F.maximum(x, x1+x2)
        return x
