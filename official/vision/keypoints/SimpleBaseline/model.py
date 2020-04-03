import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import official.vision.classification.resnet.model as B

class SimpelBaseline(M.Module):
    def __init__(
        self,
        args
    ):

    super(SimpelBaseline, self).__init__()

    self.Backbone = getattr(B, args.backbone)
    self.deconv_layers = self._make_deconv_layers(args.num_deconv_layers, args.deconv_kernel_size)


    def _make_deconv(self, num_deconv_layers, deconv_kernel_size):
        if deconv_kernel_size == 