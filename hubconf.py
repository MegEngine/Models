import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "official/vision/classification"))
from official.vision.classification.resnet.model import (
    BasicBlock,
    Bottleneck,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
)
from official.vision.classification.shufflenet.model import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
sys.path.pop(0)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "official/nlp/bert"))
from official.nlp.bert.model import (
    uncased_L_12_H_768_A_12,
    cased_L_12_H_768_A_12,
    uncased_L_24_H_1024_A_16,
    cased_L_24_H_1024_A_16,
    chinese_L_12_H_768_A_12,
    multi_cased_L_12_H_768_A_12,
    wwm_uncased_L_24_H_1024_A_16,
    wwm_cased_L_24_H_1024_A_16,
)
sys.path.pop(0)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "official/vision/detection"))
from official.vision.detection.retinanet_res50_1x_800size import (
    retinanet_res50_1x_800size,
    RetinaNet,
)
from official.vision.detection.tools.test import DetEvaluator
sys.path.pop(0)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "official/vision/segmentation"))
from official.vision.segmentation.deeplabv3plus import (
    deeplabv3plus_res101,
    DeepLabV3Plus,
)
sys.path.pop(0)
