from .faster_rcnn_res50_coco_1x_800size import faster_rcnn_res50_coco_1x_800size
from .faster_rcnn_res50_coco_1x_800size_syncbn import faster_rcnn_res50_coco_1x_800size_syncbn
from .faster_rcnn_res101_coco_2x_800size import faster_rcnn_res101_coco_2x_800size
from .retinanet_res50_coco_1x_800size import retinanet_res50_coco_1x_800size
from .retinanet_res50_coco_1x_800size_syncbn import retinanet_res50_coco_1x_800size_syncbn
from .retinanet_res101_coco_2x_800size import retinanet_res101_coco_2x_800size

_EXCLUDE = {}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
