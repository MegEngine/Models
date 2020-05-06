import megengine.functional as F
from megengine.core import Tensor


def box_overlap_ignore_opr(boxes1: Tensor, boxes2: Tensor, ignore_label=-1) -> Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    box = boxes1
    gt = boxes2
    target_shape = (boxes1.shapeof(0), boxes2.shapeof(0), 4)

    b_box = F.add_axis(boxes1, 1).broadcast(*target_shape)
    b_gt = F.add_axis(boxes2[:, :4], 0).broadcast(*target_shape)

    iw = F.minimum(b_box[:, :, 2], b_gt[:, :, 2]) - F.maximum(
        b_box[:, :, 0], b_gt[:, :, 0]
    )
    ih = F.minimum(b_box[:, :, 3], b_gt[:, :, 3]) - F.maximum(
        b_box[:, :, 1], b_gt[:, :, 1]
    )
    inter = F.maximum(iw, 0) * F.maximum(ih, 0)

    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    area_target_shape = (box.shapeof(0), gt.shapeof(0))

    b_area_box = F.add_axis(area_box, 1).broadcast(*area_target_shape)
    b_area_gt = F.add_axis(area_gt, 0).broadcast(*area_target_shape)

    union = b_area_box + b_area_gt - inter

    overlaps_normal = F.maximum(inter / union, 0)
    overlaps_ignore = F.maximum(inter / b_area_box, 0)

    # TODO comment next lines since gt is all valid
    gt_ignore_mask = F.add_axis(F.equal(gt[:, 4], ignore_label), 0).broadcast(*area_target_shape)
    overlaps_normal *= (1 - gt_ignore_mask)
    overlaps_ignore *= gt_ignore_mask
    return overlaps_normal, overlaps_ignore
