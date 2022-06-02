import numpy as np

from graph_cut_base import COLORS


def compute_iou(
    segmentation: np.ndarray,
    mask: np.ndarray,
    n_labels,
) -> float:
    """
    Compute Intersection over Union metric
    :param segmentation: array, predicted segmentation
    :param mask: array, ground truth segmentation
    :param n_labels: int, number of labels
    :return: float, Ious
    """
    iou = 0
    for i in range(n_labels + 1):
        # n_labels being the number of objects that are not background
        color = COLORS[i]
        indices_segmentation = np.where(np.all(segmentation == color, axis=-1))
        coordinates_segmentation = set(
            zip(indices_segmentation[0], indices_segmentation[1])
        )

        indices_mask = np.where(np.all(mask == color, axis=-1))
        coordinates_mask = set(zip(indices_mask[0], indices_mask[1]))

        intersection_length = len(
            coordinates_segmentation.intersection(coordinates_mask)
        )
        union_length = len(coordinates_segmentation.union(coordinates_mask))
        iou += intersection_length / union_length
    return iou / (n_labels + 1)
