from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

from coco_dataset import Dataset
from evaluation import compute_iou
from graph_cut_binary import GraphCutBinary
from graph_cut_alpha import GraphCutAlphaExpansion

algorithms = {"binary": GraphCutBinary, "alpha-expansion": GraphCutAlphaExpansion}


def main_graph_cut_segmentation(
    dataset: Dataset,
    image_index: int,
    algorithm: str,
    unary_cost: str,
    sigma: int,
    lmbda: int,
    weight_binary: int,
    plot_densities: bool = True,
    plot_result: bool = True,
    load_precomputed: bool = False,
    save_path: str = None,
    auto_select_algo: bool = False,
) -> Dict:
    """
    Main function used by the main.py file to run the graph cut based segmentation on an image
    :param dataset: COCO Dataset instance of the Dataset class implemented in coco_dataset.py file
    :param image_index: int, image index in the COCO dataset
    :param algorithm: str, "binary" or "alpha-expansion"
    :param unary_cost: str, "normal" or "l2_dist"
    :param sigma: float, bandwidth for the binary potentials
    :param lmbda: float, weight of the unary potentials
    :param weight_binary: float, weight of the binary potentials
    :param plot_densities: bool, if True, it will plot the annotations densities
    across the labels (on the first dimension of the colors)
    :param plot_result: bool, if True, it will plot the segmentation result
    :param load_precomputed: bool, if True, it will use the precomputed annotation
    :param save_path: str, path to save the annotation
    :param auto_select_algo: bool, if True, it will select either "binary" algorithm if this is a binary problem
    or "alpha-expansion" if multi-label problem (it is given by the information contained in the dataset)
    :return: Dictionary containing the IoU, the predicted segmentation, the ground truth segmentation and the number
    of labels.
    """
    if load_precomputed:
        with open(save_path, "rb") as f:
            graph_cut = pickle.load(f)
            graph_cut.sigma = sigma
            graph_cut.lmbda = lmbda
            graph_cut.unary_type = unary_cost
            graph_cut.weight_binary = weight_binary
            image = np.copy(dataset.images[image_index])
            if unary_cost == "normal":
                graph_cut.yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else:
                graph_cut.yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) / 255
    else:
        if auto_select_algo:
            algorithm = (
                "binary"
                if dataset.n_labels_list[image_index] == 1
                else "alpha-expansion"
            )
        print("------------ Annotate image ------------")
        graph_cut = algorithms[algorithm](
            np.copy(dataset.images[image_index]),
            unary_type=unary_cost,
            sigma=sigma,
            lmbda=lmbda,
            weight_binary=weight_binary,
        )
        graph_cut.annotate_image()
    graph_cut.compute_statistics()

    if plot_densities:
        print("------------ Plot graph ------------")
        graph_cut.plot_densities(0)

    segmentation = graph_cut.run_segmentation()
    mask = dataset.segmentation_masks[image_index]
    n_labels = dataset.n_labels_list[image_index]

    # Evaluation
    iou = compute_iou(segmentation, mask, n_labels)

    if plot_result:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle("Graph Cut segmentation result")
        ax1.imshow(dataset.images[image_index])
        ax1.axis("off")
        ax2.imshow(segmentation)
        ax2.axis("off")
        ax3.imshow(mask)
        ax3.axis("off")
        plt.show()

    with open(save_path, "wb") as f:
        pickle.dump(graph_cut, f)

    return {
        "iou": iou,
        "segmentation": segmentation,
        "mask": mask,
        "n_labels": n_labels,
    }


def separate_binary_from_multi(
    results: Dict, n_labels_list: List[int]
) -> Tuple[Dict, Dict]:
    """
    Helper function to seaprate the results between
    binary results and multi-labels results
    :param results: List of result (each one having the format
    return by the function main_graph_cut_segmentation above)
    :param n_labels_list: List of integers representing the number of labels
    for each image
    :return: Tuple of dictionary, respectively binary and multi-labels results
    """
    results_binary = {}
    results_alpha_expansion = {}
    for i, result in results.items():
        if n_labels_list[i] == 1:
            results_binary[i] = result
        else:
            results_alpha_expansion[i] = result
    return results_binary, results_alpha_expansion
