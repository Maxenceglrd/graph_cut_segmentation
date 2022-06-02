from math import floor
from typing import Union, Tuple, List, Dict

import cv2
import fiftyone.zoo as foz
import numpy as np

from graph_cut_base import COLORS


class Dataset:
    """
    Class to handle the loading of the COCO dataset
    """

    def __init__(
        self,
        dataset_dir: Union[None, str] = "./coco-dataset",
        dataset_name: Union[None, str] = "coco-dataset",
        segmentation_labels=None,
    ):
        if segmentation_labels is None:
            self.segmentation_labels = {"dog": 0, "cat": 2}
            # Label 1 is kept for the background to be consistent with the binary case
        self.scale_percent = 60  # percent of original size

        self.dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            classes=["cat", "dog"],
            label_types=["segmentations"],
            max_samples=25,
        )

        (
            self.images,
            self.segmentation_masks,
            self.n_labels_list,
        ) = self.process_dataset()
        self.resize_data()

    def process_dataset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Main function to process the COCO dataset
        :return:
        images: List of arrays containing the loaded images in RGB format
        segmentation_masks: List of arrays containing the ground truth segmentations
        n_labels_list: List of integers containing the number of labels in each image
        """
        images = []
        segmentation_masks = []
        n_labels_list = []
        for sample in list(self.dataset.iter_samples()):
            image = cv2.cvtColor(cv2.imread(sample["filepath"]), cv2.COLOR_BGR2RGB)
            kept_annotations = [
                d
                for d in sample.ground_truth["detections"]
                if d["label"] in self.segmentation_labels.keys()
            ]
            n_labels = len(np.unique([d["label"] for d in kept_annotations]))
            segmentation_mask = self.get_segmentation_mask(
                image, kept_annotations, n_labels
            )
            images.append(image)
            segmentation_masks.append(segmentation_mask)
            n_labels_list.append(n_labels)
        return images, segmentation_masks, n_labels_list

    def get_segmentation_mask(
        self, image: np.ndarray, annotations: List[Dict], n_labels: int
    ) -> np.ndarray:
        """

        :param image: np.ndarray (current image)
        :param annotations: dictionary of annotations (from COCO dataset)
        It namely contains the segmentation mask and the bounding boxes
        to locate the mask on the full image
        :param n_labels: number of labels in the image
        :return: Segmentation mask
        """
        height, width, _ = image.shape
        mask = np.zeros((height, width, 3))
        for annotation in annotations:
            bounding_box = annotation["bounding_box"]
            x_1, y_1, bb_width, bb_height = bounding_box
            x_1 = floor(x_1 * width)
            y_1 = floor(y_1 * height)

            current_mask = np.array(annotation["mask"], dtype=np.int32).T
            current_mask = current_mask[:, :, np.newaxis].repeat(3, 2)
            x_2 = current_mask.shape[0] + x_1
            y_2 = current_mask.shape[1] + y_1
            if n_labels == 1:
                mask_color = COLORS[0]
            else:
                mask_color = COLORS[self.segmentation_labels[annotation["label"]]]
            current_mask = current_mask * mask_color
            mask[y_1:y_2, x_1:x_2, :] = np.swapaxes(current_mask, 0, 1)
        return mask

    def resize_data(self) -> None:
        """
        Resize the images and the mask given the
        class attributed self.scale_percent
        """
        for i, (image, segmentation_mask) in enumerate(
            zip(self.images, self.segmentation_masks)
        ):
            width = int(image.shape[1] * self.scale_percent / 100)
            height = int(image.shape[0] * self.scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            segmentation_mask = np.array(
                cv2.resize(segmentation_mask, dim, interpolation=cv2.INTER_NEAREST),
                dtype=np.int32,
            )
            self.images[i] = image
            self.segmentation_masks[i] = segmentation_mask
