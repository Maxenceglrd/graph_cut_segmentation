from typing import Tuple, List

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from coco_dataset import Dataset
from evaluation import compute_iou
from utils import coco_names


class MaskRCNNSegmentation:
    """Implementation of the MaskRCNN using a pretrained model"""

    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91
        )
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_outputs(
        self, image: np.ndarray, threshold: float = 0.85
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get the predicted masks and labels for a given image
        :param image: array representing the image
        :param threshold: float, detection threshold
        :return: List of masks and list of labels
        """
        # Acknowledgement to https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
        # for the code for this method
        with torch.no_grad():
            outputs = self.model(image)
        scores = list(outputs[0]["scores"].detach().cpu().numpy())
        kept_indices = [scores.index(i) for i in scores if i > threshold]
        masks = (outputs[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
        masks = masks[: len(kept_indices)]
        labels = [coco_names[k] for k in outputs[0]["labels"]]
        return masks, labels

    @staticmethod
    def get_segmentation_map(
        image: np.ndarray, masks: List[np.ndarray], labels: List[str]
    ) -> np.ndarray:
        """
        Fuse the mask returned by the different prediction of the MaskRCNN
        :param image: array representing the image
        :param masks: list of predicted mask
        :param labels: list of labels
        :return: array being the final segmentation result
        """
        height, width, _ = image.shape
        mask = np.zeros((height, width, 3))
        for i in range(len(masks)):
            current_mask = masks[i]
            current_mask = current_mask[:, :, np.newaxis].repeat(3, 2)
            if labels[i] == "dog":
                color = (0, 0, 255)
            elif labels[i] == "cat":
                color = (255, 0, 0)
            else:
                color = (0, 0, 0)
            current_mask = current_mask * color
            mask += current_mask
        return mask

    def run_rcnn(self, image: np.ndarray) -> np.ndarray:
        """
        Main function to get the segmentation for a given image
        :param image: Array representing the image
        :return: Array represented the segmentation
        """
        orig_image = image.copy()
        image = self.transform(image)
        image = image.unsqueeze(0)

        masks, labels = self.get_outputs(image)
        total_mask = self.get_segmentation_map(orig_image, masks, labels)
        return total_mask


if __name__ == "__main__":
    dataset = Dataset()
    rcnn = MaskRCNNSegmentation()
    results_binary = {}
    results_multi_label = {}
    for i in range(len(dataset.dataset)):
        segmentation = rcnn.run_rcnn(dataset.images[i])
        mask = dataset.segmentation_masks[i]
        n_labels = dataset.n_labels_list[i]
        iou = compute_iou(segmentation, mask, n_labels)
        result = {
            "iou": iou,
            "segmentation": segmentation,
            "mask": mask,
            "n_labels": n_labels,
        }
        if n_labels == 1:
            results_binary[i] = result
        else:
            results_multi_label[i] = result
