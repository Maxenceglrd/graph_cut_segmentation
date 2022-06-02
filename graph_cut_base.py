from collections import defaultdict
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

COLORS = {
    0: (0, 0, 255),  # dogs
    1: (0, 0, 0),  # background
    2: (255, 0, 0),  # cats
}
Node = Tuple[int, int]


class GraphCutBase:
    """
    Base class used by GraphCutBinary and GraphCutAlphaExpansion
    containing helper function (namely the interactive annotation part)
    """

    def __init__(
        self,
        image: np.ndarray,
        unary_type: str = "l2_dist",
        sigma: float = 1.0,
        lmbda: float = 1.0,
        weight_binary: float = 1.0,
    ):
        self.image = image
        if unary_type == "l2_dist":
            self.yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) / 255
        else:
            # no normalization needed to have better conditioned covariance matrices
            self.yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        self.unary_type = unary_type
        self.sigma = sigma
        self.lmbda = lmbda
        self.weight_binary = weight_binary

        self.foreground_pixels = []
        self.background_pixels = []

        self.annotated_pixels_locations = defaultdict(list)
        self.annotated_pixels = defaultdict(list)

        self.annotating_label = False
        self.current_label = 0
        self.n_labels = None

        self.means = None
        self.covariances = None

        self.graph = None

    def annotate_image(self) -> None:
        """
        Main function annotate the image
        """
        cv2.namedWindow("Image annotation")
        cv2.setMouseCallback("Image annotation", self.annotate_on_image)

        is_annotating = True
        while is_annotating:
            cv2.imshow("Image annotation", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(20)
            if key == 27:
                break
        self.n_labels = self.current_label

    def annotate_on_image(self, event, x: int, y: int, flags, param) -> None:
        """
        Helper function to annotate_image function, which is an event callback used
        to handle mouse movement / clicks
        :param event: cv2 Event (
        :param x: int, horizontal location of the pixel
        :param y: int, vertical location of the pixel
        :param flags: Additional cv2 flags, not used but need to be in the signature
        :param param: Additional cv2 param, not used but need to be in the signature
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.annotating_label:
                self.current_label += 1
                self.annotating_label = False
            else:
                self.annotating_label = True
                self.annotated_pixels_locations[self.current_label].append((y, x))
                # self.annotated_pixels[self.current_label].append(self.yuv_image[y, x])
                cv2.rectangle(
                    self.image,
                    (x - 1, y - 1),
                    (x + 1, y + 1),
                    COLORS[self.current_label],
                    -1,
                )

        elif event == cv2.EVENT_MOUSEMOVE and self.annotating_label:
            self.annotated_pixels_locations[self.current_label].append((y, x))
            # self.annotated_pixels[self.current_label].append(self.yuv_image[y, x])
            cv2.rectangle(
                self.image,
                (x - 1, y - 1),
                (x + 1, y + 1),
                COLORS[self.current_label],
                -1,
            )

    def compute_statistics(self) -> None:
        """
        Compute the annotations statistic after the interactive annotations
        (compute means and covariances matrices for each label)
        """
        self.means = np.zeros((self.n_labels, 3))
        self.covariances = np.zeros((self.n_labels, 3, 3))
        for label in range(self.n_labels):
            self.annotated_pixels[label] = [
                self.yuv_image[y, x]
                for (y, x) in self.annotated_pixels_locations[label]
            ]
            pixel_values = np.array(self.annotated_pixels[label])
            self.means[label] = np.mean(pixel_values, axis=0)
            self.covariances[label] = np.cov(pixel_values.T)

    def plot_densities(self, component: int = 0) -> None:
        """
        Plot the pixels densities (approximated by normal distribution) for each label
        :param component: int, color component to be represented in the YUV format (default 0 so Y channel)
        """
        n_labels = self.covariances.shape[0]

        df = pd.DataFrame(
            np.concatenate(
                [
                    np.array(
                        [
                            [i, x]
                            for x in np.random.normal(
                                loc=self.means[i][component],
                                scale=self.covariances[i][component, component],
                                size=1000,
                            )
                        ]
                    )
                    for i in range(n_labels)
                ]
            ),
            columns=["component", "intensity"],
        )
        sns.displot(df, kind="kde", hue="component", x="intensity")
        plt.title("Distribution of the different components")
        plt.show()

    def build_graph(self) -> None:
        """
        Main function to build the directed graph that will be used for Graph Cut
        """
        self.graph = nx.DiGraph()

        width, height = self.image.shape[0], self.image.shape[1]
        self.graph.add_nodes_from([(x, y) for x in range(width) for y in range(height)])

        self.graph.add_edges_from(
            [((x, y), (x + 1, y)) for x in range(width - 1) for y in range(height)]
        )
        self.graph.add_edges_from(
            [((x, y), (x, y + 1)) for x in range(width) for y in range(height - 1)]
        )

        self.graph.add_edges_from(
            [((x + 1, y), (x, y)) for x in range(width - 1) for y in range(height)]
        )
        self.graph.add_edges_from(
            [((x, y + 1), (x, y)) for x in range(width) for y in range(height - 1)]
        )

    @staticmethod
    def normal_distribution(
        p: np.ndarray, mean: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """
        Return PDF of normal distribution according to the class label
        :param p: pixel values in the YUV format
        :param mean: pixels mean for a given label
        :param sigma: pixels covariance matric for a given label
        :return: array, normal PDF for each pixels in p
        """
        sigma_inv = np.linalg.pinv(sigma)
        normalize_factor = 1 / np.sqrt(((2 * np.pi) ** 3) * np.linalg.det(sigma))
        return normalize_factor * np.exp(-0.5 * (p - mean).T @ sigma_inv @ (p - mean))

    def compute_unary_terms(self):
        width, height = self.image.shape[0], self.image.shape[1]
        initial_labeling = np.zeros((width, height))
        unary_costs = np.zeros((width, height, self.n_labels))
        for v in self.graph.nodes():
            for label in range(self.n_labels):
                p = self.yuv_image[v]
                mean = self.means[label]
                if self.unary_type == "l2_dist":
                    unary_costs[v][label] = self.lmbda * np.linalg.norm(p - mean) ** 2
                else:
                    sigma = self.covariances[label]
                    unary_costs[v][label] = -self.lmbda * np.log(
                        self.normal_distribution(p, mean, sigma)
                    )
            initial_labeling[v] = np.argmin(unary_costs[v])
        return unary_costs, initial_labeling

    def run_segmentation(self):
        """
        Main function to be implemented by each children class
        to run the end to end segmentation using graph cut techniques
        """
        raise Exception("Not Implemented Error")
