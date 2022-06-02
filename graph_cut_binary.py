import networkx as nx
import numpy as np

from ford_fulkerson import MinimumCutFordFulkerson
from graph_cut_base import GraphCutBase, COLORS, Node


class GraphCutBinary(GraphCutBase):
    """
    Implementation of the binary segmentation using Graph Cut
    """

    def __init__(
        self,
        image: np.ndarray,
        unary_type: str = "l2_dist",
        sigma: float = 1.0,
        lmbda: float = 1.0,
        weight_binary: float = 1.0,
    ):
        super().__init__(image, unary_type, sigma, lmbda, weight_binary)

    def run_segmentation(self) -> np.ndarray:
        """
        Main function that will get the segmentation given an image
        from the graph building to the run of the minimum cut algorithù
        :return: Array representing the segmented image
        """
        # Building the graph
        print("------------ Build graph ------------")
        self.build_graph()

        # Compute unary cost, use it for initializing labels
        unary_costs, initial_labeling = self.compute_unary_terms()

        # Compute binary cost
        max_binary = self.compute_binary_terms(initial_labeling)

        # Add sink and source nodes
        s_node = (self.image.shape[0], self.image.shape[1])
        t_node = (self.image.shape[0], self.image.shape[1] + 1)
        self.graph.add_nodes_from([s_node, t_node])

        # Set weights on edges to sink and source nodes
        self.set_edges_weights(s_node, t_node, max_binary, unary_costs)

        # Run Max flow algorithm and get the segmented image
        print("------------ Run min cut algorithm ------------")
        segmentation = self.get_min_cut_and_label(s_node, t_node)

        return segmentation

    def get_min_cut_and_label(
        self, s: Node, t: Node, use_own_maxflow: bool = False
    ) -> np.ndarray:
        """
        Main function that finds the minimum cut on the defined graph
        :param s: source node (represented as a 2D point to stick with the pixels locations notation)
        :param t: sink node
        :param use_own_maxflow: bool, if True it will use the hand-made Ford Fulkerson implementation
        :return: Segmentation array
        """
        # By default, we will be using the networkx implementation
        # of minimum cut as it is more efficient (with respect
        # to sparse matrices manipulations)
        segmentation = np.zeros(
            (self.image.shape[0], self.image.shape[1], 3), dtype=np.int32
        )
        if use_own_maxflow:
            adjacency_matrix = nx.adjacency_matrix(self.graph, weight="weight")
            s_index = adjacency_matrix.shape[0] - 2
            t_index = adjacency_matrix.shape[0] - 1
            min_cut_algo = MinimumCutFordFulkerson(
                adjacency_matrix, with_sparse_matrix=True
            )
            partition = min_cut_algo.find_minimum_cut(s_index, t_index)
            S, T = partition
            for node in self.graph.nodes():
                if node not in [s, t]:
                    node_index = node[0] * self.image.shape[0] + node[1]
                    if node_index in S:
                        segmentation[node] = COLORS[0]  # label 0 corresponds to object
        else:
            cut_value, partition = nx.minimum_cut(self.graph, s, t, capacity="weight")
            S, T = partition
            for node in self.graph.nodes():
                if node in S and node not in [s, t]:
                    segmentation[node] = COLORS[0]  # label 0 corresponds to object
        return segmentation

    def compute_binary_terms(self, labeling: np.ndarray) -> float:
        """
        Compute the binary potentials
        :param labeling: current labeling
        :return: Max binary cost
        """
        max_binary = -float("inf")
        for edge in self.graph.edges():
            source = edge[0]
            target = edge[1]
            p = self.yuv_image[source]
            q = self.yuv_image[target]
            different_label = int(labeling[source] != labeling[target])
            binary_cost = (
                self.weight_binary
                * np.exp(-1 / (2 * (self.sigma ** 2)) * (np.linalg.norm(p - q) ** 2))
                / np.linalg.norm(np.array(source) - np.array(target))
            )
            self.graph[source][target]["weight"] = binary_cost * different_label
            self.graph[target][source]["weight"] = binary_cost * different_label
            if binary_cost > max_binary:
                max_binary = binary_cost
        max_binary += 1
        return max_binary

    def set_edges_weights(
        self,
        s_node: Node,
        t_node: Node,
        max_binary: float,
        unary_costs: np.ndarray,
    ) -> None:
        """
        Set the unary cost between nodes and source / sink
        :param s_node: source node
        :param t_node: sink nnode
        :param max_binary: float, maximum binary potentials
        :param unary_costs: array of binary potentials
        """
        for v in self.graph.nodes():
            if v != s_node and v != t_node:
                self.graph.add_edges_from([(s_node, v), (v, t_node)])

                # v has been annotated as part of label 0
                if v in self.annotated_pixels_locations[0]:
                    unary_t = 0
                    unary_s = max_binary

                # v has been annotated as part of label 1
                elif v in self.annotated_pixels_locations[1]:
                    unary_t = max_binary
                    unary_s = 0

                else:
                    unary_s = unary_costs[v][1]
                    unary_t = unary_costs[v][0]
                self.graph[s_node][v]["weight"] = unary_s
                self.graph[v][t_node]["weight"] = unary_t
