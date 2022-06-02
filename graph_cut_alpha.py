from typing import Tuple

import networkx as nx
import numpy as np

from graph_cut_base import GraphCutBase, COLORS, Node


class GraphCutAlphaExpansion(GraphCutBase):
    def __init__(
        self,
        image: np.ndarray,
        unary_type: str = "l2_dist",
        sigma: float = 1.0,
        lmbda: float = 1.0,
        weight_binary: float = 1.0,
        n_iter: int = 1,
    ):
        super().__init__(image, unary_type, sigma, lmbda, weight_binary)
        self.n_iter = n_iter
        self.alpha = None

        # Define helping list to cope with the additional / removal of auxiliary nodes
        self.auxiliary_nodes = []
        self.aux_to_sink_edged = []
        self.edges_to_add = []

    def run_segmentation(self):
        """
        Main function that will get the segmentation given an image
        from the graph building to the run of the minimum cut algorithù
        :return: Array representing the segmented image
        """
        # Build the graph
        print("------------ Build graph ------------")
        self.build_graph()

        # Compute unary cost, use it for initializing labels
        width, height = self.image.shape[0], self.image.shape[1]
        unary_costs, current_labeling = self.compute_unary_terms()
        segmentation = np.zeros((width, height, 3), dtype=np.int32)
        for x in range(width):
            for y in range(height):
                segmentation[x, y] = COLORS[int(current_labeling[x, y])]

        # Set weights on edges to sink and source nodes
        for k in range(self.n_iter):
            for label in range(self.n_labels):
                self.alpha = label

                # Add sink and source nodes
                s_node = (self.image.shape[0], self.image.shape[1])
                t_node = (self.image.shape[0], self.image.shape[1] + 1)
                self.graph.add_nodes_from([t_node, s_node])

                # Add auxiliary nodes
                self.update_auxiliary_nodes(current_labeling, t_node)

                # Compute binary cost
                self.compute_binary_terms(current_labeling, t_node)

                # Set sink / source edges weights
                self.set_edges_weights(t_node, s_node, current_labeling, unary_costs)
                # Run Max flow algorithm and get the segmented image
                print(
                    f"------------ Run min cut algorithm: iter {k}, alpha: {label} ------------"
                )
                current_labeling, segmentation = self.get_min_cut_and_label(
                    current_labeling, segmentation, s_node, t_node
                )

                self.graph.remove_nodes_from([t_node, s_node])
                self.graph.remove_edges_from(self.edges_to_add)
                self.graph.remove_edges_from(self.aux_to_sink_edged)

        return segmentation

    def get_min_cut_and_label(
        self,
        current_labeling: np.ndarray,
        segmentation: np.ndarray,
        s: Node,
        t: Node,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main function that finds the minimum cut on the defined graph
        :param current_labeling: array corresponding to the current image labeling
        :param segmentation: array corresponding to the current image segmentation
        :param s: source node (represented as a 2D point to stick with the pixels locations notation)
        :param t: sink node
        :return: Updated labeling and segmentation arrays
        """
        cut_value, partition = nx.minimum_cut(self.graph, s, t, capacity="weight")
        S, T = partition
        for node in self.graph.nodes():
            if node in T and type(node[0]) != tuple and node not in [t, s]:
                current_labeling[node] = self.alpha
                segmentation[node] = COLORS[self.alpha]
        return current_labeling, segmentation

    def compute_binary_terms(self, labeling: np.ndarray, t_node: Node) -> None:
        """
        Compute the binary potentials
        :param labeling: current labeling
        :param t_node: sink node
        """
        for edge in self.graph.edges():
            if type(edge[1][0]) == tuple:
                edge = edge[1]  # cope with auxiliary nodes
            elif type(edge[0][0]) == tuple:
                edge = edge[0]
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
            if not different_label:
                # There is no auxiliary nodes between the two neighbors
                is_label_not_alpha = int(labeling[source] != self.alpha)
                self.graph[source][target]["weight"] = int(
                    binary_cost * is_label_not_alpha
                )
                self.graph[target][source]["weight"] = int(
                    binary_cost * is_label_not_alpha
                )
            else:
                # It means that the edge is actually an auxiliary node

                # Weight between aux node and s-node
                self.graph[edge][t_node]["weight"] = int(binary_cost * different_label)

                # Weight between nodes and aux node
                source_label_match = int(labeling[source] != self.alpha)
                self.graph[source][edge]["weight"] = int(
                    binary_cost * source_label_match
                )
                self.graph[edge][source]["weight"] = int(
                    binary_cost * source_label_match
                )

                target_label_match = int(labeling[target] != self.alpha)
                self.graph[target][edge]["weight"] = int(
                    binary_cost * target_label_match
                )
                self.graph[edge][target]["weight"] = int(
                    binary_cost * target_label_match
                )

    def update_auxiliary_nodes(
        self, current_labeling: np.ndarray, t_node: Node
    ) -> None:
        """
        Update the current graph by adding / removing auxiliary nodes and edges
        :param current_labeling: array representing the current labeling
        :param t_node: sink node
        """
        for auxiliary_node in self.auxiliary_nodes:
            source_node = auxiliary_node[0]
            target_node = auxiliary_node[1]
            self.graph.add_edge(source_node, target_node)

        self.graph.remove_nodes_from(self.auxiliary_nodes)
        self.auxiliary_nodes = []
        self.aux_to_sink_edged = []
        self.edges_to_add = []

        # Add auxiliary edges are the boundaries of labels partitions
        edges_to_remove = []
        for edge in self.graph.edges():
            source = edge[0]
            target = edge[1]
            if current_labeling[source] != current_labeling[target]:
                # These two neighbors separate the labeling partition
                self.auxiliary_nodes.append(edge)
                self.aux_to_sink_edged.append((edge, t_node))
                # the auxiliary node is defined as the tuple containing
                # the information of both nodes
                self.edges_to_add += [
                    (source, edge),
                    (target, edge),
                    (edge, source),
                    (edge, target),
                ]
                edges_to_remove.append((source, target))

        self.graph.add_nodes_from(self.auxiliary_nodes)
        self.graph.add_edges_from(self.edges_to_add)
        self.graph.add_edges_from(self.aux_to_sink_edged)
        self.graph.remove_edges_from(edges_to_remove)

    def set_edges_weights(
        self, t_node: Node, s_node: Node, current_labeling, unary_costs
    ):
        """
        Set the unary cost between nodes and source / sink
        :param t_node: sink node
        :param s_node: source node
        :param current_labeling: array of current image labeling
        :param unary_costs: array of binary potentials
        """
        inf_weight = float("inf")
        for v in self.graph.nodes():
            if v != t_node and v != s_node:
                if type(v[0]) != tuple:  # we already coped with aux nodes unary cost
                    self.graph.add_edges_from([(s_node, v), (v, t_node)])
                    unary_s = unary_costs[v][self.alpha]
                    if int(current_labeling[v]) == self.alpha:
                        unary_t = inf_weight
                    else:
                        unary_t = unary_costs[v][int(current_labeling[v])]
                    self.graph[v][t_node]["weight"] = unary_t
                    self.graph[s_node][v]["weight"] = unary_s
