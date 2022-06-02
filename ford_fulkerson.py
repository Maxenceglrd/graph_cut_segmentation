from typing import List, Tuple, Set

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class MinimumCutFordFulkerson:
    """
    Implementation of the minimym cut algorithm using
    the Ford Fulkerson algorithm
    """

    def __init__(self, adjacency_matrix: np.ndarray, with_sparse_matrix: bool = False):
        self.residual_graph = adjacency_matrix
        self.with_sparse_matrix = (
            with_sparse_matrix  # to handle networkx format of adjacency matrix
        )
        self.n_nodes = adjacency_matrix.shape[0]

    def find_augmenting_path(self, s: int, t: int, parent_list: List) -> bool:
        """
        Find an augmenting
        :param s: int, source node
        :param t: int, sink node
        :param parent_list: List of nodes parent
        :return: True if we found an augmenting path, else False
        """
        is_visited = [
            False if node_index != s else True for node_index in range(self.n_nodes)
        ]
        nodes_queue = [s]

        while len(nodes_queue) > 0:
            current_node = nodes_queue.pop(0)
            if self.with_sparse_matrix:
                iterator = self.residual_graph[current_node].nonzero()[1]
            else:
                iterator = range(len(self.residual_graph[current_node]))
            for node_index in iterator:
                if (
                    not is_visited[node_index]
                    and self.residual_graph[current_node, node_index] > 0
                ):
                    nodes_queue.append(node_index)
                    is_visited[node_index] = True
                    parent_list[node_index] = current_node
        return is_visited[t]

    def update_residual_graph(self, s, t, parents_list: List, path_flow) -> None:
        """
        Update the residual graph given the current path flow
        :param s: int, source node
        :param t: int, sink node
        :param parents_list:
        :param path_flow: float/int, flow of the selected math
        """
        current_node = t
        while current_node != s:
            current_parent = parents_list[current_node]
            self.residual_graph[current_parent, current_node] -= path_flow
            self.residual_graph[current_node, current_parent] += path_flow
            current_node = current_parent

    def find_minimum_cut(self, s: int, t: int) -> Tuple[Set[int], Set[int]]:
        """
        Main function to run the Ford Fulkerson algorithm
        to find the minimum cut
        :param s: int, source node
        :param t: int, sink node
        :return: Partition of nodes, tuple of two sets,
        the first one being the nodes connected to the source and
        the second the nodes connected to the sink
        """
        parents_list = [-1 for _ in range(self.n_nodes)]
        while self.find_augmenting_path(s, t, parents_list):
            s_t_flow = np.inf
            current_node = t
            while current_node != s:
                # we backtrack to get the residual flow
                residual_flow = self.residual_graph[
                    parents_list[current_node], current_node
                ]
                s_t_flow = min(s_t_flow, residual_flow)
                current_node = parents_list[current_node]
            # Update the residual graph
            self.update_residual_graph(s, t, parents_list, s_t_flow)

        # Find the s-t cut (nodes partition)
        connected_to_source = np.array([False for _ in range(self.n_nodes)])
        self.traverse_graph(s, connected_to_source)
        s_cut = np.where(connected_to_source)[0]
        t_cut = np.where(1 - connected_to_source)[0]
        return set(s_cut), set(t_cut)

    def traverse_graph(self, starting_node: int, is_visited: np.ndarray) -> None:
        """
        DFS done on the graph, implemented in a recursive way
        :param starting_node: int
        :param is_visited: Array of booleans, with True if the node is visited
        (this array is modified in place)
        """
        is_visited[starting_node] = True
        for k in range(self.residual_graph.shape[0]):
            if self.residual_graph[starting_node, k] > 0 and not is_visited[k]:
                self.traverse_graph(k, is_visited)


if __name__ == "__main__":
    # Below is given an example of a run of the manual
    # implementation of the Ford-Fulkerson algorithm
    # as a "proof" of the correctness of the algorithm
    # as by default, we use the networkx implementation
    # for the image segmentation for performance purposes.
    adjacency_matrix = np.array(
        [
            [0, 9, 8, 0, 0, 0],  # source node
            [0, 0, 0, 4, 4, 0],
            [0, 2, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0],  # sink node
        ]
    )

    # Run the manual implementation of Ford-Fulkerson algorithm
    min_cut_algo = MinimumCutFordFulkerson(adjacency_matrix)
    s_node = 0
    t_node = 5
    partial_manual = min_cut_algo.find_minimum_cut(s_node, t_node)

    # Use the networkx implementation
    g_net = nx.DiGraph()
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:
                g_net.add_edge(i, j, capacity=adjacency_matrix[i, j])
    cut_value, partition = nx.minimum_cut(g_net, s_node, t_node)

    # Plot the residual graph we got using the manual
    # implementation of Ford-Fulkerson algorithm
    g_net_residual = nx.DiGraph()
    for i in range(min_cut_algo.residual_graph.shape[0]):
        for j in range(min_cut_algo.residual_graph.shape[1]):
            if min_cut_algo.residual_graph[i, j] > 0:
                g_net_residual.add_edge(
                    i, j, capacity=min_cut_algo.residual_graph[i, j]
                )
    pos = nx.spring_layout(g_net_residual)
    nx.draw(g_net_residual, pos, with_labels=True)
    labels = nx.get_edge_attributes(g_net_residual, "capacity")
    nx.draw_networkx_edge_labels(g_net_residual, pos, edge_labels=labels)
    plt.show()
