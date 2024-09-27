import numpy as np
from mbpi.graph_partition import GraphPartition
from mbpi.adjacency_matrix import AdjacencyMatrix


class Scorers:
    @staticmethod
    def diversity_measure(
        d_matrix: AdjacencyMatrix, partition: GraphPartition
    ) -> float:
        group_dissimilarities = {}
        for group_num in range(partition.group_num):
            vertices = partition[group_num]
            sub_mat = d_matrix[vertices].matrix
            if sub_mat.shape[0] > 1:
                group_dissimilarities[group_num] = np.sum(np.triu(sub_mat, k=1))
            else:
                group_dissimilarities[group_num] = 0

        return sum(group_dissimilarities.values())

    @staticmethod
    def dispersion_measure(
        d_matrix: AdjacencyMatrix, partition: GraphPartition
    ) -> float:
        min_dissimilarity = float("inf")
        for group_num in range(partition.group_num):
            vertices = partition[group_num]
            sub_mat = d_matrix[vertices].matrix
            np.fill_diagonal(sub_mat, np.inf)
            group_min = np.min(sub_mat)

            if group_min < min_dissimilarity:
                min_dissimilarity = group_min

        return min_dissimilarity

    @staticmethod
    def weighted_dissimilarity(
        d_matrix: AdjacencyMatrix,
        partition: GraphPartition,
        diversity_weight: float,
        dispersion_weight: float,
    ) -> float:
        return diversity_weight * Scorers.diversity_measure(
            d_matrix, partition
        ) + dispersion_weight * Scorers.dispersion_measure(d_matrix, partition)
