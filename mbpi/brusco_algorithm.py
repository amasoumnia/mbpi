import random
import itertools
import numpy as np
from typing import List, NamedTuple
from mbpi.scorers import Scorers
from mbpi.graph_partition import GraphPartition
from mbpi.adjacency_matrix import AdjacencyMatrix

class Solution(NamedTuple):
    partition: GraphPartition
    diversity_score: float
    dispersion_score: float

def brusco_bicriterion(
    d_matrix: AdjacencyMatrix,
    group_sizes: List[int],
    diversity_weight_options: List[float],
    restarts: int,
) -> List[List[int]]:
    """
    Performs the diversity and dispersion bicriterion method proposed in
    Brusco 2020, specifically the multistart pairwise interchange heuristic.

    Args:
    d_matrix (AdjacencyMatrix): Adjacency matrix representing the dissimilarity between elements.
    group_sizes (List[int]): The number of elements in each group.
    diversity_weight_options (List[float]): List of possible weights for diversity criterion.
    restarts (int): Number of restarts for the algorithm.

    Returns:
    List[List[int]]: The optimal partition, where the i-th list contains the indices of the items in group i.
    """

    assert isinstance(
        d_matrix, AdjacencyMatrix
    ), "The input must be an AdjacencyMatrix object."
    # Enforce 0 diagonals and symmetry in the dissimilarity matrix
    assert np.allclose(
        np.diag(d_matrix.matrix), 0
    ), "The diagonal of the dissimilarity matrix must be 0."
    assert np.allclose(
        d_matrix.matrix, d_matrix.matrix.T
    ), "The dissimilarity matrix must be symmetric."
    assert (
        sum(group_sizes) == d_matrix.N
    ), "The group sizes must cover all provided vertices."
    assert all(
        x > 0 and x < 1 for x in diversity_weight_options
    ), "The options for diversity weights must be within (0, 1)."

    pareto_batch: List[Solution] = []  # Store partitions with their scores
    visited_score_pairs = set()
    visited_partitions_hashes = set()

    # begin optimzation
    for r in range(restarts):
        diversity_weight = random.choice(diversity_weight_options)
        dispersion_weight = 1 - diversity_weight

        shuffled_vertices = np.random.permutation(d_matrix.labels)
        candidate_partition = GraphPartition(shuffled_vertices, group_sizes)

        z_curr = Scorers.weighted_dissimilarity(
            d_matrix, candidate_partition, diversity_weight, dispersion_weight
        )

        can_improve = True

        while can_improve:
            can_improve = False

            for label_i, label_j in itertools.combinations(shuffled_vertices, 2):
                if candidate_partition.__hash__() in visited_partitions_hashes:
                    candidate_partition.swap_by_elements(label_i, label_j)
                    continue

                if candidate_partition.get_group_index(
                    label_i
                ) != candidate_partition.get_group_index(label_j):
                    candidate_partition.swap_by_elements(label_i, label_j)

                    div_measure = Scorers.diversity_measure(
                        d_matrix, candidate_partition
                    )
                    disp_measure = Scorers.dispersion_measure(
                        d_matrix, candidate_partition
                    )
                    global_measure = (
                        diversity_weight * div_measure
                        + dispersion_weight * disp_measure
                    )

                    add_to_pareto = True
                    dominated_by_candidate = []

                    for i, (pareto_partition, pareto_div, pareto_disp) in enumerate(
                        pareto_batch
                    ):
                        if (pareto_div >= div_measure and pareto_disp >= disp_measure) and \
                            (pareto_div > div_measure or pareto_disp > disp_measure):
                            add_to_pareto = False
                            break
                        elif (div_measure >= pareto_div and disp_measure >= pareto_disp) and \
                            (div_measure > pareto_div or disp_measure > pareto_disp):
                            dominated_by_candidate.append(i)

                    if add_to_pareto:
                        pareto_batch.append(
                            Solution(candidate_partition.copy(), div_measure, disp_measure)
                        )

                        for index in sorted(dominated_by_candidate, reverse=True):
                            del pareto_batch[index]

                    # score improvement check
                    if global_measure > z_curr:
                        z_curr = global_measure
                        can_improve = True
                    else:
                        candidate_partition.swap_by_elements(label_i, label_j)

                    score_pair = (div_measure, disp_measure)
                    visited_score_pairs.add(score_pair)
                    visited_partitions_hashes.add(candidate_partition.__hash__())

    return [partition for partition, _, _ in pareto_batch]