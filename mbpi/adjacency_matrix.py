import numpy as np
from operator import itemgetter
from typing import Optional, Iterable


class AdjacencyMatrix:
    def __init__(self, matrix: np.ndarray, labels: Optional[np.ndarray] = None):
        self.N = len(matrix)
        self.matrix = matrix
        self.labels = labels

        if self.labels is not None:
            if len(matrix) != len(labels):
                raise ValueError("Mismatch between size of matrix and labels.")

            self.label_mapping = dict(zip(self.labels, range(self.N)))

    def __getitem__(self, index_iterable: Iterable):
        if not isinstance(index_iterable, Iterable):
            raise TypeError("Indexing requires an iterable.")

        if self.labels:
            get_indices = itemgetter(*index_iterable)
            indices = get_indices(self.label_mapping)

            indices = (indices,) if isinstance(indices, int) else indices
            indices = np.fromiter(indices, dtype=int)
        else:
            indices = index_iterable

        submatrix = self.matrix[np.ix_(indices, indices)]
        return AdjacencyMatrix(submatrix, index_iterable if self.labels else None)

    def __str__(self):
        matrix_str = np.array2string(self.matrix, separator=", ")
        if self.labels:
            labels_str = ", ".join(
                f"'{label}'" if isinstance(label, str) else str(label)
                for label in self.labels
            )
            labels_str = f"({labels_str})"
        else:
            labels_str = "None"
        return f"AdjacencyMatrix(\nmatrix=\n{matrix_str},\nlabels={labels_str})"

    def __repr__(self):
        return self.__str__()
