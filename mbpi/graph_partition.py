import random
import numpy as np
from typing import TypeVar, Generic, Dict, Tuple, List
import xxhash

T = TypeVar("T")

class GraphPartition(Generic[T]):
    def __init__(self, labels: np.ndarray, group_lengths: List[int]):
        self.N = len(labels)
        self.group_num = len(group_lengths)
        self.data = labels.copy()
        self.positions: Dict[str, int] = {label: i for i, label in enumerate(labels)}

        self.group_lengths = group_lengths
        if len(self.group_lengths) < 2:
            raise ValueError("At least two groups are required.")
        if sum(self.group_lengths) != self.N:
            raise ValueError("Sum of group lengths must equal the total number of labels.")

        self.group_ends = np.cumsum([0] + self.group_lengths)
        self.group_slices = tuple(zip(self.group_ends[:-1], self.group_ends[1:]))

        self.position_to_group = np.zeros(self.N, dtype=np.uint32)
        for group_index, (start, end) in enumerate(self.group_slices):
            self.position_to_group[start:end] = group_index

        self.label_to_hash = {label: self._hash_label(label) for label in labels}

        self.group_hashes = np.array(
            [self._hash_group(self.data[start:end]) for start, end in self.group_slices]
        )

    def swap_by_indices(self, i: int, j: int) -> None:
        u, v = self.data[i], self.data[j]
        self.data[i], self.data[j] = self.data[j], self.data[i]
        self.positions[u], self.positions[v] = j, i

        # begin hash update
        group_i, group_j = self.position_to_group[i], self.position_to_group[j]

        start_i, end_i = self.group_slices[group_i]
        start_j, end_j = self.group_slices[group_j]
        self.group_hashes[group_i] = self._hash_group(self.data[start_i:end_i])
        self.group_hashes[group_j] = self._hash_group(self.data[start_j:end_j])

    def swap_by_elements(self, u: T, v: T) -> None:
        i, j = self.positions[u], self.positions[v]
        self.swap_by_indices(i, j)

    def get_group_representations(self) -> np.ndarray:
        return np.split(self.data, self.group_ends[1:-1])

    def get_group_index(self, label: T) -> int:
        position = self.positions[label]
        return self.position_to_group[position]

    def random_label_pair(self) -> Tuple[T, T]:
        first_group, second_group = random.sample(range(self.group_num), 2)
        first_start, first_end = self.group_slices[first_group]
        second_start, second_end = self.group_slices[second_group]

        first_label = self.data[random.randint(first_start, first_end - 1)]
        second_label = self.data[random.randint(second_start, second_end - 1)]

        return first_label, second_label

    def __getitem__(self, group_index: int) -> np.ndarray:
        start, end = self.group_slices[group_index]
        return self.data[start:end]

    def __len__(self) -> int:
        return self.N

    def __str__(self) -> str:
        group_representations = self.get_group_representations()
        return "\n".join(
            f"Group {i}: {group}" for i, group in enumerate(group_representations)
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __copy__(self):
        return GraphPartition(self.data.copy(), self.group_lengths.copy())

    def copy(self):
        return self.__copy__()

    def _hash_label(self, label: str) -> int:
        return xxhash.xxh32(label.encode("utf-8")).intdigest()

    def _hash_group(self, group: np.ndarray) -> int:
        hashed_elems = np.array([self.label_to_hash[label] for label in group])
        return np.sum(hashed_elems) + np.sum(np.square(hashed_elems))

    def __hash__(self) -> int:
        large_prime = 2**31 - 1  # Mersenne prime (2^31 - 1)
        hash_value = (
        np.sum(self.group_hashes, dtype=object) +
            np.sum(np.square(self.group_hashes, dtype=object), dtype=object)
        )
        return (hash_value % large_prime)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphPartition):
            return False
        return self.__hash__() == other.__hash__()
