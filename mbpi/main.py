import os
import json
import numpy as np
from mbpi.adjacency_matrix import AdjacencyMatrix
from mbpi.brusco_algorithm import brusco_bicriterion

def run_brusco_test(folder_path, fname=None):
    json_files = [fname] if fname else [f for f in os.listdir(folder_path) if f.endswith(".json")]
    
    if not json_files:
        print("No JSON files found in the directory.")
        return

    for file in json_files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, "r") as f:
            data = json.load(f)
            matrix, labels, group_sizes, diversity_weight_options, restarts = map(
                data.get,
                [
                    "matrix",
                    "labels",
                    "group_sizes",
                    "diversity_weight_options",
                    "restarts",
                ],
            )
            matrix = np.array(matrix)

            adj_matrix = AdjacencyMatrix(matrix, labels)

            result = brusco_bicriterion(
                adj_matrix, group_sizes, diversity_weight_options, restarts
            )

            # Report the result
            print(f"Results for {file}:")
            for i, partition in enumerate(result):
                print(f"Partition {i}:")
                for group_index, group in enumerate(partition.groups):
                    print(f"  Group {group_index}: {[labels[idx] for idx in group]}")
                print()

if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_file_dir, "tests")
    run_brusco_test(folder_path, 'example_case.json')
