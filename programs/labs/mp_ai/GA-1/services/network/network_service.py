import numpy as np
from typing import List, Tuple


class NetworkService:
    def __init__(self):
        self.network_size = 0
        self.adj_matrix = None

    def set_network(self, size: int, adj_matrix: List[List[float]]) -> None:
        """Initialize network topology"""
        self.network_size = size
        self.adj_matrix = np.array(adj_matrix)

        # Validate and correct adjacency matrix
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.adj_matrix[i][j] = 0
                elif self.adj_matrix[i][j] == 0:
                    self.adj_matrix[i][j] = 1e9  # Large number for no connection

    def validate_path(self, path: List[int]) -> Tuple[bool, float]:
        """Validate path and calculate its length"""
        total_length = 0

        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]

            if from_node == to_node or self.adj_matrix[from_node][to_node] >= 1e9:
                return False, float("inf")

            total_length += self.adj_matrix[from_node][to_node]

        return True, total_length

    def get_network_size(self) -> int:
        """Get network size"""
        return self.network_size

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix"""
        return self.adj_matrix.copy()
