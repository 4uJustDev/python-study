from typing import List
from services.network.network_service import NetworkService


class FitnessService:
    def __init__(self, network_service: NetworkService):
        self.network_service = network_service
        self.start_node = 0
        self.end_node = 0

    def set_path_ends(self, start: int, end: int) -> None:
        """Set start and end nodes for path calculation"""
        self.start_node = start
        self.end_node = end

    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Calculate fitness score for a chromosome (inverse of path length)"""
        path = [self.start_node] + chromosome + [self.end_node]
        is_valid, length = self.network_service.validate_path(path)

        if not is_valid:
            return 0.0

        return 1.0 / length if length > 0 else 0.0

    def calculate_population_fitness(self, population: List[List[int]]) -> List[float]:
        """Calculate fitness scores for entire population"""
        return [self.calculate_fitness(chromo) for chromo in population]
