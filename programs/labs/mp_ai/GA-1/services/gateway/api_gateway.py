from typing import List, Tuple, Dict, Any
from services.genetic.genetic_service import GeneticService


class APIGateway:
    def __init__(self):
        self.genetic_service = GeneticService()
        self.author_info = {
            "author": "Зарубин Александр Николаевич",
            "email": "zarubin_14@list.com",
            "course": "Интеллектуальные системы, 2025 (http://it-claim.ru/Education/Course/AI/ai_practice.htm)",
        }

    def get_author_info(self) -> Dict[str, str]:
        """Get author information"""
        return self.author_info

    def configure_network(self, size: int, adj_matrix: List[List[float]]) -> None:
        """Configure network topology"""
        self.genetic_service.set_network(size, adj_matrix)

    def configure_algorithm(self, params: Dict[str, Any]) -> None:
        """Configure algorithm parameters"""
        self.genetic_service.set_algorithm_params(
            pop_size=params.get("population_size", 50),
            generations=params.get("generations", 100),
            mutation_rate=params.get("mutation_rate", 0.1),
            crossover_rate=params.get("crossover_rate", 0.9),
        )

    def set_path_ends(self, start: int, end: int) -> None:
        """Set start and end nodes for path finding"""
        self.genetic_service.set_path_ends(start, end)

    def set_step_mode(self, mode: bool) -> None:
        """Set step mode for debugging"""
        self.genetic_service.set_step_mode(mode)

    def find_optimal_path(self, chromosome_length: int = 5) -> Tuple[List[int], float]:
        """Find optimal path using genetic algorithm"""
        return self.genetic_service.find_path(chromosome_length)
