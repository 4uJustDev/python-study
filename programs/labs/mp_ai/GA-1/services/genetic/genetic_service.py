from typing import List, Tuple
from services.network.network_service import NetworkService
from services.fitness.fitness_service import FitnessService
from services.population.population_service import PopulationService


class GeneticService:
    def __init__(self):
        self.network_service = NetworkService()
        self.fitness_service = FitnessService(self.network_service)
        self.population_service = PopulationService(self.network_service)
        self.generations = 100
        self.step_mode = False

    def set_network(self, size: int, adj_matrix: List[List[float]]) -> None:
        """Set network topology"""
        self.network_service.set_network(size, adj_matrix)

    def set_algorithm_params(
        self,
        pop_size: int,
        generations: int,
        mutation_rate: float,
        crossover_rate: float,
    ) -> None:
        """Set algorithm parameters"""
        self.generations = generations
        self.population_service.set_parameters(
            pop_size=pop_size,
            tournament_size=3,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

    def set_path_ends(self, start: int, end: int) -> None:
        """Set start and end nodes"""
        self.fitness_service.set_path_ends(start, end)

    def set_step_mode(self, mode: bool) -> None:
        """Set step mode for debugging"""
        self.step_mode = mode

    def find_path(self, chromosome_length: int = 5) -> Tuple[List[int], float]:
        """Find optimal path using genetic algorithm"""
        # Initialize population
        population = self.population_service.initialize_population(chromosome_length)
        fitness = self.fitness_service.calculate_population_fitness(population)

        for generation in range(self.generations):
            if self.step_mode:
                self._print_generation_info(generation, population, fitness)

            # Selection
            selected = self.population_service.tournament_selection(population, fitness)

            # Crossover
            new_population = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self.population_service.crossover(
                    selected[i], selected[i + 1]
                )
                new_population.extend([child1, child2])

            if len(new_population) < len(population):
                new_population.append(selected[-1])

            # Mutation
            new_population = [
                self.population_service.mutate(chromo) for chromo in new_population
            ]
            new_fitness = self.fitness_service.calculate_population_fitness(
                new_population
            )

            # Population reduction
            population, fitness = self.population_service.reduce_population(
                population, fitness, new_population, new_fitness
            )

            if not self.step_mode:
                best_fit = max(fitness)
                avg_fit = sum(fitness) / len(fitness)
                print(
                    f"Generation {generation}: Best fitness = {best_fit:.4f}, Average fitness = {avg_fit:.4f}"
                )

        # Return best path found
        best_idx = fitness.index(max(fitness))
        best_path = (
            [self.fitness_service.start_node]
            + population[best_idx]
            + [self.fitness_service.end_node]
        )
        best_length = 1 / fitness[best_idx] if fitness[best_idx] > 0 else float("inf")

        # Clean path (remove consecutive duplicates)
        cleaned_path = [best_path[0]]
        for node in best_path[1:]:
            if node != cleaned_path[-1]:
                cleaned_path.append(node)

        return cleaned_path, best_length

    def _print_generation_info(
        self, generation: int, population: List[List[int]], fitness: List[float]
    ) -> None:
        """Print detailed generation information in step mode"""
        print(f"\nGeneration {generation}:")
        print("Population:")
        for i, (chromo, fit) in enumerate(zip(population, fitness)):
            print(f"{i}: {chromo} (fitness: {fit:.4f})")
        input("\nPress Enter to continue...")
