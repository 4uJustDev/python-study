import random
from typing import List, Tuple
from services.network.network_service import NetworkService


class PopulationService:
    def __init__(self, network_service: NetworkService):
        self.network_service = network_service
        self.population_size = 50
        self.tournament_size = 3
        self.mutation_rate = 0.1
        self.crossover_rate = 0.9

    def set_parameters(
        self,
        pop_size: int,
        tournament_size: int,
        mutation_rate: float,
        crossover_rate: float,
    ) -> None:
        """Set genetic algorithm parameters"""
        self.population_size = pop_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self, chromosome_length: int) -> List[List[int]]:
        """Initialize random population"""
        population = []
        network_size = self.network_service.get_network_size()

        for _ in range(self.population_size):
            chromosome = [
                random.randint(0, network_size - 1) for _ in range(chromosome_length)
            ]
            population.append(chromosome)
        return population

    def tournament_selection(
        self, population: List[List[int]], fitness: List[float]
    ) -> List[List[int]]:
        """Perform tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(
                list(zip(population, fitness)), self.tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, chromosome: List[int]) -> List[int]:
        """Perform mutation"""
        network_size = self.network_service.get_network_size()
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, network_size - 1)
        return chromosome

    def reduce_population(
        self,
        population: List[List[int]],
        fitness: List[float],
        new_population: List[List[int]],
        new_fitness: List[float],
    ) -> Tuple[List[List[int]], List[float]]:
        """Reduce population to maintain size"""
        combined = list(zip(population + new_population, fitness + new_fitness))
        combined.sort(key=lambda x: x[1], reverse=True)
        best = combined[: self.population_size]
        return [x[0] for x in best], [x[1] for x in best]
