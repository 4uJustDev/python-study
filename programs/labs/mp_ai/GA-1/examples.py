from services.gateway.api_gateway import APIGateway


def example_small_network():
    """Example 1: Small network (5 nodes) with simple topology"""
    gateway = APIGateway()

    # Small network with 5 nodes
    matrix = [
        [0, 10, 0, 30, 100],
        [10, 0, 50, 0, 0],
        [0, 50, 0, 20, 10],
        [30, 0, 20, 0, 60],
        [100, 0, 10, 60, 0],
    ]

    gateway.configure_network(5, matrix)
    gateway.configure_algorithm(
        {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
        }
    )
    gateway.set_path_ends(0, 4)

    print("\nExample 1: Small Network")
    print("Network size: 5 nodes")
    print("Looking for path from node 0 to node 4")

    best_path, length = gateway.find_optimal_path(chromosome_length=3)
    print("Found path:", best_path)
    print("Path length:", length)


def example_grid_network():
    """Example 2: Grid network (9 nodes) with grid-like topology"""
    gateway = APIGateway()

    # Grid network with 9 nodes (3x3 grid)
    matrix = [
        [0, 1, 0, 1, 0, 0, 0, 0, 0],  # 0
        [1, 0, 1, 0, 1, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 1, 0, 0, 0],  # 2
        [1, 0, 0, 0, 1, 0, 1, 0, 0],  # 3
        [0, 1, 0, 1, 0, 1, 0, 1, 0],  # 4
        [0, 0, 1, 0, 1, 0, 0, 0, 1],  # 5
        [0, 0, 0, 1, 0, 0, 0, 1, 0],  # 6
        [0, 0, 0, 0, 1, 0, 1, 0, 1],  # 7
        [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 8
    ]

    gateway.configure_network(9, matrix)
    gateway.configure_algorithm(
        {
            "population_size": 30,
            "generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.9,
        }
    )
    gateway.set_path_ends(0, 8)  # From top-left to bottom-right

    print("\nExample 2: Grid Network")
    print("Network size: 9 nodes (3x3 grid)")
    print("Looking for path from node 0 to node 8")

    best_path, length = gateway.find_optimal_path(chromosome_length=5)
    print("Found path:", best_path)
    print("Path length:", length)


def example_complex_network():
    """Example 3: Complex network (10 nodes) with multiple possible paths"""
    gateway = APIGateway()

    # Complex network with 10 nodes
    matrix = [
        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [5, 0, 3, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 3, 0, 4, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 4, 0, 2, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 2, 0, 6, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 6, 0, 3, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 3, 0, 4, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 4, 0, 5, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 2],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],  # 9
    ]

    gateway.configure_network(10, matrix)
    gateway.configure_algorithm(
        {
            "population_size": 50,
            "generations": 150,
            "mutation_rate": 0.15,
            "crossover_rate": 0.85,
        }
    )
    gateway.set_path_ends(0, 9)

    print("\nExample 3: Complex Network")
    print("Network size: 10 nodes")
    print("Looking for path from node 0 to node 9")

    best_path, length = gateway.find_optimal_path(chromosome_length=7)
    print("Found path:", best_path)
    print("Path length:", length)


def example_sparse_network():
    """Example 4: Sparse network (8 nodes) with few connections"""
    gateway = APIGateway()

    # Sparse network with 8 nodes
    matrix = [
        [0, 10, 0, 0, 0, 0, 0, 0],  # 0
        [10, 0, 15, 0, 0, 0, 0, 0],  # 1
        [0, 15, 0, 20, 0, 0, 0, 0],  # 2
        [0, 0, 20, 0, 25, 0, 0, 0],  # 3
        [0, 0, 0, 25, 0, 30, 0, 0],  # 4
        [0, 0, 0, 0, 30, 0, 35, 0],  # 5
        [0, 0, 0, 0, 0, 35, 0, 40],  # 6
        [0, 0, 0, 0, 0, 0, 40, 0],  # 7
    ]

    gateway.configure_network(8, matrix)
    gateway.configure_algorithm(
        {
            "population_size": 25,
            "generations": 75,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
        }
    )
    gateway.set_path_ends(0, 7)

    print("\nExample 4: Sparse Network")
    print("Network size: 8 nodes")
    print("Looking for path from node 0 to node 7")

    best_path, length = gateway.find_optimal_path(chromosome_length=4)
    print("Found path:", best_path)
    print("Path length:", length)


def example_dense_network():
    """Example 5: Dense network (6 nodes) with many connections"""
    gateway = APIGateway()

    # Dense network with 6 nodes
    matrix = [
        [0, 5, 8, 12, 0, 0],  # 0
        [5, 0, 6, 0, 9, 0],  # 1
        [8, 6, 0, 7, 0, 10],  # 2
        [12, 0, 7, 0, 4, 0],  # 3
        [0, 9, 0, 4, 0, 11],  # 4
        [0, 0, 10, 0, 11, 0],  # 5
    ]

    gateway.configure_network(6, matrix)
    gateway.configure_algorithm(
        {
            "population_size": 40,
            "generations": 100,
            "mutation_rate": 0.25,
            "crossover_rate": 0.75,
        }
    )
    gateway.set_path_ends(0, 5)

    print("\nExample 5: Dense Network")
    print("Network size: 6 nodes")
    print("Looking for path from node 0 to node 5")

    best_path, length = gateway.find_optimal_path(chromosome_length=3)
    print("Found path:", best_path)
    print("Path length:", length)
