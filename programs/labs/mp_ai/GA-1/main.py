from services.gateway.api_gateway import APIGateway
import examples


def main():
    # Create API gateway
    gateway = APIGateway()

    # Print author information
    author_info = gateway.get_author_info()
    print(f"Author: {author_info['author']}")
    print(f"Email: {author_info['email']}")
    print(f"Course: {author_info['course']}")

    examples.example_small_network()

    # examples.example_small_network()
    # examples.example_grid_network()
    # examples.example_complex_network()
    # examples.example_sparse_network()
    # examples.example_dense_network()


if __name__ == "__main__":
    main()
