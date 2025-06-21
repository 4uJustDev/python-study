class NeuralNetwork:
    def __init__(self):
        """Инициализация нейросети"""
        print("Нейросеть инициализирована")

    def process_image(self, is_editable_picture, vector_of_values):
        if is_editable_picture:
            print("ПРЕДУПРЕЖДЕНИЕ: фото редактируется")

        print(f"Вектор значений: {vector_of_values}")


# Тестовый код для проверки
if __name__ == "__main__":
    network = NeuralNetwork()
