import numpy as np


class NeuralNetwork:
    def __init__(self):
        """Инициализация нейросети"""
        print("Нейросеть инициализирована")
        # Список для хранения данных всех обработанных изображений
        self.processed_images = []
        self.batch_data = []
        self.amount_neurons = 1  # Количество нейронов по умолчанию

        self.weights = 0.10 * np.random.randn(9, self.amount_neurons)
        self.biases = np.zeros((1, self.amount_neurons))

    """ОБРАБОТКА"""

    def process_image(
        self,
        is_editable_picture,
        vector_of_values,
        filename="Неизвестно",
        is_correct=False,
    ):
        """
        Обрабатывает изображение

        Args:
            is_editable_picture (bool): Можно ли редактировать изображение
            vector_of_values (list): Вектор значений пикселей [0,1,0,0,0,1,...]
            filename (str): Имя файла изображения
            is_correct (bool): Правильное ли изображение
        """
        print(f"Вектор значений: {vector_of_values}")

        # Статистика
        black_pixels = vector_of_values.count(0)
        white_pixels = vector_of_values.count(1)
        print("-" * 40)

        # Сохраняем данные для дальнейшей обработки
        image_data = {
            "filename": filename,
            "vector": vector_of_values,
            "is_editable": is_editable_picture,
            "is_correct": is_correct,
            "black_pixels": black_pixels,
            "white_pixels": white_pixels,
            "matrix": [vector_of_values[i : i + 3] for i in range(0, 9, 3)],
        }

        self.processed_images.append(image_data)
        self.batch_data.append(image_data)

    def start_batch_processing(self, neurons_count=1):
        """Начинает обработку батча изображений"""
        print("\n" + "=" * 60)
        print("НАЧАЛО ОБРАБОТКИ БАТЧА ИЗОБРАЖЕНИЙ")
        print("=" * 60)
        print(f"Количество нейронов: {neurons_count}")
        self.batch_data = []  # Очищаем предыдущий батч
        self.amount_neurons = neurons_count  # Сохраняем количество нейронов

        # Обновляем веса и смещения для нового количества нейронов
        self._update_network_parameters()

    def _update_network_parameters(self):
        """Обновляет веса и смещения в соответствии с текущим количеством нейронов"""
        self.weights = 0.10 * np.random.randn(9, self.amount_neurons)
        self.biases = np.zeros((1, self.amount_neurons))

    def finish_batch_processing(self, learning_rate=0.1, error_threshold=0.2):
        """Завершает обработку батча и выводит общую статистику"""
        if not self.batch_data:
            print("Нет данных для обработки")
            return

        print("\n" + "=" * 60)
        print("ЗАВЕРШЕНИЕ ОБРАБОТКИ БАТЧА")
        print("=" * 60)

        self.train(learning_rate, error_threshold)

        # Возвращаем данные для дальнейшей работы
        return self.batch_data

    def get_batch_data(self):
        """Возвращает данные текущего батча в виде векторов признаков и меток классов

        Returns:
            tuple: (X, y) где:
                X - список векторов признаков (пиксельные векторы)
                y - список меток классов (1 для корректных, 0 для некорректных)
        """
        X = []

        y = []

        for item in self.batch_data:
            # Добавляем вектор признаков
            X.append(item["vector"])

            # Добавляем метку класса (1 для корректных, 0 для некорректных)
            y.append(1 if item["is_correct"] else 0)

        return X, y

    def get_all_processed_data(self):
        """Возвращает данные всех обработанных изображений"""
        return self.processed_images

    def export_batch_to_file(self, filename="batch_data.txt"):
        """Экспортирует данные батча в файл"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("ДАННЫЕ БАТЧА ИЗОБРАЖЕНИЙ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Количество нейронов: {getattr(self, 'amount_neurons', 1)}\n")
            f.write(f"Всего изображений: {len(self.batch_data)}\n\n")

            for i, img in enumerate(self.batch_data, 1):
                f.write(f"Изображение {i}: {img['filename']}\n")
                f.write(f"Вектор: {img['vector']}\n")
                f.write(f"Редактируемое: {'Да' if img['is_editable'] else 'Нет'}\n")
                f.write(f"Правильное: {'Да' if img['is_correct'] else 'Нет'}\n")
                f.write(f"Чёрных пикселей: {img['black_pixels']}\n")
                f.write("Матрица:\n")
                for row in img["matrix"]:
                    f.write(f"  {row[0]} {row[1]} {row[2]}\n")
                f.write("-" * 30 + "\n")

    """ВЫЧИСЛЕНИЯ"""

    def activare_RELU(self, inputs):
        return np.maximum(0, inputs)

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases

        return self.activare_RELU(z)

    def train(self, learning_rate=0.1, error_threshold=0.2, max_epochs=100):
        """Обучение сети"""
        X, y = self.get_batch_data()

        # Преобразуем в numpy массивы для корректной работы
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Преобразуем в столбец

        epoch = 0
        total_error = float("inf")

        print("Начальные параметры:")
        print(f"X,: {X}")
        print(f"y,: {y}")
        print(f"Веса: {self.weights}")
        print(f"Смещение: {self.biases}")
        print(f"Скорость обучения: {learning_rate}")
        print(f"Порог ошибки: {error_threshold}")
        print(f"Размер данных: X={X.shape}, y={y.shape}")
        print("\nНачало обучения...\n")

        while total_error > error_threshold and epoch < max_epochs:
            total_error = 0
            epoch += 1

            for i in range(len(X)):
                # Прямой проход
                x_i = X[i].reshape(1, -1)  # Преобразуем в строку
                output = self.forward(x_i)

                # Вычисление ошибки
                error = (y[i] - output) ** 2
                total_error += error[0, 0]  # Извлекаем скалярное значение

                # Производная функции потерь по выходу
                d_loss_d_output = -2 * (y[i] - output)

                # Производная ReLU
                if output[0, 0] > 0:  # ReLU активна
                    d_output_d_z = 1
                else:  # ReLU неактивна
                    d_output_d_z = 0

                # Градиент по весам и смещениям
                delta = d_loss_d_output * d_output_d_z

                # Обновление весов (правильное матричное умножение)
                weight_gradient = x_i.T * delta
                self.weights -= learning_rate * weight_gradient

                # Обновление смещений
                self.biases -= learning_rate * delta

            # Средняя ошибка на эпохе
            total_error /= len(X)

            if epoch % 10 == 0 or total_error <= error_threshold:
                print(f"Эпоха {epoch}: Средняя ошибка = {total_error:.4f}")
                print(f"Текущие веса: {self.weights.flatten()}")
                print(f"Текущее смещение: {self.biases.flatten()}\n")

                if total_error <= error_threshold:
                    print(
                        f"Обучение завершено на эпохе {epoch} с ошибкой {total_error:.4f}"
                    )
                    break

        print("Конечные параметры:")
        print(f"X,: {self.weights}")
        print(f"y,: {self.biases}")
        return self.weights, self.biases

    def predict(self, input_vector):
        """Предсказание для одного изображения"""
        # Преобразуем входной вектор в правильный формат
        if isinstance(input_vector, list):
            input_vector = np.array(input_vector)

        # Убеждаемся, что это 2D массив (строка)
        if input_vector.ndim == 1:
            input_vector = input_vector.reshape(1, -1)

        output = self.forward(input_vector)
        return 1 if output[0, 0] > 0.5 else 0


# Тестовый код для проверки
if __name__ == "__main__":
    network = NeuralNetwork()
