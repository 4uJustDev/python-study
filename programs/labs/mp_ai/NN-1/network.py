class NeuralNetwork:
    def __init__(self):
        """Инициализация нейросети"""
        print("Нейросеть инициализирована")
        # Список для хранения данных всех обработанных изображений
        self.processed_images = []
        self.batch_data = []

    def process_image(
        self, is_editable_picture, vector_of_values, filename="Неизвестно"
    ):
        """
        Обрабатывает изображение

        Args:
            is_editable_picture (bool): Можно ли редактировать изображение
            vector_of_values (list): Вектор значений пикселей [0,1,0,0,0,1,...]
            filename (str): Имя файла изображения
        """

        print(f"Изображение: {filename}")
        print(f"Вектор значений: {vector_of_values}")

        # Выводим изображение в виде матрицы 3x3
        print("Матрица изображения (0 - чёрный, 1 - белый):")
        for i in range(0, 9, 3):
            row = vector_of_values[i : i + 3]
            print(f"  {row[0]} {row[1]} {row[2]}")

        # Статистика
        black_pixels = vector_of_values.count(0)
        white_pixels = vector_of_values.count(1)
        print(f"Чёрных пикселей: {black_pixels}, Белых пикселей: {white_pixels}")
        print("-" * 40)

        # Сохраняем данные для дальнейшей обработки
        image_data = {
            "filename": filename,
            "vector": vector_of_values,
            "is_editable": is_editable_picture,
            "black_pixels": black_pixels,
            "white_pixels": white_pixels,
            "matrix": [vector_of_values[i : i + 3] for i in range(0, 9, 3)],
        }

        self.processed_images.append(image_data)
        self.batch_data.append(image_data)

    def start_batch_processing(self):
        """Начинает обработку батча изображений"""
        print("\n" + "=" * 60)
        print("НАЧАЛО ОБРАБОТКИ БАТЧА ИЗОБРАЖЕНИЙ")
        print("=" * 60)
        self.batch_data = []  # Очищаем предыдущий батч

    def finish_batch_processing(self):
        """Завершает обработку батча и выводит общую статистику"""
        if not self.batch_data:
            print("Нет данных для обработки")
            return

        print("\n" + "=" * 60)
        print("ЗАВЕРШЕНИЕ ОБРАБОТКИ БАТЧА")
        print("=" * 60)

        print(f"Всего обработано изображений: {len(self.batch_data)}")
        print(f"Данные: {self.batch_data}")

        # Возвращаем данные для дальнейшей работы
        return self.batch_data

    def get_batch_data(self):
        """Возвращает данные текущего батча"""
        return self.batch_data

    def get_all_processed_data(self):
        """Возвращает данные всех обработанных изображений"""
        return self.processed_images

    def export_batch_to_file(self, filename="batch_data.txt"):
        """Экспортирует данные батча в файл"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("ДАННЫЕ БАТЧА ИЗОБРАЖЕНИЙ\n")
            f.write("=" * 50 + "\n\n")

            for i, img in enumerate(self.batch_data, 1):
                f.write(f"Изображение {i}: {img['filename']}\n")
                f.write(f"Вектор: {img['vector']}\n")
                f.write(f"Редактируемое: {'Да' if img['is_editable'] else 'Нет'}\n")
                f.write(f"Чёрных пикселей: {img['black_pixels']}\n")
                f.write("Матрица:\n")
                for row in img["matrix"]:
                    f.write(f"  {row[0]} {row[1]} {row[2]}\n")
                f.write("-" * 30 + "\n")

        print(f"Данные экспортированы в файл: {filename}")


# Тестовый код для проверки
if __name__ == "__main__":
    network = NeuralNetwork()
