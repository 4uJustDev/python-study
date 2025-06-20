import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from network import NeuralNetwork


class BW_BMP_Viewer:
    def __init__(self, root, images_folder):
        self.root = root
        self.images_folder = images_folder
        self.root.title("Чёрно-белый BMP 3x3 Viewer")

        # Пиксели: 0 (чёрный) или 255 (белый)
        self.pixels = [[255] * 3 for _ in range(3)]

        # Инициализируем нейросеть
        self.network = NeuralNetwork()

        # Список выбранных файлов
        self.selected_files = []

        if not os.path.exists(self.images_folder):
            messagebox.showerror("Ошибка", f"Папка '{self.images_folder}' не найдена!")
            return

        self.create_widgets()

    def create_widgets(self):
        # Главное окно
        paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Левая панель
        left_panel = tk.Frame(paned_window, width=200)
        paned_window.add(left_panel)

        # Кнопки
        tk.Button(left_panel, text="Новый", command=self.new_image).pack(
            fill=tk.X, padx=2, pady=2
        )
        tk.Button(left_panel, text="Сохранить", command=self.save_image).pack(
            fill=tk.X, padx=2, pady=2
        )
        tk.Button(left_panel, text="Анализ", command=self.analyze_image).pack(
            fill=tk.X, padx=2, pady=2
        )
        tk.Button(
            left_panel, text="Обучить нейросеть", command=self.analyze_selected_files
        ).pack(fill=tk.X, padx=2, pady=2)

        # Кнопки управления выбором
        selection_frame = tk.Frame(left_panel)
        selection_frame.pack(fill=tk.X, padx=2, pady=2)
        tk.Button(
            selection_frame, text="Выбрать все", command=self.select_all_files
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            selection_frame, text="Снять выбор", command=self.deselect_all_files
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Дерево файлов с чекбоксами
        self.tree = ttk.Treeview(
            left_panel, columns=("selected", "correct"), show="tree headings"
        )
        self.tree.heading("#0", text="Файлы")
        self.tree.heading("selected", text="✓")
        self.tree.heading("correct", text="✅")
        self.tree.column("#0", width=150)
        self.tree.column("selected", width=30, anchor="center")
        self.tree.column("correct", width=30, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Кнопка для добавления/исключения выбранного файла
        self.toggle_button = tk.Button(
            left_panel, text="Добавить/Исключить", command=self.toggle_selected_file
        )
        self.toggle_button.pack(fill=tk.X, padx=2, pady=2)

        # Кнопка для разметки правильности (заблокирована по умолчанию)
        self.correct_button = tk.Button(
            left_panel,
            text="Правильно/Неправильно",
            command=self.toggle_correctness,
            state=tk.DISABLED,
        )
        self.correct_button.pack(fill=tk.X, padx=2, pady=2)

        # Поле для количества нейронов
        neurons_frame = tk.Frame(left_panel)
        neurons_frame.pack(fill=tk.X, padx=2, pady=2)
        tk.Label(neurons_frame, text="Количество нейронов:").pack(side=tk.LEFT)
        self.neurons_var = tk.StringVar(value="1")
        self.neurons_entry = tk.Entry(
            neurons_frame, textvariable=self.neurons_var, width=10
        )
        self.neurons_entry.pack(side=tk.RIGHT)

        # Скорость обучения
        error_threshold_frame = tk.Frame(left_panel)
        error_threshold_frame.pack(fill=tk.X, padx=2, pady=2)
        tk.Label(error_threshold_frame, text="Порог ошибки:").pack(side=tk.LEFT)
        self.error_threshold_var = tk.StringVar(value="0.1")
        self.error_threshold_entry = tk.Entry(
            error_threshold_frame, textvariable=self.error_threshold_var, width=10
        )
        self.error_threshold_entry.pack(side=tk.RIGHT)

        # Поле для порога ошибки
        learning_rate_frame = tk.Frame(left_panel)
        learning_rate_frame.pack(fill=tk.X, padx=2, pady=2)
        tk.Label(learning_rate_frame, text="Скорость обучения:").pack(side=tk.LEFT)
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.learning_rate_entry = tk.Entry(
            learning_rate_frame, textvariable=self.learning_rate_var, width=10
        )
        self.learning_rate_entry.pack(side=tk.RIGHT)

        # Холст для рисования
        self.canvas = tk.Canvas(paned_window, width=300, height=300, bg="white")
        paned_window.add(self.canvas)
        self.canvas.bind("<Button-1>", self.toggle_pixel)

        # Консоль
        self.console = tk.Text(
            self.root, height=8, state=tk.DISABLED, bg="black", fg="white"
        )
        self.console.pack(fill=tk.BOTH, expand=False)

        self.draw_grid()
        self.draw_pixels()
        self.populate_tree()

    def new_image(self):
        self.pixels = [[255] * 3 for _ in range(3)]
        self.draw_pixels()
        self.log("Создано новое изображение 3x3")

    def save_image(self):
        filepath = filedialog.asksaveasfilename(
            initialdir=self.images_folder,
            title="Сохранить как BMP",
            filetypes=(("BMP files", "*.bmp"), ("All files", "*.*")),
            defaultextension=".bmp",
        )

        if filepath:
            img = Image.new("L", (3, 3))
            for y in range(3):
                for x in range(3):
                    img.putpixel((x, y), self.pixels[y][x])
            img.save(filepath)
            self.log(f"Сохранено: {os.path.basename(filepath)}")
            self.populate_tree()

    def analyze_image(self):
        """Анализирует текущее изображение через нейросеть"""
        # Преобразуем пиксели в вектор (0 или 1)
        vector = []
        for row in self.pixels:
            for pixel in row:
                vector.append(
                    1 if pixel == 0 else 0
                )  # 1 для чёрного (правильного), 0 для белого (неправильного)

        # Проверяем, можно ли редактировать изображение
        is_editable = self.check_if_editable()

        # Передаём данные в нейросеть
        self.network.process_image(is_editable, vector, "Текущее изображение")

        # Вызов предсказания
        prediction = self.network.predict(vector)
        self.log(f"Вектор изображения: {vector}")
        self.log(
            f"Результат предсказания: {'Правильное' if prediction == 1 else 'Неправильное'}"
        )

    def analyze_selected_files(self):
        """Анализирует все выбранные файлы"""
        if not self.selected_files:
            self.log("Нет выбранных файлов для анализа")
            return

        # Получаем количество нейронов
        try:
            neurons_count = int(self.neurons_var.get())
            if neurons_count <= 0:
                self.log("Ошибка: количество нейронов должно быть положительным числом")
                return
        except ValueError:
            self.log("Ошибка: введите корректное число нейронов")
            return

        # Получаем порог ошибки
        try:
            error_threshold = float(self.error_threshold_var.get())
            if error_threshold <= 0:
                self.log("Ошибка: порог ошибки должен быть положительным числом")
                return
        except ValueError:
            self.log("Ошибка: введите корректное значение порога ошибки")
            return

        # Получаем Скорость
        try:
            learning_rate = float(self.learning_rate_var.get())
            if learning_rate <= 0:
                self.log("Ошибка: Скорость должен быть положительным числом")
                return
        except ValueError:
            self.log("Ошибка: введите корректное значение скорости")
            return

        self.log(
            f"Анализ {len(self.selected_files)} выбранных файлов с {neurons_count} нейронами и порогом ошибки {error_threshold}..."
        )

        # Начинаем batch processing
        self.network.start_batch_processing(neurons_count)

        processed_count = 0
        for filepath in self.selected_files:
            try:
                # Проверяем, что файл существует и это действительно файл
                if not os.path.isfile(filepath):
                    self.log(f"Пропуск {filepath}: не является файлом")
                    continue

                # Загружаем изображение
                img = Image.open(filepath)
                width, height = img.size

                if (width, height) != (3, 3):
                    self.log(
                        f"Пропуск {os.path.basename(filepath)}: неверный размер {width}x{height}"
                    )
                    continue

                # Преобразуем в вектор
                vector = []
                for y in range(height):
                    for x in range(width):
                        pixel = img.getpixel((x, y))
                        if isinstance(pixel, int):
                            vector.append(
                                1 if pixel == 0 else 0
                            )  # 1 для чёрного, 0 для белого
                        elif len(pixel) in (1, 3):
                            vector.append(
                                1 if pixel[0] == 0 else 0
                            )  # 1 для чёрного, 0 для белого
                        else:
                            vector.append(0)  # По умолчанию белый (0)

                # Проверяем редактируемость
                black_pixels = vector.count(0)
                is_editable = black_pixels <= 4

                # Получаем информацию о правильности
                filename = os.path.basename(filepath)
                is_correct = self.get_file_correctness(filename)

                # Передаём в нейросеть
                self.network.process_image(is_editable, vector, filename, is_correct)
                processed_count += 1

            except PermissionError:
                self.log(f"Ошибка доступа к файлу: {os.path.basename(filepath)}")
            except Exception as e:
                self.log(f"Ошибка анализа {os.path.basename(filepath)}: {str(e)}")

        # Завершаем batch processing и получаем данные
        batch_data = self.network.finish_batch_processing(
            learning_rate, error_threshold
        )

        if batch_data:
            self.log(
                f"Успешно обработано {processed_count} из {len(self.selected_files)} файлов"
            )
            self.log("Данные доступны в нейросети для дальнейшей работы")

            # Экспортируем данные в файл
            self.network.export_batch_to_file("selected_images_data.txt")
        else:
            self.log("Не удалось обработать файлы")

    def get_file_correctness(self, filename):
        """Получает информацию о правильности файла из дерева"""
        for item in self.tree.get_children():
            if self.tree.item(item, "text") == filename:
                return self.tree.item(item, "values")[1] == "✅"
        return False

    def check_if_editable(self):
        """Проверяет, можно ли редактировать изображение (простая логика)"""
        # Считаем количество чёрных пикселей (в self.pixels чёрный = 0)
        black_pixels = sum(1 for row in self.pixels for pixel in row if pixel == 0)
        # Если чёрных пикселей больше 4, считаем изображение нередактируемым
        return black_pixels <= 4

    def toggle_pixel(self, event):
        x, y = event.x // 100, event.y // 100
        if 0 <= x < 3 and 0 <= y < 3:
            self.pixels[y][x] = 0 if self.pixels[y][x] == 255 else 255
            self.draw_pixels()
            self.log(
                f"Пиксель ({x}, {y}): {'чёрный' if self.pixels[y][x] == 0 else 'белый'}"
            )

    def draw_grid(self):
        for i in range(1, 3):
            self.canvas.create_line(i * 100, 0, i * 100, 300, fill="gray")
            self.canvas.create_line(0, i * 100, 300, i * 100, fill="gray")

    def draw_pixels(self):
        self.canvas.delete("pixel")
        for y in range(3):
            for x in range(3):
                color = "black" if self.pixels[y][x] == 0 else "white"
                self.canvas.create_rectangle(
                    x * 100,
                    y * 100,
                    (x + 1) * 100,
                    (y + 1) * 100,
                    fill=color,
                    outline="gray",
                    tags="pixel",
                )

    def populate_tree(self):
        """Заполняет дерево файлами"""
        self.tree.delete(*self.tree.get_children())

        # Собираем все BMP файлы
        bmp_files = []
        for filename in os.listdir(self.images_folder):
            if filename.lower().endswith(".bmp"):
                bmp_files.append(filename)

        # Сортируем файлы по имени
        bmp_files.sort()

        # Добавляем файлы в дерево
        for i, filename in enumerate(bmp_files):
            is_selected = i < 10  # Первые 10 файлов выбраны по умолчанию
            is_correct = i < 5  # Первые 5 файлов правильные по умолчанию

            item = self.tree.insert(
                "",
                "end",
                text=filename,
                values=("✓" if is_selected else "", "✅" if is_correct else ""),
            )

            if is_selected:
                full_path = os.path.join(self.images_folder, filename)
                self.selected_files.append(full_path)

        self.log(
            f"Загружено {len(bmp_files)} файлов, выбрано {len(self.selected_files)}, правильных: 5"
        )

    def on_tree_select(self, event):
        selection = self.tree.selection()
        if selection:
            filename = self.tree.item(selection[0], "text")
            filepath = os.path.join(self.images_folder, filename)
            self.load_image(filepath)

            # Обновляем текст кнопки выбора
            current_selected = self.tree.item(selection[0], "values")[0] == "✓"
            if current_selected:
                self.toggle_button.config(text="Исключить из выборки")
                # Активируем кнопку правильности только для выбранных файлов
                self.correct_button.config(state=tk.NORMAL)
                # Обновляем текст кнопки правильности
                current_correct = self.tree.item(selection[0], "values")[1] == "✅"
                if current_correct:
                    self.correct_button.config(text="Пометить как неправильное")
                else:
                    self.correct_button.config(text="Пометить как правильное")
            else:
                self.toggle_button.config(text="Добавить в выборку")
                # Блокируем кнопку правильности для невыбранных файлов
                self.correct_button.config(state=tk.DISABLED)

    def load_image(self, filepath):
        try:
            img = Image.open(filepath)
            width, height = img.size

            if (width, height) != (3, 3):
                self.log(f"Предупреждение: размер {width}x{height}, ожидается 3x3")

            # Загружаем пиксели
            self.pixels = []
            is_bw = True

            for y in range(height):
                row = []
                for x in range(width):
                    pixel = img.getpixel((x, y))
                    if isinstance(pixel, int):
                        if pixel not in (0, 255):
                            is_bw = False
                        row.append(pixel)
                    elif len(pixel) in (1, 3):
                        if len(set(pixel)) != 1 or pixel[0] not in (0, 255):
                            is_bw = False
                        row.append(pixel[0])
                    else:
                        is_bw = False
                        row.append(255)
                self.pixels.append(row)

            if not is_bw:
                messagebox.showwarning("Предупреждение", "Изображение не чёрно-белое!")

            self.draw_pixels()
        except Exception as e:
            self.log(f"Ошибка загрузки: {str(e)}")

    def log(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.config(state=tk.DISABLED)
        self.console.see(tk.END)

    def select_all_files(self):
        """Выбирает все файлы в дереве"""
        self.selected_files = []
        for item in self.tree.get_children():
            filename = self.tree.item(item, "text")
            full_path = os.path.join(self.images_folder, filename)
            self.selected_files.append(full_path)
            self.tree.set(item, "selected", "✓")
        self.log(f"Выбраны все файлы: {len(self.selected_files)}")

    def deselect_all_files(self):
        """Снимает выбор со всех файлов"""
        self.selected_files = []
        for item in self.tree.get_children():
            self.tree.set(item, "selected", "")
        self.log("Выбор снят для всех файлов")
        # Блокируем кнопку правильности
        self.correct_button.config(state=tk.DISABLED)

    def toggle_selected_file(self):
        """Переключает выбор файла"""
        selection = self.tree.selection()
        if selection:
            filename = self.tree.item(selection[0], "text")
            filepath = os.path.join(self.images_folder, filename)
            current_selected = self.tree.item(selection[0], "values")[0] == "✓"

            if current_selected:
                # Убираем из выбранных
                if filepath in self.selected_files:
                    self.selected_files.remove(filepath)
                self.tree.set(selection[0], "selected", "")
                self.log(f"✓ Файл '{filename}' убран из выбора")
                self.toggle_button.config(text="Добавить в выборку")
                # Блокируем кнопку правильности
                self.correct_button.config(state=tk.DISABLED)
            else:
                # Добавляем в выбранные
                if filepath not in self.selected_files:
                    self.selected_files.append(filepath)
                self.tree.set(selection[0], "selected", "✓")
                self.log(f"✓ Файл '{filename}' добавлен в выбор")
                self.toggle_button.config(text="Исключить из выборки")
                # Активируем кнопку правильности
                self.correct_button.config(state=tk.NORMAL)
                # Обновляем текст кнопки правильности
                current_correct = self.tree.item(selection[0], "values")[1] == "✅"
                if current_correct:
                    self.correct_button.config(text="Пометить как неправильное")
                else:
                    self.correct_button.config(text="Пометить как правильное")

            self.log(f"📊 Выбрано файлов: {len(self.selected_files)}")

    def toggle_correctness(self):
        """Переключает правильность выбранного файла"""
        selection = self.tree.selection()
        if selection:
            filename = self.tree.item(selection[0], "text")
            current_correct = self.tree.item(selection[0], "values")[1] == "✅"

            if current_correct:
                # Помечаем как неправильное
                self.tree.set(selection[0], "correct", "")
                self.correct_button.config(text="Пометить как правильное")
                self.log(f"❌ Файл '{filename}' помечен как неправильное")
            else:
                # Помечаем как правильное
                self.tree.set(selection[0], "correct", "✅")
                self.correct_button.config(text="Пометить как неправильное")
                self.log(f"✅ Файл '{filename}' помечен как правильное")
