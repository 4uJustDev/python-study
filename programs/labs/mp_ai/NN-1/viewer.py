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

        # Дерево файлов
        self.tree = ttk.Treeview(left_panel)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

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
                vector.append(0 if pixel == 0 else 1)  # 0 для чёрного, 1 для белого

        # Проверяем, можно ли редактировать изображение
        is_editable = self.check_if_editable()

        # Передаём данные в нейросеть
        self.network.process_image(is_editable, vector)

        self.log(f"Вектор изображения: {vector}")
        self.log(f"Редактируемое: {'Да' if is_editable else 'Нет'}")

    def check_if_editable(self):
        """Проверяет, можно ли редактировать изображение (простая логика)"""
        # Считаем количество чёрных пикселей
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
        self.tree.delete(*self.tree.get_children())
        root_node = self.tree.insert(
            "", "end", text=os.path.basename(self.images_folder), open=True
        )

        for dirpath, _, filenames in os.walk(self.images_folder):
            rel_path = os.path.relpath(dirpath, self.images_folder)
            parent_node = (
                root_node
                if rel_path == "."
                else self._get_or_create_node(root_node, rel_path)
            )

            for filename in filenames:
                if filename.lower().endswith(".bmp"):
                    full_path = os.path.join(dirpath, filename)
                    self.tree.insert(
                        parent_node, "end", text=filename, values=[full_path]
                    )

    def _get_or_create_node(self, root_node, rel_path):
        parent_node = root_node
        for part in rel_path.split(os.sep):
            existing_child = next(
                (
                    child
                    for child in self.tree.get_children(parent_node)
                    if self.tree.item(child, "text") == part
                ),
                None,
            )
            if existing_child is None:
                parent_node = self.tree.insert(parent_node, "end", text=part, open=True)
            else:
                parent_node = existing_child
        return parent_node

    def on_tree_select(self, event):
        selection = self.tree.selection()
        if selection:
            item_values = self.tree.item(selection[0], "values")
            if item_values:
                self.load_image(item_values[0])

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
            self.print_image_info(filepath)
        except Exception as e:
            self.log(f"Ошибка загрузки: {str(e)}")

    def print_image_info(self, filepath):
        self.log(f"Файл: {os.path.basename(filepath)}")
        self.log("Пиксели (0 — чёрный, 255 — белый):")
        for row in self.pixels:
            self.log(" ".join(str(pixel).ljust(3) for pixel in row))
        self.log("-" * 40)

    def log(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.config(state=tk.DISABLED)
        self.console.see(tk.END)
