import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сравнение алгоритмов выделения контуров")
        self.root.geometry("1200x800")

        # Переменные
        self.original_image = None
        self.processed_image = None
        self.current_operator = tk.StringVar(value="sobel")

        self.create_menu()
        self.create_main_layout()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        # Меню "Файл"
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть изображение", command=self.load_image)
        file_menu.add_command(label="Сохранить результат", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        # Меню "Операторы"
        operators_menu = tk.Menu(menubar, tearoff=0)
        operators_menu.add_radiobutton(
            label="Робертса", variable=self.current_operator, value="roberts"
        )
        operators_menu.add_radiobutton(
            label="Превитта", variable=self.current_operator, value="prewitt"
        )
        operators_menu.add_radiobutton(
            label="Собела", variable=self.current_operator, value="sobel"
        )
        operators_menu.add_radiobutton(
            label="Лапласиан Гауссиана", variable=self.current_operator, value="log"
        )
        operators_menu.add_radiobutton(
            label="Канни", variable=self.current_operator, value="canny"
        )
        menubar.add_cascade(label="Операторы", menu=operators_menu)

        self.root.config(menu=menubar)

    def create_main_layout(self):
        # Основной контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель (исходное изображение)
        left_frame = ttk.LabelFrame(main_frame, text="Исходное изображение")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_image_label = ttk.Label(left_frame)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)

        # Правая панель (обработанное изображение)
        right_frame = ttk.LabelFrame(main_frame, text="Обработанное изображение")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.processed_image_label = ttk.Label(right_frame)
        self.processed_image_label.pack(fill=tk.BOTH, expand=True)

        # Панель управления
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            control_frame, text="Применить оператор", command=self.apply_operator
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Очистить", command=self.clear_images).pack(
            side=tk.LEFT, padx=5
        )

        # Таблица результатов
        results_frame = ttk.LabelFrame(self.root, text="Результаты")
        results_frame.pack(fill=tk.X, padx=10, pady=5)

        self.results_tree = ttk.Treeview(
            results_frame, columns=("operator", "area", "efficiency"), show="headings"
        )
        self.results_tree.heading("operator", text="Оператор")
        self.results_tree.heading("area", text="Площадь контуров")
        self.results_tree.heading("efficiency", text="Эффективность")
        self.results_tree.pack(fill=tk.X, padx=5, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.display_image(self.original_image, self.original_image_label)
            except Exception as e:
                messagebox.showerror(
                    "Ошибка", f"Не удалось загрузить изображение: {str(e)}"
                )

    def display_image(self, cv_image, label):
        if cv_image is None:
            return

        # Конвертация из BGR в RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Изменение размера изображения для отображения
        height, width = rgb_image.shape[:2]
        max_size = 400
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))

        # Конвертация в формат для Tkinter
        image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image=image)

        label.configure(image=photo)
        label.image = photo

    def apply_operator(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        operator = self.current_operator.get()
        try:
            if operator == "sobel":
                self.processed_image = self.apply_sobel(self.original_image)
            # Добавление других операторов будет реализовано позже

            self.display_image(self.processed_image, self.processed_image_label)
            self.update_results(operator)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при применении оператора: {str(e)}")

    def apply_sobel(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    def update_results(self, operator):
        if self.processed_image is None:
            return

        # Расчет площади контуров (пример)
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        area = np.sum(binary > 0)

        # Добавление результатов в таблицу
        self.results_tree.insert("", "end", values=(operator, area, "В разработке"))

    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning(
                "Предупреждение", "Нет обработанного изображения для сохранения"
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Успех", "Изображение успешно сохранено")
            except Exception as e:
                messagebox.showerror(
                    "Ошибка", f"Не удалось сохранить изображение: {str(e)}"
                )

    def clear_images(self):
        self.original_image = None
        self.processed_image = None
        self.original_image_label.configure(image="")
        self.processed_image_label.configure(image="")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)


if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
