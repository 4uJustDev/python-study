import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar, Frame, Label


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Просмотр черно-белых изображений с гистограммой")

        # Установим начальный путь к папке с изображениями
        self.folder_path = "programs/labs/photoCorrection/photos/"

        # Создаем контейнеры
        self.left_frame = Frame(root, width=200, height=600, bg="lightgray")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = Frame(root, width=800, height=600)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Разделим правый фрейм на две части: изображение и гистограмма
        self.image_frame = Frame(self.right_frame, width=400, height=600, bg="white")
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.hist_frame = Frame(self.right_frame, width=400, height=600, bg="white")
        self.hist_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Элементы в левом фрейме
        self.list_label = Label(
            self.left_frame, text="Черно-белые изображения:", bg="lightgray"
        )
        self.list_label.pack(pady=5)

        self.scrollbar = Scrollbar(self.left_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_list = Listbox(
            self.left_frame, yscrollcommand=self.scrollbar.set, width=30
        )
        self.image_list.pack(expand=True, fill=tk.BOTH)
        self.image_list.bind("<<ListboxSelect>>", self.show_image_and_histogram)

        self.scrollbar.config(command=self.image_list.yview)

        # Элементы в правом фрейме (изображение)
        self.image_label = Label(self.image_frame, bg="white")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Элементы для гистограммы
        self.hist_label = Label(self.hist_frame, text="Гистограмма яркости", bg="white")
        self.hist_label.pack(pady=5)

        # Создаем фигуру matplotlib для гистограммы
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Кнопка для выбора другой папки
        self.btn_change_folder = tk.Button(
            self.left_frame, text="Выбрать другую папку", command=self.change_folder
        )
        self.btn_change_folder.pack(pady=10, fill=tk.X)

        # Загружаем изображения
        self.load_images()

    def load_images(self):
        """Загружаем список изображений с префиксом bw_"""
        self.image_list.delete(0, tk.END)
        try:
            files = os.listdir(self.folder_path)
            bw_images = [
                f
                for f in files
                if f.startswith("bw_")
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]

            for img in bw_images:
                self.image_list.insert(tk.END, img)

            if bw_images:
                self.image_list.selection_set(0)
                self.show_image_and_histogram(None)
        except Exception as e:
            print(f"Ошибка при загрузке изображений: {e}")

    def show_image_and_histogram(self, event):
        """Отображаем выбранное изображение и его гистограмму"""
        selection = self.image_list.curselection()
        if not selection:
            return

        selected_image = self.image_list.get(selection[0])
        image_path = os.path.join(self.folder_path, selected_image)

        try:
            img = Image.open(image_path)

            # Отображение изображения
            self.display_image(img)

            # Построение гистограммы
            self.plot_histogram(img)

        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")

    def display_image(self, img):
        """Отображаем изображение с масштабированием"""
        # Масштабируем изображение под размер окна
        max_width = self.image_frame.winfo_width() - 20
        max_height = self.image_frame.winfo_height() - 20

        img.thumbnail((max_width, max_height), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # сохраняем ссылку

    def plot_histogram(self, img):
        """Строим гистограмму для черно-белого изображения"""
        # Преобразуем изображение в numpy array
        img_array = np.array(img)

        # Очищаем предыдущую гистограмму
        self.ax.clear()

        # Строим гистограмму
        self.ax.hist(
            img_array.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7
        )
        self.ax.set_title("Гистограмма яркости")
        self.ax.set_xlabel("Уровень яркости")
        self.ax.set_ylabel("Количество пикселей")
        self.ax.set_xlim(0, 255)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # Обновляем canvas
        self.canvas.draw()

    def change_folder(self):
        """Меняем папку с изображениями"""
        folder = filedialog.askdirectory(initialdir=self.folder_path)
        if folder:
            self.folder_path = folder
            self.load_images()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x600")
    app = ImageViewer(root)
    root.mainloop()
