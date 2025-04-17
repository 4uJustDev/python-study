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

        # Создаем главные контейнеры
        self.main_frame = Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Нижний фрейм для текста
        self.bottom_frame = Frame(root)
        self.bottom_frame.pack(fill=tk.X, padx=5, pady=5)

        # Левая панель (список изображений)
        self.left_frame = Frame(self.main_frame, width=200, bg="lightgray")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Правая панель (изображение и гистограмма)
        self.right_frame = Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Верхняя часть правой панели (изображение)
        self.image_frame = Frame(self.right_frame, height=400, bg="white")
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Нижняя часть правой панели (гистограмма)
        self.hist_frame = Frame(self.right_frame, height=300)
        self.hist_frame.pack(fill=tk.BOTH)

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

        # Элементы для изображения (сделаем побольше)
        self.image_label = Label(self.image_frame, bg="white")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Элементы для гистограммы
        self.fig, self.ax = plt.subplots(figsize=(6, 2.5), dpi=100)
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        # Кнопка для выбора другой папки
        self.btn_change_folder = tk.Button(
            self.left_frame, text="Выбрать другую папку", command=self.change_folder
        )
        self.btn_change_folder.pack(pady=10, fill=tk.X)

        # Текстовое поле внизу
        self.text_label = Label(
            self.bottom_frame, text="Hello world", font=("Arial", 12)
        )
        self.text_label.pack(fill=tk.X)

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
            self.display_image(img)
            self.plot_histogram(img)
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")

    def display_image(self, img):
        """Отображаем изображение с масштабированием"""
        # Масштабируем изображение, сохраняя пропорции
        frame_width = self.image_frame.winfo_width() - 20
        frame_height = self.image_frame.winfo_height() - 20

        img_ratio = img.width / img.height
        frame_ratio = frame_width / frame_height

        if frame_ratio > img_ratio:
            new_height = frame_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = frame_width
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def plot_histogram(self, img):
        """Строим гистограмму для черно-белого изображения"""
        img_array = np.array(img)
        self.ax.clear()

        # Строим гистограмму с увеличенным отступом снизу
        self.ax.hist(
            img_array.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7
        )
        self.ax.set_title("Гистограмма яркости")
        self.ax.set_xlabel("Уровень яркости")
        self.ax.set_ylabel(
            "Количество пикселей", labelpad=15
        )  # Добавляем отступ для подписи
        self.ax.set_xlim(0, 255)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # Увеличиваем отступы вокруг графика
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    def change_folder(self):
        """Меняем папку с изображениями"""
        folder = filedialog.askdirectory(initialdir=self.folder_path)
        if folder:
            self.folder_path = folder
            self.load_images()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x700")
    app = ImageViewer(root)
    root.mainloop()
