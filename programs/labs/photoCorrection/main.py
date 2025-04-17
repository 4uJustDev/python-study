import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar, Frame, Label, Text, END


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор черно-белых изображений")

        # Установим начальный путь к папке с изображениями
        self.folder_path = "programs/labs/photoCorrection/photos/"

        # Создаем главные контейнеры
        self.main_frame = Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Нижний фрейм для текста
        self.bottom_frame = Frame(root)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

        # Элементы для изображения
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

        # Текстовое поле для анализа изображения
        self.analysis_label = Label(
            self.bottom_frame, text="Анализ изображения:", font=("Arial", 10, "bold")
        )
        self.analysis_label.pack(anchor="w")

        self.analysis_text = Text(self.bottom_frame, height=6, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)

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
            self.analyze_image(img)
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")

    def display_image(self, img):
        """Отображаем изображение с масштабированием"""
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

        self.ax.hist(
            img_array.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7
        )
        self.ax.set_title("Гистограмма яркости")
        self.ax.set_xlabel("Уровень яркости")
        self.ax.set_ylabel("Количество пикселей", labelpad=15)
        self.ax.set_xlim(0, 255)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    def analyze_image(self, img):
        """Улучшенный анализ изображения с детальной проверкой гистограммы"""

        img_array = np.array(img)
        width, height = img.size
        depth = img_array.itemsize * 8

        # Анализ гистограммы
        hist, bins = np.histogram(img_array, bins=256, range=(0, 256))
        total_pixels = width * height

        # Основные параметры
        analysis = f"▌ Анализ изображения {width}x{height}px, {depth} бит\n"
        analysis += f"▌ Общее количество пикселей: {total_pixels:,}\n"

        # 1. Проверка на полностью чёрное/белое изображение
        if np.all(img_array == 0):
            self.analysis_text.delete(1.0, END)
            self.analysis_text.insert(END, "▌ ВНИМАНИЕ: Полностью чёрное изображение!")
            return
        elif np.all(img_array == 255):
            self.analysis_text.delete(1.0, END)
            self.analysis_text.insert(END, "▌ ВНИМАНИЕ: Полностью белое изображение!")
            return

        # 2. Анализ распределения яркостей
        dark_threshold = int(0.1 * 255)  # Порог для теней
        light_threshold = int(0.9 * 255)  # Порог для светов

        dark_pixels = np.sum(hist[:dark_threshold])
        light_pixels = np.sum(hist[light_threshold:])
        mid_pixels = np.sum(hist[dark_threshold:light_threshold])

        # analysis += f"\n▌ Распределение тонов:\n"
        # analysis += f"• Тени (0-{dark_threshold}): {dark_pixels/total_pixels:.1%}\n"
        # analysis += f"• Средние тона: {mid_pixels/total_pixels:.1%}\n"
        # analysis += (
        #     f"• Света ({light_threshold}-255): {light_pixels/total_pixels:.1%}\n"
        # )

        # 3. Поиск проблем
        problems = []

        # Проверка на переэкспонирование (пересветы)
        if light_pixels > 0.3 * total_pixels:
            problems.append("Переэкспонирование: более 30% пикселей в ярких областях")

        # Проверка на недоэкспонирование (недосвет)
        if dark_pixels > 0.3 * total_pixels:
            problems.append("Недоэкспонирование: более 30% пикселей в тёмных областях")

        # Проверка на низкий контраст
        dynamic_range = np.max(img_array) - np.min(img_array)
        if dynamic_range < 100:
            problems.append(f"Низкий контраст (диапазон всего {dynamic_range} из 255)")

        # Проверка на постеризацию
        unique_values = len(np.unique(img_array))
        if unique_values < 100:
            problems.append(f"Постеризация: только {unique_values} уникальных оттенков")

        # 4. Формирование итогового вывода
        # if problems:
        #     analysis += "\n▌ ПРОБЛЕМЫ ОБНАРУЖЕНЫ:\n"
        #     for i, problem in enumerate(problems, 1):
        #         analysis += f"{i}. {problem}\n"

        #     # Дополнительные рекомендации
        #     analysis += "\n▌ Рекомендации:\n"
        #     if "Переэкспонирование" in problems:
        #         analysis += "- Уменьшите экспозицию или восстановите света\n"
        #     if "Недоэкспонирование" in problems:
        #         analysis += "- Увеличьте экспозицию или вытяните тени\n"
        #     if "Низкий контраст" in problems:
        #         analysis += "- Примените коррекцию уровней или кривых\n"
        #     if "Постеризация" in problems:
        #         analysis += "- Используйте дизеринг при конвертации в ЧБ\n"
        # else:
        #     analysis += "\n▌ Качество изображения в норме\n"
        #     analysis += "- Хороший баланс светов и теней\n"
        #     analysis += "- Достаточный динамический диапазон\n"

        self.analysis_text.delete(1.0, END)
        self.analysis_text.insert(END, analysis)

    def change_folder(self):
        """Меняем папку с изображениями"""
        folder = filedialog.askdirectory(initialdir=self.folder_path)
        if folder:
            self.folder_path = folder
            self.load_images()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x800")
    app = ImageViewer(root)
    root.mainloop()
