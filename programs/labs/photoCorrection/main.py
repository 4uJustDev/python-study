import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import (
    filedialog,
    Listbox,
    Scrollbar,
    Frame,
    Label,
    Text,
    END,
    ttk,
    LabelFrame,
)
from skimage import exposure
import cv2
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure


class FunctionVisualizer(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Визуализация функции преобразования")
        self.geometry("600x400")

        # Создаем фигуру matplotlib
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Создаем холст для отображения графика
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Добавляем панель инструментов
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        # Инициализируем график
        self.update_plot("none")

    def update_plot(self, method, **params):
        """Обновление графика в зависимости от выбранного метода"""
        self.ax.clear()
        x = np.linspace(0, 255, 256)

        if method == "logarithmic":
            c = params.get("c", 45)
            y = c * np.log(1 + x)
            self.ax.plot(x, y, "b-", label=f"c = {c}")
            self.ax.set_title("Логарифмическое преобразование")

        elif method == "power":
            gamma = params.get("gamma", 1.0)
            y = 255 * (x / 255) ** gamma
            self.ax.plot(x, y, "b-", label=f"γ = {gamma:.1f}")
            self.ax.set_title("Степенное преобразование")

        elif method == "piecewise":
            threshold = params.get("threshold", 128)
            slope_low = params.get("slope_low", 0.5)
            slope_high = params.get("slope_high", 1.5)

            y = np.piecewise(
                x,
                [x < threshold, x >= threshold],
                [
                    lambda x: slope_low * x,
                    lambda x: slope_low * threshold + slope_high * (x - threshold),
                ],
            )

            self.ax.plot(
                x,
                y,
                "b-",
                label=f"Порог = {threshold}, k₁ = {slope_low:.1f}, k₂ = {slope_high:.1f}",
            )
            self.ax.set_title("Кусочно-линейное преобразование")

        elif method == "gaussian":
            mean = params.get("mean", 128)
            std = params.get("std", 50)
            y = 255 * np.exp(-((x - mean) ** 2) / (2 * std**2))
            self.ax.plot(x, y, "b-", label=f"μ = {mean}, σ = {std}")
            self.ax.set_title("Гауссово распределение")

        elif method == "exponential":
            lambda_param = params.get("lambda_param", 0.05)
            y = 255 * (1 - np.exp(-lambda_param * x))
            self.ax.plot(x, y, "b-", label=f"λ = {lambda_param:.2f}")
            self.ax.set_title("Экспоненциальное распределение")

        else:
            # Линейное преобразование (y = x)
            self.ax.plot(x, x, "b-", label="y = x")
            self.ax.set_title("Линейное преобразование")

        # Настройка осей и сетки
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.set_xlabel("Входное значение")
        self.ax.set_ylabel("Выходное значение")
        self.ax.legend()

        # Обновление холста
        self.canvas.draw()


class ImageCorrector:
    @staticmethod
    def polarity(image):
        """Инверсия изображения (полярность)"""
        img_array = np.array(image)
        return Image.fromarray(255 - img_array)

    @staticmethod
    def logarithmic(image, c=45):
        """Логарифмическое преобразование"""
        img_array = np.array(image).astype(float)
        img_log = c * np.log(1 + img_array)
        img_log = np.clip(img_log, 0, 255).astype(np.uint8)
        return Image.fromarray(img_log)

    @staticmethod
    def gamma(image, gamma=1.0):
        """Степенное преобразование (гамма-коррекция)"""
        img_array = np.array(image)
        img_gamma = exposure.adjust_gamma(img_array, gamma)
        return Image.fromarray(img_gamma)

    @staticmethod
    def piecewise_linear(image, threshold=128, slope_low=0.5, slope_high=1.5):
        """Кусочно-линейное преобразование"""
        img_array = np.array(image).astype(float)

        # Создаем маску для темных и светлых пикселей
        mask_dark = img_array < threshold
        mask_light = ~mask_dark

        # Применяем разные наклоны
        result = np.zeros_like(img_array)
        result[mask_dark] = slope_low * img_array[mask_dark]
        result[mask_light] = threshold + slope_high * (
            img_array[mask_light] - threshold
        )

        # Нормализация в диапазон [0, 255]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    @staticmethod
    def normalize(image):
        """Нормализация гистограммы"""
        img_array = np.array(image)
        img_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(img_norm)

    @staticmethod
    def equalize(image):
        """Эквализация гистограммы"""
        img_array = np.array(image)
        img_eq = cv2.equalizeHist(img_array)
        return Image.fromarray(img_eq)

    @staticmethod
    def gaussian_mapping(image, mean=128, std=50):
        """Приведение к гауссовому распределению"""
        img_array = np.array(image)

        # Создаем целевое гауссово распределение
        x = np.arange(256)
        target_hist = np.exp(-((x - mean) ** 2) / (2 * std**2))
        target_hist = target_hist / np.sum(target_hist)

        # Вычисляем кумулятивную функцию распределения
        cdf = np.cumsum(target_hist)

        # Нормализуем CDF
        cdf = cdf * 255

        # Применяем преобразование
        img_mapped = cdf[img_array]
        return Image.fromarray(img_mapped.astype(np.uint8))

    @staticmethod
    def exponential_mapping(image, lambda_param=0.05):
        """Приведение к экспоненциальному распределению"""
        img_array = np.array(image)

        # Создаем целевое экспоненциальное распределение
        x = np.arange(256)
        target_hist = lambda_param * np.exp(-lambda_param * x)
        target_hist = target_hist / np.sum(target_hist)

        # Вычисляем кумулятивную функцию распределения
        cdf = np.cumsum(target_hist)

        # Нормализуем CDF
        cdf = cdf * 255

        # Применяем преобразование
        img_mapped = cdf[img_array]
        return Image.fromarray(img_mapped.astype(np.uint8))


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Градационная коррекция чёрно-белых изображений")

        # Создаем окно визуализации функций
        self.function_visualizer = FunctionVisualizer(root)
        self.function_visualizer.withdraw()  # Скрываем окно при старте

        # Установим начальный путь к папке с изображениями
        self.folder_path = "photos/"

        # Текущее изображение
        self.current_image = None
        self.original_image = None

        # Создаем главные контейнеры
        self.main_frame = Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Нижний фрейм для текста
        self.bottom_frame = Frame(root)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая панель (список изображений и настройки)
        self.left_frame = Frame(self.main_frame, width=300, bg="lightgray")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Правая панель (изображение и гистограмма)
        self.right_frame = Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Верхняя часть правой панели (изображение)
        self.image_frame = Frame(self.right_frame, width=600, height=400, bg="black")
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)

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

        # Фрейм для формул
        self.formula_frame = LabelFrame(
            self.left_frame, text="Формула преобразования", bg="#f8f8f8"
        )
        self.formula_frame.pack(fill=tk.X, pady=5, padx=5)

        self.formula_text = Text(
            self.formula_frame, height=4, wrap=tk.WORD, bg="#f8f8f8", font=("Arial", 10)
        )
        self.formula_text.pack(fill=tk.X, padx=5, pady=5)
        self.formula_text.insert(END, "Выберите метод преобразования")
        self.formula_text.config(state=tk.DISABLED)

        # Фрейм для базовых преобразований
        self.basic_frame = LabelFrame(
            self.left_frame, text="Базовые преобразования", bg="#f0f8ff"
        )
        self.basic_frame.pack(fill=tk.X, pady=5, padx=5)

        self.basic_method = ttk.Combobox(
            self.basic_frame,
            values=[
                "Без коррекции",
                "Полярность",
                "Логарифмическое",
                "Степенное (гамма)",
                "Кусочно-линейное",
            ],
            state="readonly",
        )
        self.basic_method.set("Без коррекции")
        self.basic_method.pack(fill=tk.X, pady=5, padx=5)
        self.basic_method.bind("<<ComboboxSelected>>", self.on_basic_method_change)

        # Фрейм для гистограммных преобразований
        self.hist_frame = LabelFrame(
            self.left_frame, text="Гистограммные преобразования", bg="#f5f5f5"
        )
        self.hist_frame.pack(fill=tk.X, pady=5, padx=5)

        self.hist_method = ttk.Combobox(
            self.hist_frame,
            values=[
                "Без коррекции",
                "Нормализация",
                "Эквализация",
                "Приведение к заданной функции",
            ],
            state="readonly",
        )
        self.hist_method.set("Без коррекции")
        self.hist_method.pack(fill=tk.X, pady=5, padx=5)
        self.hist_method.bind("<<ComboboxSelected>>", self.on_hist_method_change)

        # Фрейм для параметров функции
        self.function_frame = LabelFrame(
            self.left_frame, text="Параметры функции", bg="#f0f0f0"
        )
        self.function_frame.pack(fill=tk.X, pady=5, padx=5)

        self.dist_method = ttk.Combobox(
            self.function_frame,
            values=["Гауссова", "Экспоненциальная"],
            state="readonly",
        )
        self.dist_method.set("Гауссова")
        self.dist_method.pack(fill=tk.X, pady=5, padx=5)
        self.dist_method.bind("<<ComboboxSelected>>", self.on_dist_method_change)

        # Кнопка для показа/скрытия окна визуализации
        self.btn_show_visualizer = tk.Button(
            self.left_frame,
            text="Показать визуализацию функции",
            command=self.toggle_visualizer,
        )
        self.btn_show_visualizer.pack(pady=5, fill=tk.X)

        # Фреймы для параметров
        self.params_frame = Frame(self.left_frame, bg="lightgray")
        self.params_frame.pack(fill=tk.X, pady=5)

        # Параметры для логарифмического преобразования
        self.log_frame = Frame(self.params_frame, bg="lightgray")
        self.log_label = Label(self.log_frame, text="Коэффициент C:", bg="lightgray")
        self.log_label.pack(side=tk.LEFT)
        self.log_value = Label(self.log_frame, text="45", bg="lightgray")
        self.log_value.pack(side=tk.RIGHT)
        self.log_scale = ttk.Scale(
            self.log_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.update_log,
        )
        self.log_scale.set(45)
        self.log_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Параметры для гамма-коррекции
        self.gamma_frame = Frame(self.params_frame, bg="lightgray")
        self.gamma_label = Label(self.gamma_frame, text="Гамма:", bg="lightgray")
        self.gamma_label.pack(side=tk.LEFT)
        self.gamma_value = Label(self.gamma_frame, text="1.0", bg="lightgray")
        self.gamma_value.pack(side=tk.RIGHT)
        self.gamma_scale = ttk.Scale(
            self.gamma_frame,
            from_=0.1,
            to=3.0,
            orient=tk.HORIZONTAL,
            command=self.update_gamma,
        )
        self.gamma_scale.set(1.0)
        self.gamma_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Параметры для кусочно-линейного преобразования
        self.piecewise_frame = Frame(self.params_frame, bg="lightgray")

        # Порог
        self.threshold_frame = Frame(self.piecewise_frame, bg="lightgray")
        self.threshold_label = Label(
            self.threshold_frame, text="Порог T:", bg="lightgray"
        )
        self.threshold_label.pack(side=tk.LEFT)
        self.threshold_value = Label(self.threshold_frame, text="128", bg="lightgray")
        self.threshold_value.pack(side=tk.RIGHT)
        self.threshold_scale = ttk.Scale(
            self.threshold_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            command=self.update_threshold,
        )
        self.threshold_scale.set(128)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Наклон для темных пикселей
        self.slope_low_frame = Frame(self.piecewise_frame, bg="lightgray")
        self.slope_low_label = Label(
            self.slope_low_frame, text="Наклон a:", bg="lightgray"
        )
        self.slope_low_label.pack(side=tk.LEFT)
        self.slope_low_value = Label(self.slope_low_frame, text="0.5", bg="lightgray")
        self.slope_low_value.pack(side=tk.RIGHT)
        self.slope_low_scale = ttk.Scale(
            self.slope_low_frame,
            from_=0.1,
            to=2.0,
            orient=tk.HORIZONTAL,
            command=self.update_slope_low,
        )
        self.slope_low_scale.set(0.5)
        self.slope_low_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Наклон для светлых пикселей
        self.slope_high_frame = Frame(self.piecewise_frame, bg="lightgray")
        self.slope_high_label = Label(
            self.slope_high_frame, text="Наклон b:", bg="lightgray"
        )
        self.slope_high_label.pack(side=tk.LEFT)
        self.slope_high_value = Label(self.slope_high_frame, text="1.5", bg="lightgray")
        self.slope_high_value.pack(side=tk.RIGHT)
        self.slope_high_scale = ttk.Scale(
            self.slope_high_frame,
            from_=0.1,
            to=2.0,
            orient=tk.HORIZONTAL,
            command=self.update_slope_high,
        )
        self.slope_high_scale.set(1.5)
        self.slope_high_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Параметры для гауссова распределения
        self.gaussian_frame = Frame(self.params_frame, bg="lightgray")

        # Среднее значение
        self.mean_frame = Frame(self.gaussian_frame, bg="lightgray")
        self.mean_label = Label(self.mean_frame, text="μ:", bg="lightgray")
        self.mean_label.pack(side=tk.LEFT)
        self.mean_value = Label(self.mean_frame, text="128", bg="lightgray")
        self.mean_value.pack(side=tk.RIGHT)
        self.mean_scale = ttk.Scale(
            self.mean_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            command=self.update_mean,
        )
        self.mean_scale.set(128)
        self.mean_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Стандартное отклонение
        self.std_frame = Frame(self.gaussian_frame, bg="lightgray")
        self.std_label = Label(self.std_frame, text="σ:", bg="lightgray")
        self.std_label.pack(side=tk.LEFT)
        self.std_value = Label(self.std_frame, text="50", bg="lightgray")
        self.std_value.pack(side=tk.RIGHT)
        self.std_scale = ttk.Scale(
            self.std_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.update_std,
        )
        self.std_scale.set(50)
        self.std_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Параметры для экспоненциального распределения
        self.exp_frame = Frame(self.params_frame, bg="lightgray")
        self.lambda_label = Label(self.exp_frame, text="λ:", bg="lightgray")
        self.lambda_label.pack(side=tk.LEFT)
        self.lambda_value = Label(self.exp_frame, text="0.05", bg="lightgray")
        self.lambda_value.pack(side=tk.RIGHT)
        self.lambda_scale = ttk.Scale(
            self.exp_frame,
            from_=0.01,
            to=0.2,
            orient=tk.HORIZONTAL,
            command=self.update_lambda,
        )
        self.lambda_scale.set(0.05)
        self.lambda_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Текстовое поле для анализа изображения
        self.analysis_label = Label(
            self.bottom_frame, text="Анализ изображения:", font=("Arial", 10, "bold")
        )
        self.analysis_label.pack(anchor="w")

        self.analysis_text = Text(self.bottom_frame, height=6, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)

        # Загружаем изображения
        self.load_images()

    def toggle_visualizer(self):
        """Показать/скрыть окно визуализации функций"""
        if self.function_visualizer.state() == "withdrawn":
            self.function_visualizer.deiconify()
            self.btn_show_visualizer.config(text="Скрыть визуализацию функции")
        else:
            self.function_visualizer.withdraw()
            self.btn_show_visualizer.config(text="Показать визуализацию функции")

    def update_formula(self):
        """Обновление отображаемой формулы"""
        self.formula_text.config(state=tk.NORMAL)
        self.formula_text.delete(1.0, END)

        basic_method = self.basic_method.get()
        hist_method = self.hist_method.get()
        dist_method = self.dist_method.get()

        if basic_method == "Полярность":
            formula = "I_out = 255 - I_in"
        elif basic_method == "Логарифмическое":
            formula = "I_out = c * log(1 + I_in)"
        elif basic_method == "Степенное (гамма)":
            formula = "I_out = 255 * (I_in/255)^γ"
        elif basic_method == "Кусочно-линейное":
            formula = (
                "I_out = a*I_in (при I_in < T)\nI_out = b*(I_in-T)+a*T (при I_in >= T)"
            )
        elif hist_method == "Приведение к заданной функции":
            if dist_method == "Гауссова":
                formula = "PDF = exp(-(x-μ)²/(2σ²))"
            else:  # Экспоненциальная
                formula = "PDF = λ * exp(-λx)"
        else:
            formula = "Выберите метод преобразования"

        self.formula_text.insert(END, formula)
        self.formula_text.config(state=tk.DISABLED)

    def update_correction_ui(self, event=None):
        """Обновление интерфейса в зависимости от выбранных методов"""
        # Скрываем все фреймы параметров
        self.log_frame.pack_forget()
        self.gamma_frame.pack_forget()
        self.piecewise_frame.pack_forget()
        self.threshold_frame.pack_forget()
        self.slope_low_frame.pack_forget()
        self.slope_high_frame.pack_forget()
        self.gaussian_frame.pack_forget()
        self.mean_frame.pack_forget()
        self.std_frame.pack_forget()
        self.exp_frame.pack_forget()
        self.function_frame.pack_forget()

        # Показываем нужные элементы в зависимости от выбранных методов
        basic_method = self.basic_method.get()
        hist_method = self.hist_method.get()

        if basic_method == "Логарифмическое":
            self.log_frame.pack(fill=tk.X, pady=5)
        elif basic_method == "Степенное (гамма)":
            self.gamma_frame.pack(fill=tk.X, pady=5)
        elif basic_method == "Кусочно-линейное":
            self.piecewise_frame.pack(fill=tk.X, pady=5)
            self.threshold_frame.pack(fill=tk.X, pady=5)
            self.slope_low_frame.pack(fill=tk.X, pady=5)
            self.slope_high_frame.pack(fill=tk.X, pady=5)

        if hist_method == "Приведение к заданной функции":
            self.function_frame.pack(fill=tk.X, pady=5)

            if self.dist_method.get() == "Гауссова":
                self.gaussian_frame.pack(fill=tk.X, pady=5)
                self.mean_frame.pack(fill=tk.X, pady=5)
                self.std_frame.pack(fill=tk.X, pady=5)
            else:  # Экспоненциальная
                self.exp_frame.pack(fill=tk.X, pady=5)

        # Обновляем формулу
        self.update_formula()

        # Обновляем визуализацию функции
        self.update_function_visualization()

        self.apply_correction()

    def update_function_visualization(self):
        """Обновление визуализации функции в зависимости от выбранного метода"""
        if not hasattr(self, "function_visualizer"):
            self.function_visualizer = FunctionVisualizer(self)

        method = self.basic_method.get()
        if method == "Логарифмическое":
            c = float(self.log_scale.get())
            self.function_visualizer.update_plot("logarithmic", c=c)
        elif method == "Степенное (гамма)":
            gamma = float(self.gamma_scale.get())
            self.function_visualizer.update_plot("power", gamma=gamma)
        elif method == "Кусочно-линейное":
            threshold = float(self.threshold_scale.get())
            slope_low = float(self.slope_low_scale.get())
            slope_high = float(self.slope_high_scale.get())
            self.function_visualizer.update_plot(
                "piecewise",
                threshold=threshold,
                slope_low=slope_low,
                slope_high=slope_high,
            )
        elif (
            self.hist_method.get() == "Приведение к заданной функции"
            and self.dist_method.get() == "Гауссова"
        ):
            mean = float(self.mean_scale.get())
            std = float(self.std_scale.get())
            self.function_visualizer.update_plot("gaussian", mean=mean, std=std)
        elif (
            self.hist_method.get() == "Приведение к заданной функции"
            and self.dist_method.get() == "Экспоненциальная"
        ):
            lambda_param = float(self.lambda_scale.get())
            self.function_visualizer.update_plot(
                "exponential", lambda_param=lambda_param
            )

    def update_log(self, value):
        """Обновление значения логарифмического преобразования"""
        self.log_value.config(text=f"{float(value):.0f}")
        if self.basic_method.get() == "Логарифмическое":
            self.apply_correction()
            self.update_function_visualization()

    def update_gamma(self, value):
        """Обновление значения гамма-коррекции"""
        self.gamma_value.config(text=f"{float(value):.1f}")
        if self.basic_method.get() == "Степенное (гамма)":
            self.apply_correction()
            self.update_function_visualization()

    def update_threshold(self, value):
        """Обновление значения порога для кусочно-линейного преобразования"""
        self.threshold_value.config(text=f"{float(value):.0f}")
        if self.basic_method.get() == "Кусочно-линейное":
            self.apply_correction()
            self.update_function_visualization()

    def update_slope_low(self, value):
        """Обновление значения наклона для темных пикселей"""
        self.slope_low_value.config(text=f"{float(value):.1f}")
        if self.basic_method.get() == "Кусочно-линейное":
            self.apply_correction()
            self.update_function_visualization()

    def update_slope_high(self, value):
        """Обновление значения наклона для светлых пикселей"""
        self.slope_high_value.config(text=f"{float(value):.1f}")
        if self.basic_method.get() == "Кусочно-линейное":
            self.apply_correction()
            self.update_function_visualization()

    def update_mean(self, value):
        """Обновление значения среднего для гауссова распределения"""
        self.mean_value.config(text=f"{float(value):.0f}")
        if (
            self.hist_method.get() == "Приведение к заданной функции"
            and self.dist_method.get() == "Гауссова"
        ):
            self.apply_correction()
            self.update_function_visualization()

    def update_std(self, value):
        """Обновление значения стандартного отклонения для гауссова распределения"""
        self.std_value.config(text=f"{float(value):.0f}")
        if (
            self.hist_method.get() == "Приведение к заданной функции"
            and self.dist_method.get() == "Гауссова"
        ):
            self.apply_correction()
            self.update_function_visualization()

    def update_lambda(self, value):
        """Обновление значения λ для экспоненциального распределения"""
        self.lambda_value.config(text=f"{float(value):.2f}")
        if (
            self.hist_method.get() == "Приведение к заданной функции"
            and self.dist_method.get() == "Экспоненциальная"
        ):
            self.apply_correction()
            self.update_function_visualization()

    def apply_correction(self, event=None):
        """Применение выбранных методов коррекции"""
        if not self.original_image:
            return

        # Начинаем с оригинального изображения
        img = self.original_image

        # Применяем базовое преобразование
        basic_method = self.basic_method.get()
        if basic_method == "Полярность":
            img = ImageCorrector.polarity(img)
        elif basic_method == "Логарифмическое":
            c = float(self.log_scale.get())
            img = ImageCorrector.logarithmic(img, c)
        elif basic_method == "Степенное (гамма)":
            gamma = float(self.gamma_scale.get())
            img = ImageCorrector.gamma(img, gamma)
        elif basic_method == "Кусочно-линейное":
            threshold = float(self.threshold_scale.get())
            slope_low = float(self.slope_low_scale.get())
            slope_high = float(self.slope_high_scale.get())
            img = ImageCorrector.piecewise_linear(img, threshold, slope_low, slope_high)

        # Применяем коррекцию гистограммы
        hist_method = self.hist_method.get()
        if hist_method == "Нормализация":
            img = ImageCorrector.normalize(img)
        elif hist_method == "Эквализация":
            img = ImageCorrector.equalize(img)
        elif hist_method == "Приведение к заданной функции":
            if self.dist_method.get() == "Гауссова":
                mean = float(self.mean_scale.get())
                std = float(self.std_scale.get())
                img = ImageCorrector.gaussian_mapping(img, mean, std)
            else:  # Экспоненциальная
                lambda_param = float(self.lambda_scale.get())
                img = ImageCorrector.exponential_mapping(img, lambda_param)

        self.current_image = img
        self.display_image(img)
        self.plot_histogram(img)
        self.analyze_image(img)

    def show_image_and_histogram(self, event):
        """Отображаем выбранное изображение и его гистограмму"""
        selection = self.image_list.curselection()
        if not selection:
            return

        selected_image = self.image_list.get(selection[0])
        image_path = os.path.join(self.folder_path, selected_image)

        try:
            self.original_image = Image.open(image_path)
            self.current_image = self.original_image
            self.apply_correction(None)
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")

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

    def display_image(self, img):
        """Отображаем изображение с фиксированными размерами"""
        # Фиксированные размеры для отображения
        display_width = 600
        display_height = 400

        # Вычисляем новые размеры с сохранением пропорций
        img_ratio = img.width / img.height
        display_ratio = display_width / display_height

        if display_ratio > img_ratio:
            new_height = display_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = display_width
            new_height = int(new_width / img_ratio)

        # Изменяем размер изображения
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Создаем новое изображение с черным фоном нужного размера
        background = Image.new("L", (display_width, display_height), 0)

        # Вычисляем позицию для центрирования изображения
        x = (display_width - new_width) // 2
        y = (display_height - new_height) // 2

        # Вставляем изображение в центр фона
        background.paste(img, (x, y))

        # Конвертируем в формат для отображения
        photo = ImageTk.PhotoImage(background)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def plot_histogram(self, img):
        """Строим гистограмму для черно-белого изображения"""
        img_array = np.array(img)
        self.ax.clear()

        # Определяем, нужно ли показывать две гистограммы
        show_original = self.basic_method.get() == "Кусочно-линейное" or (
            self.hist_method.get() == "Приведение к заданной функции"
            and self.dist_method.get() == "Гауссова"
        )

        if show_original and self.original_image is not None:
            # Строим две гистограммы
            original_array = np.array(self.original_image)
            self.ax.hist(
                original_array.ravel(),
                bins=256,
                range=(0, 256),
                color="gray",
                alpha=0.5,
                label="До коррекции",
            )
            self.ax.hist(
                img_array.ravel(),
                bins=256,
                range=(0, 256),
                color="black",
                alpha=0.7,
                label="После коррекции",
            )
            self.ax.legend()
        else:
            # Строим одну гистограмму
            self.ax.hist(
                img_array.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7
            )

        self.ax.set_title("Гистограмма")
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

        analysis += f"\n▌ Распределение тонов:\n"
        analysis += f"• Тени (0-{dark_threshold}): {dark_pixels/total_pixels:.1%}\n"
        analysis += f"• Средние тона: {mid_pixels/total_pixels:.1%}\n"
        analysis += (
            f"• Света ({light_threshold}-255): {light_pixels/total_pixels:.1%}\n"
        )

        # 3. Поиск проблем и рекомендации
        problems = []
        recommendations = []

        # Проверка на переэкспонирование (пересветы)
        if light_pixels > 0.3 * total_pixels:
            problems.append("Переэкспонирование: более 30% пикселей в ярких областях")
            recommendations.append(
                "• Рекомендуется: Логарифмическая коррекция (для восстановления деталей в светах)"
            )

        # Проверка на недоэкспонирование (недосвет)
        if dark_pixels > 0.3 * total_pixels:
            problems.append("Недоэкспонирование: более 30% пикселей в тёмных областях")
            recommendations.append(
                "• Рекомендуется: Гамма-коррекция (gamma < 1.0 для осветления теней)"
            )

        # Проверка на низкий контраст
        dynamic_range = np.max(img_array) - np.min(img_array)
        if dynamic_range < 100:
            problems.append(f"Низкий контраст (диапазон всего {dynamic_range} из 255)")
            recommendations.append(
                "• Рекомендуется: Линейное растяжение (для увеличения контраста)"
            )

        # Проверка на постеризацию
        unique_values = len(np.unique(img_array))
        if unique_values < 100:
            problems.append(f"Постеризация: только {unique_values} уникальных оттенков")
            recommendations.append(
                "• Рекомендуется: S-образная коррекция (для сглаживания переходов)"
            )

        # 4. Формирование итогового вывода
        if problems:
            analysis += "\n▌ ПРОБЛЕМЫ ОБНАРУЖЕНЫ:\n"
            for i, problem in enumerate(problems, 1):
                analysis += f"{i}. {problem}\n"

            # Добавляем рекомендации
            analysis += "\n▌ Рекомендуемые методы коррекции:\n"
            for rec in recommendations:
                analysis += f"{rec}\n"
        else:
            analysis += "\n▌ Качество изображения в норме\n"
            analysis += "• Хороший баланс светов и теней\n"
            analysis += "• Достаточный динамический диапазон\n"
            analysis += (
                "• Рекомендуется: Без коррекции (изображение не требует обработки)"
            )

        self.analysis_text.delete(1.0, END)
        self.analysis_text.insert(END, analysis)

    def change_folder(self):
        """Меняем папку с изображениями"""
        folder = filedialog.askdirectory(initialdir=self.folder_path)
        if folder:
            self.folder_path = folder
            self.load_images()

    def on_basic_method_change(self, event=None):
        """Обработчик изменения базового метода преобразования"""
        self.update_correction_ui()
        self.apply_correction()
        self.update_function_visualization()

    def on_hist_method_change(self, event=None):
        """Обработчик изменения метода коррекции гистограммы"""
        self.update_correction_ui()
        self.apply_correction()
        self.update_function_visualization()

    def on_dist_method_change(self, event=None):
        """Обработчик изменения метода распределения"""
        self.update_correction_ui()
        self.apply_correction()
        self.update_function_visualization()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x800")
    app = ImageViewer(root)
    root.mainloop()
