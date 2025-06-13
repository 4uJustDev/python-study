import tkinter as tk
import cv2
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from image_processor import ImageProcessor
import numpy as np


class ImageDenoisingApp:
    # Filter names in Russian
    filter_names = {
        "median": "Медианный",
        "gaussian": "Гауссовский",
        "ihpf": "Идеальный высокочастотный",
        "ghpf": "Гауссов высокочастотный",
        "bandpass": "Полосовой",
    }

    # Noise type descriptions
    noise_descriptions = {
        "Гауссов шум": "Равномерный зернистый шум, характерный для цифровых камер",
        "Импульсный шум (соль/перец)": "Точечные выбросы белого и черного цвета",
        "Муар (периодический шум)": "Периодические узоры, возникающие при наложении регулярных структур",
        "JPEG артефакты": "Блоки 8x8 пикселей, характерные для JPEG-сжатия",
        "Квантование": "Ступенчатость градиентов из-за ограниченной глубины цвета",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Приложение для удаления шума с изображений")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

        # Initialize variables
        self.current_image = None
        self.processed_image = None
        self.original_photo = None
        self.processed_photo = None
        self.histogram_canvas = None
        self.psnr_canvas = None
        self.fft_canvas = None
        self.current_file = None

        # Список активных фильтров
        self.active_filters = []

        # Initialize image processor
        self.processor = ImageProcessor()

        # Updated default parameters
        self.filter_params = {
            "median": {"kernel_size": 3},
            "gaussian": {"kernel_size": (3, 3), "sigma": 0.8},
            "ihpf": {"d0": 30},
            "ghpf": {"d0": 30},
            "bandpass": {"d0": 30, "w": 10},
        }

        # Parameter names in Russian
        self.param_names = {
            "kernel_size": "Размер ядра",
            "sigma": "Сигма",
            "d0": "Радиус отсечки",
            "w": "Ширина полосы",
        }

        # Store last used parameters
        self.last_used_params = self.filter_params.copy()
        self.param_vars = {ftype: {} for ftype in self.filter_params}

        # Create main container
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Create file tree
        self.create_file_tree()

        # Create main workspace
        self.create_workspace()

        # Create control panel
        self.create_control_panel()

        # Load initial directory
        self.load_initial_directory()

        # Initialize matplotlib figure cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            # Update image displays
            if self.current_image is not None:
                self.display_image(self.current_image, self.original_canvas)
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.processed_canvas)

            # Update histogram and PSNR charts
            self.update_charts()

    def update_charts(self):
        """Update histogram and PSNR charts"""
        if self.current_image is not None:
            self.display_histogram()
        if self.processed_image is not None and hasattr(self.processor, "psnr_value"):
            self.display_psnr(self.processor.psnr_value)

    def on_closing(self):
        """Handle window closing"""
        # Clean up matplotlib resources
        if hasattr(self, "histogram_canvas") and self.histogram_canvas:
            self.histogram_canvas.get_tk_widget().destroy()
        if hasattr(self, "psnr_canvas") and self.psnr_canvas:
            self.psnr_canvas.get_tk_widget().destroy()
        if hasattr(self, "fft_canvas") and self.fft_canvas:
            self.fft_canvas.get_tk_widget().destroy()
        plt.close("all")
        self.root.destroy()

    def create_file_tree(self):
        """Create file system tree view"""
        self.tree_frame = ttk.Frame(self.main_container, width=200)
        self.main_container.add(self.tree_frame, weight=1)

        # Add scrollbar
        self.tree_scroll = ttk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(self.tree_frame, yscrollcommand=self.tree_scroll.set)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree_scroll.config(command=self.tree.yview)

        # Configure tree columns
        self.tree["columns"] = "fullpath"
        self.tree.heading("#0", text="Имя файла")
        self.tree.column("#0", width=150)
        self.tree.column("fullpath", width=0, stretch=tk.NO)  # Hidden column

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Add directory selection button
        self.dir_button = ttk.Button(
            self.tree_frame, text="Выбрать папку", command=self.select_directory
        )
        self.dir_button.pack(fill=tk.X, padx=5, pady=5)

        # Add FFT spectrum frame
        self.fft_frame = ttk.LabelFrame(self.tree_frame, text="FFT спектр")
        self.fft_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create initial empty FFT spectrum
        self.fft_canvas = None
        self.show_fft_spectrum()

    def create_workspace(self):
        """Create main workspace with image displays and charts"""
        self.workspace = ttk.Frame(self.main_container)
        self.main_container.add(self.workspace, weight=4)

        # Create quadrants
        self.create_quadrants()

    def create_quadrants(self):
        """Create four quadrants for images and charts"""
        # Configure grid weights
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(1, weight=1)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=1)

        # Top left - Original image
        self.original_frame = ttk.LabelFrame(
            self.workspace, text="Исходное изображение"
        )
        self.original_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.original_frame.grid_propagate(False)

        # Add canvas for original image
        self.original_canvas = tk.Canvas(self.original_frame)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Top right - Processed image
        self.processed_frame = ttk.LabelFrame(
            self.workspace, text="Обработанное изображение"
        )
        self.processed_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.processed_frame.grid_propagate(False)

        # Add canvas for processed image
        self.processed_canvas = tk.Canvas(self.processed_frame)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom left - Noise analysis
        self.noise_frame = ttk.LabelFrame(self.workspace, text="Анализ шума")
        self.noise_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.noise_frame.grid_propagate(False)

        # Add scrollable text widget for noise info
        self.noise_info = tk.Text(
            self.noise_frame,
            wrap=tk.WORD,
            width=40,
            height=10,
            font=("TkDefaultFont", 10),
        )
        self.noise_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom right - PSNR chart
        self.psnr_frame = ttk.LabelFrame(self.workspace, text="Метрики качества")
        self.psnr_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.psnr_frame.grid_propagate(False)

    def create_control_panel(self):
        """Create control panel with filter options"""
        # Создаем основной контейнер для панели управления
        control_container = ttk.Frame(self.workspace)
        control_container.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Панель управления фильтрами
        self.control_frame = ttk.LabelFrame(
            control_container, text="Управление фильтрами"
        )
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Filter selection
        self.filter_var = tk.StringVar(value="median")
        filters = list(self.filter_names.keys())

        # Создаем фрейм для выбора фильтра и кнопки добавления
        filter_select_frame = ttk.Frame(self.control_frame)
        filter_select_frame.pack(fill=tk.X, padx=5, pady=5)

        for i, filter_name in enumerate(filters):
            ttk.Radiobutton(
                filter_select_frame,
                text=self.filter_names[filter_name],
                value=filter_name,
                variable=self.filter_var,
                command=self.update_parameter_controls,
            ).grid(row=0, column=i, padx=5, pady=5)

        # Кнопка добавления фильтра
        ttk.Button(
            filter_select_frame, text="+", width=3, command=self.add_filter
        ).grid(row=0, column=len(filters), padx=5, pady=5)

        # Parameter controls frame
        self.param_frame = ttk.Frame(self.control_frame)
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Apply and Save buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame, text="Применить фильтры", command=self.apply_filters
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, text="Сохранить результат", command=self.save_result
        ).pack(side=tk.LEFT, padx=5)

        # Фрейм для списка активных фильтров
        self.active_filters_frame = ttk.LabelFrame(
            control_container, text="Активные фильтры"
        )
        self.active_filters_frame.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0)
        )

        # Initialize parameter controls
        self.update_parameter_controls()

    def update_parameter_controls(self):
        """Update parameter controls based on selected filter"""
        # Clear previous controls
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        filter_type = self.filter_var.get()
        params = self.last_used_params[filter_type].copy()

        # Prepare param_vars for this filter
        if filter_type not in self.param_vars:
            self.param_vars[filter_type] = {}

        row = 0
        for param, value in params.items():
            ttk.Label(
                self.param_frame, text=f"{self.param_names.get(param, param)}:"
            ).grid(row=row, column=0, padx=2, sticky="e")

            if isinstance(value, int):
                if param not in self.param_vars[filter_type]:
                    self.param_vars[filter_type][param] = tk.IntVar(value=value)
                else:
                    self.param_vars[filter_type][param].set(value)
                spinbox = ttk.Spinbox(
                    self.param_frame,
                    from_=1,
                    to=50,
                    textvariable=self.param_vars[filter_type][param],
                    width=5,
                )
                spinbox.grid(row=row, column=1, padx=2, sticky="w")
                spinbox.bind(
                    "<FocusOut>",
                    lambda e, p=param, f=filter_type: self.update_param_value(
                        p, e.widget.get(), f
                    ),
                )
            elif isinstance(value, tuple):
                # For kernel size (width, height)
                ttk.Label(self.param_frame, text="ширина:").grid(
                    row=row, column=1, padx=2
                )
                if param not in self.param_vars[filter_type]:
                    self.param_vars[filter_type][param] = [
                        tk.IntVar(value=value[0]),
                        tk.IntVar(value=value[1]),
                    ]
                else:
                    self.param_vars[filter_type][param][0].set(value[0])
                    self.param_vars[filter_type][param][1].set(value[1])
                width_spin = ttk.Spinbox(
                    self.param_frame,
                    from_=1,
                    to=50,
                    textvariable=self.param_vars[filter_type][param][0],
                    width=3,
                )
                width_spin.grid(row=row, column=2, padx=2)
                width_spin.bind(
                    "<FocusOut>",
                    lambda e, f=filter_type, p=param: self.update_kernel_size(
                        e.widget.get(), None, f, p
                    ),
                )

                ttk.Label(self.param_frame, text="высота:").grid(
                    row=row, column=3, padx=2
                )
                height_spin = ttk.Spinbox(
                    self.param_frame,
                    from_=1,
                    to=50,
                    textvariable=self.param_vars[filter_type][param][1],
                    width=3,
                )
                height_spin.grid(row=row, column=4, padx=2)
                height_spin.bind(
                    "<FocusOut>",
                    lambda e, f=filter_type, p=param: self.update_kernel_size(
                        None, e.widget.get(), f, p
                    ),
                )

                row += 1

                # Sigma for Gaussian
                if filter_type == "gaussian":
                    ttk.Label(self.param_frame, text="сигма:").grid(
                        row=row, column=0, padx=2, sticky="e"
                    )
                    if "sigma" not in self.param_vars[filter_type]:
                        self.param_vars[filter_type]["sigma"] = tk.DoubleVar(
                            value=params.get("sigma", 0)
                        )
                    else:
                        self.param_vars[filter_type]["sigma"].set(
                            params.get("sigma", 0)
                        )
                    sigma_spin = ttk.Spinbox(
                        self.param_frame,
                        from_=0,
                        to=50,
                        textvariable=self.param_vars[filter_type]["sigma"],
                        width=5,
                        increment=0.1,
                    )
                    sigma_spin.grid(row=row, column=1, padx=2, sticky="w")
                    sigma_spin.bind(
                        "<FocusOut>",
                        lambda e, f=filter_type: self.update_param_value(
                            "sigma", e.widget.get(), f
                        ),
                    )
            elif isinstance(value, float):
                if param not in self.param_vars[filter_type]:
                    self.param_vars[filter_type][param] = tk.DoubleVar(value=value)
                else:
                    self.param_vars[filter_type][param].set(value)
                spinbox = ttk.Spinbox(
                    self.param_frame,
                    from_=0,
                    to=1,
                    textvariable=self.param_vars[filter_type][param],
                    width=5,
                    increment=0.1,
                )
                spinbox.grid(row=row, column=1, padx=2, sticky="w")
                spinbox.bind(
                    "<FocusOut>",
                    lambda e, p=param, f=filter_type: self.update_param_value(
                        p, e.widget.get(), f
                    ),
                )

            row += 1

    def update_param_value(self, param, value, filter_type=None):
        """Update parameter value and store in last_used_params"""
        if filter_type is None:
            filter_type = self.filter_var.get()

        try:
            # Convert value to appropriate type
            if param == "kernel_size":
                if isinstance(value, str) and "x" in value:
                    width, height = map(int, value.split("x"))
                    value = (width, height)
                else:
                    value = int(value)
            elif param in ["sigma", "d0", "w"]:
                value = float(value)

            # Update the parameter in last_used_params
            if filter_type in self.last_used_params:
                self.last_used_params[filter_type][param] = value

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Недопустимое значение параметра: {str(e)}")

    def update_kernel_size(self, width, height, filter_type=None, param="kernel_size"):
        """Update kernel size (special case for tuple parameter)"""
        if filter_type is None:
            filter_type = self.filter_var.get()
        current = self.filter_params[filter_type][param]

        if width is not None:
            new_width = int(width)
            # Kernel size must be odd for most filters
            if new_width % 2 == 0:
                new_width += 1
            self.filter_params[filter_type][param] = (new_width, current[1])
            if filter_type in self.param_vars and param in self.param_vars[filter_type]:
                self.param_vars[filter_type][param][0].set(new_width)
        if height is not None:
            new_height = int(height)
            if new_height % 2 == 0:
                new_height += 1
            self.filter_params[filter_type][param] = (current[0], new_height)
            if filter_type in self.param_vars and param in self.param_vars[filter_type]:
                self.param_vars[filter_type][param][1].set(new_height)

    def load_initial_directory(self):
        """Load initial directory if photos folder exists"""
        photos_dir = Path("photos")
        if photos_dir.exists():
            self.load_directory(photos_dir)
        else:
            # Create empty photos directory if it doesn't exist
            try:
                photos_dir.mkdir()
            except OSError:
                pass

    def select_directory(self):
        """Let user select directory to load images from"""
        directory = filedialog.askdirectory()
        if directory:
            self.load_directory(Path(directory))

    def load_directory(self, directory):
        """Load images from the specified directory"""
        # Clear current tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add files to tree
        supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        image_files = []

        for file in sorted(directory.glob("*")):
            if file.suffix.lower() in supported_extensions:
                image_files.append(file)

        if not image_files:
            # Show message if no images found
            self.tree.insert("", "end", text="Изображения не найдены", values=("",))
            messagebox.showinfo(
                "Information",
                "No supported image files found in the selected directory.",
            )
            return

        for file in image_files:
            self.tree.insert("", "end", text=file.name, values=(str(file),))

        # Force tree update
        self.tree.update_idletasks()

    def on_select(self, event):
        """Handle image selection from tree"""
        selection = self.tree.selection()
        if selection:
            file_path = self.tree.item(selection[0])["values"][0]
            self.load_image(file_path)

    def load_image(self, file_path):
        """Загрузка изображения"""
        try:
            self.current_image = self.processor.load_image(file_path)
            self.current_file = file_path

            # Отображение исходного изображения
            self.display_image(self.current_image, self.original_canvas)

            # Анализ шума и обновление информации
            noise_type = self.processor.analyze_noise(self.current_image)
            self.update_noise_info(noise_type)

            # Обновление FFT спектра
            self.show_fft_spectrum()

            # Очистка обработанного изображения
            self.clear_processed_view()

            # Обновление гистограммы
            self.display_histogram()

        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить изображение: {str(e)}"
            )

    def display_image(self, image, canvas):
        """Display image in the specified canvas"""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Get canvas dimensions
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Resize image to fit canvas while maintaining aspect ratio
            img_width, img_height = pil_image.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_image)

            # Update canvas
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo
            )
            canvas.image = photo  # Keep reference

            # Store reference based on which canvas we're updating
            if canvas == self.original_canvas:
                self.original_photo = photo
            else:
                self.processed_photo = photo
        except Exception as e:
            print(f"Error displaying image: {e}")

    def display_histogram(self):
        """Display histogram of the current image"""
        if self.current_image is None:
            return

        # Clear previous histogram
        for widget in self.noise_frame.winfo_children():
            if widget != self.noise_info:
                widget.destroy()

        # Generate new histogram
        fig = self.processor.generate_histogram(self.current_image)
        if fig is None:
            return

        # Translate histogram labels
        for ax in fig.axes:
            ax.set_title("Распределение интенсивности")
            ax.set_xlabel("Интенсивность")
            ax.set_ylabel("Частота")
            # Translate tick labels
            ax.tick_params(axis="both", labelsize=8)
            # Format y-axis ticks to show percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.noise_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.histogram_canvas = canvas

        # Force update to prevent layout issues
        self.noise_frame.update_idletasks()

    def clear_processed_view(self):
        """Clear processed image and PSNR chart"""
        self.processed_canvas.delete("all")
        if hasattr(self.processed_canvas, "image"):
            del self.processed_canvas.image

        for widget in self.psnr_frame.winfo_children():
            widget.destroy()

    def show_fft_spectrum(self):
        """Показать FFT спектр изображения"""
        # Очистка предыдущего содержимого
        if self.fft_canvas:
            self.fft_canvas.get_tk_widget().destroy()

        # Создание фигуры matplotlib
        fig = plt.figure(figsize=(4, 4))

        if self.current_image is not None and self.processor.fft_spectrum is not None:
            # Отображение спектра в логарифмической шкале
            spectrum = np.log1p(self.processor.fft_spectrum)
            plt.imshow(spectrum, cmap="gray")
            plt.title("FFT спектр")
            plt.axis("off")  # Скрываем оси
        else:
            plt.text(
                0.5,
                0.5,
                "Загрузите изображение\nдля анализа спектра",
                ha="center",
                va="center",
            )
            plt.axis("off")

        # Добавление на холст
        self.fft_canvas = FigureCanvasTkAgg(fig, master=self.fft_frame)
        self.fft_canvas.draw()
        self.fft_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_noise_info(self, noise_type):
        """Обновление информации о типе шума"""
        metrics = self.processor.get_noise_metrics()

        # Формирование подробного описания
        description = f"Тип шума: {noise_type}\n\n"
        description += "Метрики:\n"
        description += f"• Стандартное отклонение: {metrics.get('image_std', 0):.3f}\n"
        description += (
            f"• Энтропия изображения: {metrics.get('image_entropy', 0):.3f}\n"
        )
        description += f"• Количество пиков в спектре: {metrics.get('fft_peaks', 0)}\n"

        # Добавление рекомендаций по фильтрации
        description += "\nРекомендуемые фильтры:\n"
        if "импульсный" in noise_type.lower():
            description += "• Медианный фильтр\n"
        elif "гауссов" in noise_type.lower():
            description += "• Гауссовский фильтр\n"
        elif "муар" in noise_type.lower():
            description += "• Полосовой фильтр\n"
        elif "jpeg" in noise_type.lower():
            description += "• Идеальный высокочастотный фильтр\n"

        # Обновление текста
        self.noise_info.delete(1.0, tk.END)
        self.noise_info.insert(tk.END, description)

    def add_filter(self):
        """Добавление нового фильтра в список активных"""
        filter_type = self.filter_var.get()
        filter_name = self.filter_names[filter_type]

        # Создаем фрейм для фильтра
        filter_frame = ttk.Frame(self.active_filters_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=2)

        # Добавляем метку с названием фильтра
        ttk.Label(filter_frame, text=filter_name).pack(side=tk.LEFT, padx=5)

        # Собираем текущие параметры фильтра
        params = {}
        for param_name, var in self.param_vars[filter_type].items():
            if isinstance(var, (tk.StringVar, tk.IntVar)):
                try:
                    value = int(var.get())
                    if param_name == "kernel_size" and filter_type == "gaussian":
                        value = (value, value)
                        value = tuple(v if v % 2 == 1 else v + 1 for v in value)
                    if param_name == "kernel_size" and filter_type == "median":
                        if value % 2 == 0:
                            value += 1
                        if value < 3:
                            value = 3
                    if param_name in ["d0", "w"] and value < 1:
                        value = 1
                    params[param_name] = value
                except Exception:
                    continue
            elif isinstance(var, tk.DoubleVar):
                value = float(var.get())
                if param_name == "sigma" and value < 0.01:
                    value = 0.01
                params[param_name] = value
            elif isinstance(var, list):
                try:
                    width = int(var[0].get())
                    height = int(var[1].get())
                    width = width if width % 2 == 1 else width + 1
                    height = height if height % 2 == 1 else height + 1
                    if width < 3:
                        width = 3
                    if height < 3:
                        height = 3
                    params[param_name] = (width, height)
                except Exception:
                    continue

        # Создаем строку с параметрами
        params_text = []
        for param_name, value in params.items():
            param_label = self.param_names.get(param_name, param_name)
            if isinstance(value, tuple):
                params_text.append(f"{param_label}: {value[0]}x{value[1]}")
            else:
                params_text.append(f"{param_label}: {value}")

        # Добавляем метку с параметрами
        params_label = ttk.Label(filter_frame, text=" | ".join(params_text))
        params_label.pack(side=tk.LEFT, padx=5)

        # Добавляем кнопку удаления
        ttk.Button(
            filter_frame,
            text="×",
            width=2,
            command=lambda f=filter_frame, ft=filter_type: self.remove_filter(f, ft),
        ).pack(side=tk.RIGHT, padx=5)

        # Сохраняем информацию о фильтре
        self.active_filters.append(
            {"frame": filter_frame, "type": filter_type, "params": params}
        )

    def remove_filter(self, filter_frame, filter_type):
        """Удаление фильтра из списка активных"""
        # Удаляем фрейм
        filter_frame.destroy()

        # Удаляем из списка активных фильтров
        self.active_filters = [
            f for f in self.active_filters if f["frame"] != filter_frame
        ]

    def apply_filters(self):
        """Применение всех активных фильтров последовательно"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        if not self.active_filters:
            messagebox.showwarning("Предупреждение", "Добавьте хотя бы один фильтр")
            return

        # Начинаем с исходного изображения
        result = self.current_image.copy()

        # Применяем каждый фильтр последовательно
        for filter_info in self.active_filters:
            filter_type = filter_info["type"]
            params = filter_info["params"]

            # Применяем фильтр
            result = self.processor.apply_filter(result, filter_type, **params)

        # Обновляем отображение
        self.processed_image = result
        self.display_image(self.processed_image, self.processed_canvas)

        # Расчет PSNR
        psnr = self.processor.calculate_psnr(self.current_image, self.processed_image)
        self.display_psnr(psnr)

    def display_psnr(self, psnr):
        """Display PSNR metric and comparison"""
        # Clear previous PSNR display
        for widget in self.psnr_frame.winfo_children():
            widget.destroy()

        # Create figure with tight layout to prevent clipping
        fig = plt.Figure(figsize=(4, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        # Create bar chart
        filters = ["Оригинал", self.filter_names[self.filter_var.get()]]
        psnr_values = [0, psnr]  # Original has 0 PSNR for comparison

        bars = ax.bar(filters, psnr_values)
        ax.set_title("Сравнение PSNR")
        ax.set_ylabel("PSNR (дБ)")
        ax.set_xlabel("Изображение")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        # Add text info
        noise_info = self.noise_descriptions.get(
            self.processor.noise_type, self.processor.noise_type
        )
        filter_type = self.filter_names[self.filter_var.get()]
        ax.text(
            0.5,
            -0.3,
            f"Шум: {noise_info}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.psnr_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.psnr_canvas = canvas

        # Force update to prevent layout issues
        self.psnr_frame.update_idletasks()

    def save_result(self):
        """Save processed image to file"""
        if self.processed_image is None:
            messagebox.showwarning(
                "Предупреждение", "Нет обработанного изображения для сохранения"
            )
            return

        if not self.current_file:
            messagebox.showwarning("Предупреждение", "Нет ссылки на исходный файл")
            return

        # Suggest a filename based on original
        original_path = Path(self.current_file)
        suggested_name = original_path.stem + "_processed" + original_path.suffix

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            initialfile=suggested_name,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg;*.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                # Convert BGR to RGB if needed before saving
                if len(self.processed_image.shape) == 3:
                    save_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                else:
                    save_image = self.processed_image

                # Save image
                cv2.imwrite(file_path, save_image)
                messagebox.showinfo(
                    "Успех", f"Изображение успешно сохранено в:\n{file_path}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Ошибка", f"Не удалось сохранить изображение: {str(e)}"
                )
