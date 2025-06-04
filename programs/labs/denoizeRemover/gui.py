import tkinter as tk
import cv2
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from image_processor import ImageProcessor


class ImageDenoisingApp:
    # Filter names in Russian
    filter_names = {
        "median": "Медианный",
        "gaussian": "Гауссовский",
        "bilateral": "Билатеральный",
        "wiener": "Винеровский",
    }

    # Noise type translations
    noise_types = {
        "Gaussian": "Гауссовский",
        "Salt and Pepper": "Соль и перец",
        "Poisson": "Пуассоновский",
        "Speckle": "Спекл",
        "Unknown": "Неизвестный",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Приложение для удаления шума с изображений")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

        # Initialize image processor
        self.processor = ImageProcessor()
        self.current_file = None

        # Updated default parameters for better results
        self.filter_params = {
            "median": {"kernel_size": 3},  # Smaller kernel for less blur
            "gaussian": {
                "kernel_size": (3, 3),
                "sigma": 0.8,
            },  # Smaller kernel, lower sigma
            "bilateral": {
                "d": 5,
                "sigma_color": 75,
                "sigma_space": 75,
            },  # Adjusted for better detail preservation
            "wiener": {
                "kernel_size": 3,
                "noise": 0.01,
            },  # Smaller kernel, lower noise estimate
        }

        # Parameter names in Russian
        self.param_names = {
            "kernel_size": "Размер ядра",
            "sigma": "Сигма",
            "d": "Диаметр",
            "sigma_color": "Сигма цвета",
            "sigma_space": "Сигма пространства",
            "noise": "Шум",
        }

        # Store last used parameters for each filter
        self.last_used_params = self.filter_params.copy()

        # Store tk variables for parameter widgets
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

        # Initialize variables
        self.current_image = None
        self.processed_image = None
        self.original_photo = None
        self.processed_photo = None
        self.histogram_canvas = None
        self.psnr_canvas = None

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

        # Bottom left - Noise distribution
        self.noise_frame = ttk.LabelFrame(self.workspace, text="Анализ шума")
        self.noise_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.noise_frame.grid_propagate(False)

        # Bottom right - PSNR chart
        self.psnr_frame = ttk.LabelFrame(self.workspace, text="Метрики качества")
        self.psnr_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.psnr_frame.grid_propagate(False)

        # Info label for noise type
        self.noise_info = ttk.Label(
            self.noise_frame, text="Тип шума: Не проанализирован"
        )
        self.noise_info.pack(pady=5)

    def create_control_panel(self):
        """Create control panel with filter options"""
        self.control_frame = ttk.LabelFrame(self.workspace, text="Управление фильтрами")
        self.control_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Filter selection
        self.filter_var = tk.StringVar(value="median")
        filters = ["median", "gaussian", "bilateral", "wiener"]

        for i, filter_name in enumerate(filters):
            ttk.Radiobutton(
                self.control_frame,
                text=self.filter_names[filter_name],
                value=filter_name,
                variable=self.filter_var,
                command=self.update_parameter_controls,
            ).grid(row=0, column=i, padx=5, pady=5)

        # Parameter controls frame
        self.param_frame = ttk.Frame(self.control_frame)
        self.param_frame.grid(row=1, column=0, columnspan=len(filters) + 1, sticky="ew")

        # Apply button
        ttk.Button(
            self.control_frame, text="Применить фильтр", command=self.apply_filter
        ).grid(row=0, column=len(filters), padx=5, pady=5)

        # Save button
        ttk.Button(
            self.control_frame, text="Сохранить результат", command=self.save_result
        ).grid(row=0, column=len(filters) + 1, padx=5, pady=5)

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
            elif param in ["sigma", "noise"]:
                value = float(value)
            elif param in ["sigma_color", "sigma_space"]:
                value = int(value)

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
        """Load and display selected image"""
        try:
            # Clear previous resources
            if hasattr(self, "histogram_canvas") and self.histogram_canvas:
                self.histogram_canvas.get_tk_widget().destroy()
            if hasattr(self, "psnr_canvas") and self.psnr_canvas:
                self.psnr_canvas.get_tk_widget().destroy()
            plt.close("all")

            self.current_file = file_path
            self.current_image = self.processor.load_image(file_path)

            # Clear previous processed image
            self.processed_image = None
            self.clear_processed_view()

            # Display original image
            self.display_image(self.current_image, self.original_canvas)

            # Analyze noise
            noise_info = self.processor.analyze_noise(self.current_image)
            self.noise_info.config(text=f"Тип шума: {noise_info}")

            # Generate histogram
            self.display_histogram()

        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить изображение: {str(e)}"
            )
            # Clear the tree selection
            self.tree.selection_remove(self.tree.selection())

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

    def apply_filter(self):
        """Apply selected filter to the image"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Изображение не загружено")
            return

        filter_type = self.filter_var.get()

        # Use last_used_params instead of filter_params
        params = self.last_used_params[filter_type].copy()

        try:
            # Validate parameters before applying
            if filter_type == "median":
                if params["kernel_size"] % 2 == 0:
                    params["kernel_size"] += 1  # Ensure odd kernel size
            elif filter_type == "gaussian":
                if isinstance(params["kernel_size"], tuple):
                    w, h = params["kernel_size"]
                    if w % 2 == 0:
                        w += 1
                    if h % 2 == 0:
                        h += 1
                    params["kernel_size"] = (w, h)
            elif filter_type == "bilateral":
                if params["d"] % 2 == 0:
                    params["d"] += 1  # Ensure odd diameter
            elif filter_type == "wiener":
                if params["kernel_size"] % 2 == 0:
                    params["kernel_size"] += 1  # Ensure odd kernel size

            # Apply filter
            self.processed_image = self.processor.apply_filter(
                self.current_image, filter_type, **params
            )

            # Display processed image
            self.display_image(self.processed_image, self.processed_canvas)

            # Calculate and display PSNR
            psnr = self.processor.calculate_psnr(
                self.current_image, self.processed_image
            )
            if psnr is not None:
                self.display_psnr(psnr)
            else:
                messagebox.showwarning("Предупреждение", "Не удалось рассчитать PSNR")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось применить фильтр: {str(e)}")

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
        noise_info = self.noise_types.get(
            self.processor.noise_type, self.processor.noise_type
        )
        filter_type = self.filter_names[self.filter_var.get()]
        ax.text(
            0.5,
            -0.3,
            f"Шум: {noise_info}\nФильтр: {filter_type}",
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
