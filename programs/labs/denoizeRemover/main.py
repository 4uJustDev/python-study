import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from pathlib import Path
import os


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.noise_type = "Not analyzed"
        self.noise_category = None
        self.psnr_value = None
        self.histogram_fig = None
        self.psnr_fig = None

    def load_image(self, file_path):
        """Load image from file"""
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            raise ValueError("Failed to load image")
        return self.original_image

    def analyze_noise(self, image):
        """Analyze the type of noise in the image"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Calculate noise statistics
            denoised = cv2.fastNlMeansDenoising(gray)
            noise = gray - denoised
            noise_std = np.std(noise)

            # Calculate noise pattern metrics
            noise_fft = np.fft.fft2(noise)
            noise_fft_shift = np.fft.fftshift(noise_fft)
            magnitude_spectrum = np.abs(noise_fft_shift)

            # Calculate energy distribution in frequency domain
            total_energy = np.sum(magnitude_spectrum)
            center_energy = np.sum(
                magnitude_spectrum[
                    magnitude_spectrum.shape[0] // 2
                    - 10 : magnitude_spectrum.shape[0] // 2
                    + 10,
                    magnitude_spectrum.shape[1] // 2
                    - 10 : magnitude_spectrum.shape[1] // 2
                    + 10,
                ]
            )
            energy_ratio = center_energy / total_energy if total_energy > 0 else 0

            # Calculate noise spatial correlation using cv2.matchTemplate instead of correlate2d
            noise_normalized = noise / np.max(noise) if np.max(noise) > 0 else noise
            corr = cv2.matchTemplate(
                noise_normalized, noise_normalized, cv2.TM_CCORR_NORMED
            )
            correlation_peak = np.max(corr)
            correlation_std = np.std(corr)

            # Determine noise category
            if energy_ratio > 0.7 and correlation_peak > 3 * correlation_std:
                self.noise_category = "deterministic"
            else:
                self.noise_category = "random"

            # Analyze specific noise type based on statistics
            if self.noise_category == "random":
                if noise_std < 5:
                    self.noise_type = "Low random noise"
                elif noise_std < 20:
                    self.noise_type = "Gaussian noise"
                else:
                    # Check for salt and pepper
                    extreme_pixels = np.sum((gray == 0) | (gray == 255))
                    if extreme_pixels / gray.size > 0.01:
                        self.noise_type = "Salt and pepper noise"
                    else:
                        self.noise_type = "High Gaussian noise"
            else:  # deterministic noise
                # Check for periodic patterns
                if energy_ratio > 0.9:
                    self.noise_type = "Periodic noise"
                # Check for structured patterns
                elif correlation_peak > 5 * correlation_std:
                    self.noise_type = "Structured noise"
                else:
                    self.noise_type = "Complex deterministic noise"

            return f"{self.noise_type} ({self.noise_category})"

        except Exception as e:
            print(f"Error analyzing noise: {e}")
            self.noise_type = "Analysis failed"
            return self.noise_type

    def apply_filter(self, image, filter_type, **params):
        """Apply selected filter to the image"""
        try:
            if filter_type == "median":
                kernel_size = params.get("kernel_size", 3)
                return cv2.medianBlur(image, kernel_size)
            elif filter_type == "gaussian":
                kernel_size = params.get("kernel_size", (3, 3))
                sigma = params.get("sigma", 0)
                return cv2.GaussianBlur(image, kernel_size, sigma)
            elif filter_type == "bilateral":
                d = params.get("d", 9)
                sigma_color = params.get("sigma_color", 75)
                sigma_space = params.get("sigma_space", 75)
                return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            elif filter_type == "wiener":
                kernel_size = params.get("kernel_size", 3)
                noise = params.get("noise", 0.1)
                if len(image.shape) == 3:
                    # Convert to YCrCb color space
                    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                    # Apply wiener filter only to Y channel
                    y_channel = ycrcb[:, :, 0]
                    filtered_y = signal.wiener(
                        y_channel, (kernel_size, kernel_size), noise
                    )
                    # Replace Y channel with filtered version
                    ycrcb[:, :, 0] = np.clip(filtered_y, 0, 255).astype(np.uint8)
                    # Convert back to BGR
                    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    return signal.wiener(image, (kernel_size, kernel_size), noise)
            return image
        except Exception as e:
            print(f"Error applying filter: {e}")
            return image

    def calculate_psnr(self, original, processed):
        """Calculate PSNR between original and processed images"""
        try:
            # Ensure images have the same size
            if original.shape != processed.shape:
                processed = cv2.resize(
                    processed, (original.shape[1], original.shape[0])
                )

            # Convert to grayscale if needed
            if len(original.shape) == 3:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # Calculate MSE
            mse = np.mean(
                (original.astype(np.float64) - processed.astype(np.float64)) ** 2
            )
            if mse == 0:
                return float("inf")

            # Calculate PSNR
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            self.psnr_value = psnr
            return psnr
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return 0

    def generate_histogram(self, image):
        """Generate histogram for the image"""
        try:
            # Clear previous figure if exists
            if self.histogram_fig:
                plt.close(self.histogram_fig)

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            self.histogram_fig = plt.Figure(
                figsize=(4, 3), tight_layout=True
            )  # Add tight_layout
            ax = self.histogram_fig.add_subplot(111)
            ax.hist(gray.ravel(), bins=256, range=[0, 256])
            ax.set_title("Pixel Intensity Histogram")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            return self.histogram_fig
        except Exception as e:
            print(f"Error generating histogram: {e}")
            return None


class ImageDenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Denoising Application")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

        # Initialize image processor
        self.processor = ImageProcessor()
        self.current_file = None
        self.filter_params = {
            "median": {"kernel_size": 3},
            "gaussian": {"kernel_size": (3, 3), "sigma": 0},
            "bilateral": {"d": 9, "sigma_color": 75, "sigma_space": 75},
            "wiener": {"kernel_size": 3, "noise": 0.1},
        }

        # Store last used parameters for each filter
        self.last_used_params = self.filter_params.copy()

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
        self.tree.heading("#0", text="Filename")
        self.tree.column("#0", width=150)
        self.tree.column("fullpath", width=0, stretch=tk.NO)  # Hidden column

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Add directory selection button
        self.dir_button = ttk.Button(
            self.tree_frame, text="Select Directory", command=self.select_directory
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
        self.original_frame = ttk.LabelFrame(self.workspace, text="Original Image")
        self.original_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.original_frame.grid_propagate(False)

        # Add canvas for original image
        self.original_canvas = tk.Canvas(self.original_frame)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Top right - Processed image
        self.processed_frame = ttk.LabelFrame(self.workspace, text="Processed Image")
        self.processed_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.processed_frame.grid_propagate(False)

        # Add canvas for processed image
        self.processed_canvas = tk.Canvas(self.processed_frame)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom left - Noise distribution
        self.noise_frame = ttk.LabelFrame(self.workspace, text="Noise Analysis")
        self.noise_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.noise_frame.grid_propagate(False)

        # Bottom right - PSNR chart
        self.psnr_frame = ttk.LabelFrame(self.workspace, text="Quality Metrics")
        self.psnr_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.psnr_frame.grid_propagate(False)

        # Info label for noise type
        self.noise_info = ttk.Label(self.noise_frame, text="Noise type: Not analyzed")
        self.noise_info.pack(pady=5)

    def create_control_panel(self):
        """Create control panel with filter options"""
        self.control_frame = ttk.LabelFrame(self.workspace, text="Filter Controls")
        self.control_frame.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

        # Filter selection
        self.filter_var = tk.StringVar(value="median")
        filters = ["median", "gaussian", "bilateral", "wiener"]

        for i, filter_name in enumerate(filters):
            ttk.Radiobutton(
                self.control_frame,
                text=filter_name.capitalize(),
                value=filter_name,
                variable=self.filter_var,
                command=self.update_parameter_controls,
            ).grid(row=0, column=i, padx=5, pady=5)

        # Parameter controls frame
        self.param_frame = ttk.Frame(self.control_frame)
        self.param_frame.grid(row=1, column=0, columnspan=len(filters) + 1, sticky="ew")

        # Apply button
        ttk.Button(
            self.control_frame, text="Apply Filter", command=self.apply_filter
        ).grid(row=0, column=len(filters), padx=5, pady=5)

        # Save button
        ttk.Button(
            self.control_frame, text="Save Result", command=self.save_result
        ).grid(row=0, column=len(filters) + 1, padx=5, pady=5)

        # Initialize parameter controls
        self.update_parameter_controls()

    def update_parameter_controls(self):
        """Update parameter controls based on selected filter"""
        # Clear previous controls
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        filter_type = self.filter_var.get()
        params = self.last_used_params.get(filter_type, self.filter_params[filter_type])

        row = 0
        for param, value in params.items():
            ttk.Label(self.param_frame, text=f"{param}:").grid(
                row=row, column=0, padx=2, sticky="e"
            )

            if isinstance(value, int):
                spinbox = ttk.Spinbox(
                    self.param_frame,
                    from_=1,
                    to=50,
                    textvariable=tk.IntVar(value=value),
                    width=5,
                )
                spinbox.grid(row=row, column=1, padx=2, sticky="w")
                spinbox.bind(
                    "<FocusOut>",
                    lambda e, p=param: self.update_param_value(p, e.widget.get()),
                )
            elif isinstance(value, tuple):
                # For kernel size (width, height)
                ttk.Label(self.param_frame, text="width:").grid(
                    row=row, column=1, padx=2
                )
                width_spin = ttk.Spinbox(
                    self.param_frame,
                    from_=1,
                    to=50,
                    textvariable=tk.IntVar(value=value[0]),
                    width=3,
                )
                width_spin.grid(row=row, column=2, padx=2)
                width_spin.bind(
                    "<FocusOut>",
                    lambda e: self.update_kernel_size(e.widget.get(), None),
                )

                ttk.Label(self.param_frame, text="height:").grid(
                    row=row, column=3, padx=2
                )
                height_spin = ttk.Spinbox(
                    self.param_frame,
                    from_=1,
                    to=50,
                    textvariable=tk.IntVar(value=value[1]),
                    width=3,
                )
                height_spin.grid(row=row, column=4, padx=2)
                height_spin.bind(
                    "<FocusOut>",
                    lambda e: self.update_kernel_size(None, e.widget.get()),
                )

                row += 1

                # Sigma for Gaussian
                if filter_type == "gaussian":
                    ttk.Label(self.param_frame, text="sigma:").grid(
                        row=row, column=0, padx=2, sticky="e"
                    )
                    sigma_spin = ttk.Spinbox(
                        self.param_frame,
                        from_=0,
                        to=50,
                        textvariable=tk.DoubleVar(value=params.get("sigma", 0)),
                        width=5,
                        increment=0.1,
                    )
                    sigma_spin.grid(row=row, column=1, padx=2, sticky="w")
                    sigma_spin.bind(
                        "<FocusOut>",
                        lambda e: self.update_param_value("sigma", e.widget.get()),
                    )
            elif isinstance(value, float):
                spinbox = ttk.Spinbox(
                    self.param_frame,
                    from_=0,
                    to=1,
                    textvariable=tk.DoubleVar(value=value),
                    width=5,
                    increment=0.1,
                )
                spinbox.grid(row=row, column=1, padx=2, sticky="w")
                spinbox.bind(
                    "<FocusOut>",
                    lambda e, p=param: self.update_param_value(p, e.widget.get()),
                )

            row += 1

    def update_param_value(self, param, value):
        """Update parameter value in the dictionary"""
        filter_type = self.filter_var.get()
        try:
            if param in ["kernel_size", "d"]:
                value = int(value)
            elif param in ["sigma", "sigma_color", "sigma_space", "noise"]:
                value = float(value)

            # Update both current and last used parameters
            self.filter_params[filter_type][param] = value
            if filter_type not in self.last_used_params:
                self.last_used_params[filter_type] = {}
            self.last_used_params[filter_type][param] = value
        except ValueError:
            pass

    def update_kernel_size(self, width, height):
        """Update kernel size (special case for tuple parameter)"""
        filter_type = self.filter_var.get()
        current = self.filter_params[filter_type]["kernel_size"]

        if width is not None:
            new_width = int(width)
            # Kernel size must be odd for most filters
            if new_width % 2 == 0:
                new_width += 1
            self.filter_params[filter_type]["kernel_size"] = (new_width, current[1])
        if height is not None:
            new_height = int(height)
            if new_height % 2 == 0:
                new_height += 1
            self.filter_params[filter_type]["kernel_size"] = (current[0], new_height)

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
            self.tree.insert("", "end", text="No images found", values=("",))
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
            self.noise_info.config(text=f"Noise type: {noise_info}")

            # Generate histogram
            self.display_histogram()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
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
            messagebox.showwarning("Warning", "No image loaded")
            return

        filter_type = self.filter_var.get()
        params = self.filter_params[filter_type]

        try:
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
            self.display_psnr(psnr)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {str(e)}")

    def display_psnr(self, psnr):
        """Display PSNR metric and comparison"""
        # Clear previous PSNR display
        for widget in self.psnr_frame.winfo_children():
            widget.destroy()

        # Create figure with tight layout to prevent clipping
        fig = plt.Figure(figsize=(4, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        # Create bar chart
        filters = ["Original", self.filter_var.get().capitalize()]
        psnr_values = [0, psnr]  # Original has 0 PSNR for comparison

        bars = ax.bar(filters, psnr_values)
        ax.set_title("PSNR Comparison")
        ax.set_ylabel("PSNR (dB)")

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
        noise_info = self.processor.noise_type
        filter_type = self.filter_var.get()
        ax.text(
            0.5,
            -0.3,
            f"Noise: {noise_info}\nFilter: {filter_type}",
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
            messagebox.showwarning("Warning", "No processed image to save")
            return

        if not self.current_file:
            messagebox.showwarning("Warning", "No original file reference")
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
                    "Success", f"Image saved successfully to:\n{file_path}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDenoisingApp(root)
    root.mainloop()
