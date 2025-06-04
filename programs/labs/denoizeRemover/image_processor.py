import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.noise_type = "Не проанализирован"
        self.noise_category = None
        self.psnr_value = None
        self.histogram_fig = None
        self.psnr_fig = None

    def load_image(self, file_path):
        """Load image from file"""
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            raise ValueError("Не удалось загрузить изображение")
        return self.original_image

    def analyze_noise(self, image):
        """Analyze the type of noise in the image"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Normalize image to 0-1 range
            gray = gray.astype(np.float32) / 255.0

            # Calculate noise statistics
            denoised = cv2.fastNlMeansDenoising((gray * 255).astype(np.uint8))
            denoised = denoised.astype(np.float32) / 255.0
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

            # Calculate noise spatial correlation
            noise_normalized = (noise - noise.min()) / (
                noise.max() - noise.min() + 1e-10
            )
            noise_normalized = (noise_normalized * 255).astype(np.uint8)
            corr = cv2.matchTemplate(
                noise_normalized, noise_normalized, cv2.TM_CCORR_NORMED
            )
            correlation_peak = np.max(corr)
            correlation_std = np.std(corr)

            # Determine noise category
            if energy_ratio > 0.5 and correlation_peak > 2 * correlation_std:
                self.noise_category = "детерминированный"
            else:
                self.noise_category = "случайный"

            # Analyze specific noise type based on statistics
            if self.noise_category == "случайный":
                if noise_std < 0.02:  # ~5/255
                    self.noise_type = "Низкий случайный шум"
                elif noise_std < 0.08:  # ~20/255
                    self.noise_type = "Гауссовский шум"
                else:
                    # Check for salt and pepper
                    extreme_pixels = np.sum((gray == 0) | (gray == 1))
                    if extreme_pixels / gray.size > 0.01:
                        self.noise_type = "Шум типа соль и перец"
                    else:
                        self.noise_type = "Сильный гауссовский шум"
            else:  # deterministic noise
                # Check for periodic patterns
                if energy_ratio > 0.7:
                    self.noise_type = "Периодический шум"
                # Check for structured patterns
                elif correlation_peak > 3 * correlation_std:
                    self.noise_type = "Структурированный шум"
                else:
                    self.noise_type = "Сложный детерминированный шум"

            return f"{self.noise_type} ({self.noise_category})"

        except Exception as e:
            print(f"Ошибка анализа шума: {e}")
            self.noise_type = "Ошибка анализа"
            return self.noise_type

    def apply_filter(self, image, filter_type, **params):
        """Apply selected filter to the image"""
        try:
            if len(image.shape) == 3:
                # For color images, process in YCrCb color space
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                y_channel = ycrcb[:, :, 0]

                # Apply filter to Y channel
                if filter_type == "wiener":
                    kernel_size = params.get("kernel_size", 3)
                    noise = params.get("noise", 0.01)
                    filtered_y = signal.wiener(
                        y_channel.astype(np.float32), (kernel_size, kernel_size), noise
                    )
                    filtered_y = np.clip(filtered_y, 0, 255).astype(np.uint8)
                elif filter_type == "median":
                    kernel_size = params.get("kernel_size", 3)
                    filtered_y = cv2.medianBlur(y_channel, kernel_size)
                elif filter_type == "gaussian":
                    kernel_size = params.get("kernel_size", (3, 3))
                    sigma = params.get("sigma", 0.8)
                    filtered_y = cv2.GaussianBlur(y_channel, kernel_size, sigma)
                elif filter_type == "bilateral":
                    d = params.get("d", 5)
                    sigma_color = params.get("sigma_color", 75)
                    sigma_space = params.get("sigma_space", 75)
                    filtered_y = cv2.bilateralFilter(
                        y_channel, d, sigma_color, sigma_space
                    )

                # Replace Y channel with filtered version
                ycrcb[:, :, 0] = filtered_y
                # Convert back to BGR
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                # For grayscale images
                if filter_type == "wiener":
                    kernel_size = params.get("kernel_size", 3)
                    noise = params.get("noise", 0.01)
                    filtered = signal.wiener(
                        image.astype(np.float32), (kernel_size, kernel_size), noise
                    )
                    return np.clip(filtered, 0, 255).astype(np.uint8)
                elif filter_type == "median":
                    kernel_size = params.get("kernel_size", 3)
                    return cv2.medianBlur(image, kernel_size)
                elif filter_type == "gaussian":
                    kernel_size = params.get("kernel_size", (3, 3))
                    sigma = params.get("sigma", 0.8)
                    return cv2.GaussianBlur(image, kernel_size, sigma)
                elif filter_type == "bilateral":
                    d = params.get("d", 5)
                    sigma_color = params.get("sigma_color", 75)
                    sigma_space = params.get("sigma_space", 75)
                    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

            return image

        except Exception as e:
            print(f"Error applying filter: {e}")
            return image

    def calculate_psnr(self, original, processed):
        """Calculate PSNR between original and processed images"""
        try:
            # Ensure images have the same size and type
            if original.shape != processed.shape:
                processed = cv2.resize(
                    processed, (original.shape[1], original.shape[0])
                )

            # Convert to float64 for precision
            original = original.astype(np.float64)
            processed = processed.astype(np.float64)

            # If color images, convert to Y channel of YCrCb
            if len(original.shape) == 3:
                # Ensure images are in BGR format before conversion
                if original.dtype != np.uint8:
                    original = np.clip(original, 0, 255).astype(np.uint8)
                if processed.dtype != np.uint8:
                    processed = np.clip(processed, 0, 255).astype(np.uint8)

                original_y = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                processed_y = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)[:, :, 0]

                original = original_y
                processed = processed_y

            # Calculate MSE
            mse = np.mean((original - processed) ** 2)
            if mse == 0:
                return float("inf")

            # Calculate PSNR
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
            self.psnr_value = psnr
            return psnr

        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return None

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

            self.histogram_fig = plt.figure(figsize=(6, 4), tight_layout=True)
            ax = self.histogram_fig.add_subplot(111)
            ax.hist(gray.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.7)
            ax.set_title("Гистограмма интенсивности пикселей")
            ax.set_xlabel("Значение пикселя")
            ax.set_ylabel("Частота")
            ax.grid(True, linestyle="--", alpha=0.5)
            return self.histogram_fig
        except Exception as e:
            print(f"Ошибка генерации гистограммы: {e}")
            return None
