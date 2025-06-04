import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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

            # Calculate noise spatial correlation using cv2.matchTemplate
            noise_normalized = noise / np.max(noise) if np.max(noise) > 0 else noise
            noise_normalized = (noise_normalized * 255).astype(np.uint8)
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
            # Convert to float32 for better precision
            image_float = image.astype(np.float32)

            if filter_type == "median":
                kernel_size = params.get("kernel_size", 5)
                if len(image.shape) == 3:
                    # Apply median filter to each channel separately
                    filtered = np.zeros_like(image)
                    for i in range(3):
                        filtered[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
                    return filtered
                else:
                    return cv2.medianBlur(image, kernel_size)

            elif filter_type == "gaussian":
                kernel_size = params.get("kernel_size", (5, 5))
                sigma = params.get("sigma", 1.5)
                if len(image.shape) == 3:
                    # Apply Gaussian filter to each channel separately
                    filtered = np.zeros_like(image)
                    for i in range(3):
                        filtered[:, :, i] = cv2.GaussianBlur(
                            image[:, :, i], kernel_size, sigma
                        )
                    return filtered
                else:
                    return cv2.GaussianBlur(image, kernel_size, sigma)

            elif filter_type == "bilateral":
                d = params.get("d", 9)
                sigma_color = params.get("sigma_color", 50)
                sigma_space = params.get("sigma_space", 50)
                if len(image.shape) == 3:
                    # Apply bilateral filter to each channel separately
                    filtered = np.zeros_like(image)
                    for i in range(3):
                        filtered[:, :, i] = cv2.bilateralFilter(
                            image[:, :, i], d, sigma_color, sigma_space
                        )
                    return filtered
                else:
                    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

            elif filter_type == "wiener":
                kernel_size = params.get("kernel_size", 5)
                noise = params.get("noise", 0.05)
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
                    filtered = signal.wiener(image, (kernel_size, kernel_size), noise)
                    return np.clip(filtered, 0, 255).astype(np.uint8)

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

            self.histogram_fig = plt.Figure(figsize=(4, 3), tight_layout=True)
            ax = self.histogram_fig.add_subplot(111)
            ax.hist(gray.ravel(), bins=256, range=[0, 256])
            ax.set_title("Pixel Intensity Histogram")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            return self.histogram_fig
        except Exception as e:
            print(f"Error generating histogram: {e}")
            return None
