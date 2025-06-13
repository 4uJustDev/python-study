import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.noise_type = "Не проанализирован"
        self.noise_category = None
        self.psnr_value = None
        self.histogram_fig = None
        self.psnr_fig = None
        self.fft_spectrum = None
        self.noise_metrics = {}

    def load_image(self, file_path):
        """Load image from file"""
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            raise ValueError("Не удалось загрузить изображение")
        return self.original_image

    def analyze_noise(self, image):
        """Улучшенный анализ типа шума в изображении"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Нормализация изображения
            gray = gray.astype(np.float32) / 255.0

            # Анализ гистограммы
            hist = cv2.calcHist([gray], [0], None, [256], [0, 1])
            hist = hist.flatten() / hist.sum()

            # Анализ FFT
            f = fft2(gray)
            fshift = fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            self.fft_spectrum = magnitude_spectrum

            # Статистический анализ
            noise_metrics = self._calculate_noise_metrics(
                gray, hist, magnitude_spectrum
            )
            self.noise_metrics = noise_metrics

            # Определение типа шума
            noise_type = self._determine_noise_type(noise_metrics)
            return noise_type

        except Exception as e:
            print(f"Ошибка анализа шума: {e}")
            return "Ошибка анализа"

    def _calculate_noise_metrics(self, image, hist, fft_spectrum):
        """Расчет метрик для анализа шума"""
        metrics = {}

        # 1. Анализ гистограммы
        metrics["hist_std"] = np.std(hist)
        metrics["hist_entropy"] = entropy(hist)

        # Поиск пиков в гистограмме
        peaks, _ = signal.find_peaks(hist, height=np.mean(hist))
        metrics["hist_peaks"] = len(peaks)

        # Анализ экстремальных значений
        extreme_pixels = np.sum((image == 0) | (image == 1))
        metrics["extreme_ratio"] = extreme_pixels / image.size

        # 2. Анализ FFT
        # Нормализация спектра
        fft_norm = fft_spectrum / np.max(fft_spectrum)

        # Поиск пиков в спектре (исключая центральную область)
        center_y, center_x = fft_spectrum.shape[0] // 2, fft_spectrum.shape[1] // 2
        mask = np.ones_like(fft_spectrum, dtype=bool)
        mask[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10] = False
        fft_peaks = fft_norm[mask] > 0.5
        metrics["fft_peaks"] = np.sum(fft_peaks)

        # Анализ крестообразных структур (JPEG артефакты)
        cross_pattern = self._detect_cross_pattern(fft_spectrum)
        metrics["cross_pattern_score"] = cross_pattern

        # 3. Статистический анализ
        # Энтропия изображения
        metrics["image_entropy"] = entropy(image.flatten())

        # Стандартное отклонение
        metrics["image_std"] = np.std(image)

        return metrics

    def _detect_cross_pattern(self, spectrum):
        """Обнаружение крестообразных структур в спектре (JPEG артефакты)"""
        center_y, center_x = spectrum.shape[0] // 2, spectrum.shape[1] // 2

        # Создаем маску для горизонтальных и вертикальных линий
        mask = np.zeros_like(spectrum, dtype=bool)
        mask[center_y - 2 : center_y + 2, :] = True
        mask[:, center_x - 2 : center_x + 2] = True

        # Нормализованный спектр
        spectrum_norm = spectrum / np.max(spectrum)

        # Подсчет ярких пикселей вдоль линий
        cross_score = np.mean(spectrum_norm[mask])
        return cross_score

    def _determine_noise_type(self, metrics):
        """Определение типа шума на основе метрик"""
        # Проверка на импульсный шум
        if metrics["extreme_ratio"] > 0.05:
            return f"Импульсный шум (соль/перец) - {metrics['extreme_ratio']*100:.1f}% выбросов"

        # Проверка на JPEG артефакты
        if metrics["cross_pattern_score"] > 0.3:
            return f"JPEG артефакты (сила: {metrics['cross_pattern_score']:.2f})"

        # Проверка на муар
        if metrics["fft_peaks"] > 5:
            return f"Муар (периодический шум) - {metrics['fft_peaks']} пиков"

        # Проверка на квантование
        if metrics["hist_peaks"] > 10:
            return f"Квантование - {metrics['hist_peaks']} уровней"

        # Проверка на гауссов шум
        if metrics["image_std"] > 0.1:
            return f"Гауссов шум (STD={metrics['image_std']:.2f})"

        return "Низкий уровень шума"

    def get_noise_metrics(self):
        """Получение метрик шума для отображения"""
        return self.noise_metrics

    def apply_filter(self, image, filter_type, **params):
        """Применение выбранного фильтра к изображению"""
        try:
            if len(image.shape) == 3:
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                y_channel = ycrcb[:, :, 0]

                filtered_y = self._apply_filter_to_channel(
                    y_channel, filter_type, **params
                )

                ycrcb[:, :, 0] = filtered_y
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                return self._apply_filter_to_channel(image, filter_type, **params)

        except Exception as e:
            print(f"Ошибка применения фильтра: {e}")
            return image

    def _apply_filter_to_channel(self, channel, filter_type, **params):
        """Применение фильтра к одному каналу"""
        if filter_type == "ihpf":  # Идеальный высокочастотный фильтр
            d0 = params.get("d0", 30)
            return self._apply_ihpf(channel, d0)
        elif filter_type == "ghpf":  # Гауссов высокочастотный фильтр
            d0 = params.get("d0", 30)
            return self._apply_ghpf(channel, d0)
        elif filter_type == "bandpass":  # Полосовой фильтр
            d0 = params.get("d0", 30)
            w = params.get("w", 10)
            return self._apply_bandpass(channel, d0, w)
        elif filter_type == "median":
            kernel_size = params.get("kernel_size", 3)
            return cv2.medianBlur(channel, kernel_size)
        elif filter_type == "gaussian":
            kernel_size = params.get("kernel_size", (3, 3))
            sigma = params.get("sigma", 0.8)
            return cv2.GaussianBlur(channel, kernel_size, sigma)

        return channel

    def _apply_ihpf(self, image, d0):
        """Применение идеального высокочастотного фильтра"""
        # Нормализация входного изображения
        image = image.astype(np.float32) / 255.0

        f = fft2(image)
        fshift = fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Создание маски
        mask = np.ones((rows, cols), dtype=np.float32)
        r = max(d0, 1)  # Минимальное значение d0 = 1
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        # Применение маски
        fshift = fshift * mask
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Нормализация результата
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())
        return np.clip(img_back * 255, 0, 255).astype(np.uint8)

    def _apply_ghpf(self, image, d0):
        """Применение гауссова высокочастотного фильтра"""
        # Нормализация входного изображения
        image = image.astype(np.float32) / 255.0

        f = fft2(image)
        fshift = fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Создание гауссовой маски
        x, y = np.meshgrid(np.arange(-crow, rows - crow), np.arange(-ccol, cols - ccol))
        d = np.sqrt(x * x + y * y)
        d0 = max(d0, 1)  # Минимальное значение d0 = 1
        mask = 1 - np.exp(-(d**2) / (2 * d0**2))

        # Применение маски
        fshift = fshift * mask
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Нормализация результата
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())
        return np.clip(img_back * 255, 0, 255).astype(np.uint8)

    def _apply_bandpass(self, image, d0, w):
        """Применение полосового фильтра"""
        # Нормализация входного изображения
        image = image.astype(np.float32) / 255.0

        f = fft2(image)
        fshift = fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Создание полосовой маски
        x, y = np.meshgrid(np.arange(-crow, rows - crow), np.arange(-ccol, cols - ccol))
        d = np.sqrt(x * x + y * y)
        d0 = max(d0, 1)  # Минимальное значение d0 = 1
        w = max(w, 1)  # Минимальное значение w = 1
        mask = np.exp(-((d - d0) ** 2) / (2 * w**2))

        # Применение маски
        fshift = fshift * mask
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Нормализация результата
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())
        return np.clip(img_back * 255, 0, 255).astype(np.uint8)

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
