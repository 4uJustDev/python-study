from PIL import Image
import os

# Путь к папке с изображениями
folder_path = "programs/labs/photoCorrection/photos/"

# Проходим по всем файлам в папке
for filename in os.listdir(folder_path):
    # Пропускаем файлы, которые уже начинаются с bw_
    if filename.startswith("bw_"):
        continue

    # Проверяем, что это файл изображения (можно добавить другие расширения)
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        try:
            # Открываем изображение
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Конвертируем в ЧБ (grayscale)
            bw_image = image.convert("L")

            # Формируем новое имя файла с префиксом bw_
            new_filename = "bw_" + filename
            new_path = os.path.join(folder_path, new_filename)

            # Сохраняем результат
            bw_image.save(new_path)
            print(f"Обработано: {filename} -> {new_filename}")

        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")
