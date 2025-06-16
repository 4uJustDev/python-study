import ollama
from PIL import Image
import base64
from io import BytesIO


# Функция для кодирования изображения в base64
def image_to_base64(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


qa_pairs = [
    {
        "question": "Видна ли на изображении надпись «Bite»? (изящный курсив на стене)",
        "expected_answer": "Да",
    },
    {
        "question": "Есть ли люди в кадре? (клиенты с десертами, официант)",
        "expected_answer": "Да",
    },
    {
        "question": "Украшена ли фотозона цветами? (весенние пастельные цветы)",
        "expected_answer": "Да",
    },
    {
        "question": "Выдержан ли стиль интерьера в минимализме?",
        "expected_answer": "Нет (лофт + неоклассика: мрамор, золото, детали)",
    },
    {
        "question": "Используется ли в декоре золотой цвет? (акценты на стойке, рамах)",
        "expected_answer": "Да",
    },
    {
        "question": "Есть ли на столе макаруны? (в хрустальных блюдцах)",
        "expected_answer": "Да",
    },
]

# Путь к изображению
image_path = "images/YA-1.jpeg"
image_base64 = image_to_base64(image_path)

# Проверка каждого вопроса
for qa in qa_pairs:
    response = ollama.generate(
        model="llava",
        prompt=f"Ответь только 'Да' или 'Нет'. {qa['question']}",
        images=[image_base64],
    )

    model_answer = response["response"].strip()

    print(f"Вопрос: {qa['question']}")
    print(f"Ожидаемый ответ: {qa['expected_answer']}")
    print(f"Ответ модели: {model_answer}")
    print("-" * 50)
