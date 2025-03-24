import asyncio
import random
import ollama
import time
import os
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


load_dotenv()

LOGIN = os.getenv("LOGIN_DICTIONARY")
PASSWORD = os.getenv("PASSWORD_DICTIONARY")


async def get_associations(word, count):
    prompt = (
        f"Ты — генератор ассоциаций. Приведи строго {count} ассоциаций к слову '{word}'.\n"
        "Формат вывода: только русские слова через запятую, без пояснений, нумерации и дополнительного текста.\n"
        "Пример: мяу, хвост, лапа, молоко, шерсть"
    )

    try:
        response = ollama.generate(
            model="deepseek-r1:14b",
            prompt=prompt,
            options={"temperature": 0.7},
        )
        # Очищаем ответ и преобразуем в массив
        clean_response = response["response"].strip()
        clean_response = clean_response.split("\n")[
            -1
        ]  # Берем последнюю строку если модель добавила пояснения
        associations = [a.strip() for a in clean_response.split(",")]
        return associations
    except Exception as e:
        print(f"Ошибка при получении ассоциаций: {e}")
        return []


async def process_page():

    # Запускаем браузер
    driver = webdriver.Chrome()

    try:
        # Открываем страницу и логинимся
        driver.get("https://dictionary-exp-frontend.vercel.app/")

        login_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//input[@placeholder="Введите username..."]')
            )
        )
        login_field.send_keys(LOGIN)

        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//input[@type="password"]'))
        )
        password_field.send_keys(PASSWORD)

        login_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
        )
        login_button.click()

        # Начинаем заполнять
        for i in range(115):
            start_time = time.time()
            # Находим слово
            word_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "form table tbody tr td:first-child")
                )
            )
            word = word_element.text.strip()
            print(f"Найдено слово: {word}")

            associations = await get_associations(word, 10)

            end_time = time.time()  # Засекаем время окончания генерации
            generation_time = end_time - start_time  # Вычисляем время генерации

            if not associations:
                print("Не удалось получить ассоциации")
                return

            association_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//input[@class="input"]'))
            )

            randomWord = random.choice(associations)

            while word.lower() == randomWord.lower():
                randomWord = random.choice(associations)

            association_field.send_keys(randomWord.lower())

            vote_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
            )
            vote_button.click()
            print(
                f"#{i+1} Для {word} выбрано слово: {randomWord} ({round(generation_time, 1)})"
            )

    except Exception as e:
        print(f"Ошибка при работе с веб-страницей: {str(e)}")
    finally:
        print("good")


async def main():
    await process_page()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
