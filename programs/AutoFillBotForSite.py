import random
import ollama
import time
import os
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Constants
DELAY_RANGE = (5.2, 11.4)
ASSOCIATIONS_COUNT = 10
MAX_WORDS_TO_PROCESS = 116
TEMPERATURE_FOR_MODEL = 0.2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
load_dotenv()

LOGIN = os.getenv("LOGIN_DICTIONARY")
PASSWORD = os.getenv("PASSWORD_DICTIONARY")


def get_associations(word, count):
    prompt = f"Строго {count} русских слов ассоциаций к слову'{word}'. Только слова через запятую, только одно слово, без пояснений. Пример: лес, дерево, лист"

    response = ollama.generate(
        model="mistral:7b",
        prompt=prompt,
        options={"temperature": TEMPERATURE_FOR_MODEL},
    )

    clean_response = response["response"].strip()
    clean_response = clean_response.split("\n")[-1]

    associations = [a.strip() for a in clean_response.split(",")]
    return associations


def process_page():
    # Browser open and set options
    options = webdriver.ChromeOptions()
    options.add_argument("--auto-open-devtools-for-tabs")

    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    )

    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3]
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['ru-RU', 'ru']
            });
        """
        },
    )

    # Login
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

    # Fill associations
    for i in range(MAX_WORDS_TO_PROCESS):
        # Generate
        start_time = time.time()
        word_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "form table tbody tr td:first-child")
            )
        )
        word = word_element.text.strip()
        print(f"#{i+1} Слово: {word}\n")
        associations = get_associations(word, ASSOCIATIONS_COUNT)
        end_time = time.time()

        generation_time = end_time - start_time

        # Calculate random delay between
        target_delay = random.uniform(*DELAY_RANGE)
        if generation_time < target_delay:
            remaining_delay = target_delay - generation_time
            time.sleep(remaining_delay)
            generation_time = target_delay

        if not associations:
            print("Не удалось получить ассоциации")
            return

        # Fill
        association_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//input[@class="input"]'))
        )

        randomWord = random.choice(associations).lower()

        while word.lower() == randomWord:
            randomWord = random.choice(associations)

        association_field.send_keys(randomWord)

        # Apply answer
        vote_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
        )

        vote_button.click()

        # Logs
        print(
            f"#{i+1} Слова : {associations}\n"
            f"#{i+1} Для {word} выбрано слово: {randomWord} ({round(generation_time, 1)})\n----\n"
        )


def main():
    process_page()
    # This used for stay browser open
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
