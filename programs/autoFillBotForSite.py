import ollama


def get_associations(word, count):
    prompt = (
        f"Ты — генератор ассоциаций. Приведи строго {count} ассоциаций к слову '{word}'.\n"
        "Формат вывода: только слова через запятую, без пояснений, нумерации и дополнительного текста.\n"
        "Пример: мяу, хвост, лапа, молоко, шерсть"
    )

    try:
        response = ollama.generate(
            model="deepseek-r1:14b", prompt=prompt, options={"temperature": 0.7}
        )
        # Очищаем ответ и преобразуем в массив
        clean_response = response["response"].strip()
        clean_response = clean_response.split("\n")[
            -1
        ]  # Берем последнюю строку если модель добавила пояснения
        associations = [a.strip() for a in clean_response.split(",")]
        print(associations)
    except Exception:
        print([])


# Пример вызова
get_associations("жена", 10)
