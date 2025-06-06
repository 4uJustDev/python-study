import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def parse_page(url):
    response = requests.get(url)
    response.encoding = "windows-1251"
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    data = []
    current_stimulus = None

    for line in lines:
        # Стимул — строка без цифр и двоеточий
        if not re.search(r"\d+:", line) and not line.endswith(")"):
            current_stimulus = line
        elif current_stimulus:
            # Найти все группы вида N: ... (до следующей N: или конца строки)
            matches = re.findall(r"(\d+):([^\d]+?)(?=(?:\d+:|$))", line)
            for freq, reaction_str in matches:
                # Разбить по ; и по , (если есть), убрать пустые
                reactions = [
                    r.strip() for r in re.split(r";|,", reaction_str) if r.strip()
                ]
                for reaction in reactions:
                    # Определяем тип отношения
                    if (
                        "," in reaction_str
                        or " и " in reaction_str
                        or ";" in reaction_str
                    ):
                        relation_type = ""
                    else:
                        relation_type = "Словосочетание"
                    data.append(
                        {
                            "Частот": freq,
                            "Стимул": current_stimulus,
                            "Реакция": reaction,
                            "Синтагматическое отношение": relation_type,
                            "Парадигматические отношения": "",
                        }
                    )
    return data


def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(
        filename,
        index=False,
        columns=[
            "Частот",
            "Стимул",
            "Реакция",
            "Синтагматическое отношение",
            "Парадигматические отношения",
        ],
    )


# URL для парсинга
# url = "http://it-claim.ru/Library/Books/Association_IT/chapter2_1.htm"
url = "http://it-claim.ru/Library/Books/Association_IT/chapter2_2.htm"

# Парсим данные
parsed_data = parse_page(url)

# Сохраняем в Excel
# save_to_excel(parsed_data, "associations_data.xlsx")
save_to_excel(parsed_data, "associations_data_2.xlsx")

print(
    f"Данные сохранены в файл associations_data.xlsx. Обработано {len(parsed_data)} записей."
)
