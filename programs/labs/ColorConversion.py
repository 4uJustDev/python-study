import math
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

# Изначальное значение в Lab - Розовый
original_lab = (62.86, 63.27, -9.64)


def lab_to_lch(lab_color):
    """Преобразование LAB в LCH"""
    l, a, b = lab_color
    c = math.sqrt(a**2 + b**2)
    h = math.degrees(math.atan2(b, a)) % 360
    return (l, c, h)


def lch_to_lab(lch_color):
    """Преобразование LCH в LAB"""
    l, c, h = lch_color
    h_rad = math.radians(h)
    a = c * math.cos(h_rad)
    b = c * math.sin(h_rad)
    return (l, a, b)


def lab_to_rgb(lab_color):
    """Преобразование LAB в RGB с использованием colormath"""
    lab = LabColor(lab_l=lab_color[0], lab_a=lab_color[1], lab_b=lab_color[2])
    rgb = convert_color(lab, sRGBColor)
    # Масштабируем значения RGB к диапазону 0-255
    return (rgb.rgb_r * 255, rgb.rgb_g * 255, rgb.rgb_b * 255)


def rgb_to_lab(rgb_color):
    """Преобразование RGB в LAB с использованием colormath"""
    # Нормализуем значения RGB к диапазону 0-1
    r, g, b = [x / 255.0 for x in rgb_color]
    rgb = sRGBColor(r, g, b)
    lab = convert_color(rgb, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)


def rgb_to_hsb(rgb_color):
    """Преобразование RGB в HSB (HSV)"""
    r, g, b = [x / 255.0 for x in rgb_color]
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    s = 0 if cmax == 0 else delta / cmax
    v = cmax

    return (h, s, v)


def hsb_to_rgb(hsb_color):
    """Преобразование HSB (HSV) в RGB"""
    h, s, v = hsb_color
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
    return (r, g, b)


def rgb_to_hsi(rgb_color):
    """Преобразование RGB в HSI"""
    r, g, b = [x / 255.0 for x in rgb_color]

    # Вычисление Hue (аналогично HSV)
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    # Вычисление Intensity
    i = (r + g + b) / 3

    # Вычисление Saturation
    s = 1 - (cmin / i) if i > 0 else 0

    return (h, s, i)


def hsi_to_rgb(hsi_color):
    """Преобразование HSI в RGB"""
    h, s, i = hsi_color

    # Нормализация угла
    h = h % 360

    # Преобразование в радианы
    h_rad = math.radians(h)

    # Случай, когда насыщенность равна 0
    if s == 0:
        return (i * 255, i * 255, i * 255)

    # Сектора
    if 0 <= h < 120:
        b = i * (1 - s)
        r = i * (1 + (s * math.cos(h_rad) / math.cos(math.radians(60) - h_rad)))
        g = 3 * i - (r + b)
    elif 120 <= h < 240:
        h -= 120
        h_rad = math.radians(h)
        r = i * (1 - s)
        g = i * (1 + (s * math.cos(h_rad) / math.cos(math.radians(60) - h_rad)))
        b = 3 * i - (r + g)
    else:
        h -= 240
        h_rad = math.radians(h)
        g = i * (1 - s)
        b = i * (1 + (s * math.cos(h_rad) / math.cos(math.radians(60) - h_rad)))
        r = 3 * i - (g + b)

    # Ограничение значений и масштабирование
    r = max(0, min(1, r)) * 255
    g = max(0, min(1, g)) * 255
    b = max(0, min(1, b)) * 255

    return (r, g, b)


def _validate_lab_colors(color1, color2):
    """Проверка, что оба цвета являются LabColor объектами."""
    if (
        color1.__class__.__name__ != "LabColor"
        or color2.__class__.__name__ != "LabColor"
    ):
        raise ValueError(
            "Delta E functions can only be used with two LabColor objects."
        )


def delta_e_cie1976(color1, color2):
    """
    Calculates the Delta E (CIE1976) of two colors.
    Формула: sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
    """
    _validate_lab_colors(color1, color2)
    return math.sqrt(
        (color1.lab_l - color2.lab_l) ** 2
        + (color1.lab_a - color2.lab_a) ** 2
        + (color1.lab_b - color2.lab_b) ** 2
    )


def delta_e_cie1994(color1, color2, K_L=1, K_C=1, K_H=1, K_1=0.045, K_2=0.015):
    """
    Calculates the Delta E (CIE1994) of two colors.
    """
    _validate_lab_colors(color1, color2)

    # Разницы компонентов
    delta_L = color1.lab_l - color2.lab_l
    delta_a = color1.lab_a - color2.lab_a
    delta_b = color1.lab_b - color2.lab_b

    # Расчет C и H
    C1 = math.sqrt(color1.lab_a**2 + color1.lab_b**2)
    C2 = math.sqrt(color2.lab_a**2 + color2.lab_b**2)
    delta_C = C1 - C2

    delta_H_squared = delta_a**2 + delta_b**2 - delta_C**2
    delta_H = math.sqrt(delta_H_squared) if delta_H_squared > 0 else 0

    # Весовые коэффициенты
    S_L = 1
    S_C = 1 + K_1 * C1
    S_H = 1 + K_2 * C1

    # Компоненты различия
    term1 = delta_L / (K_L * S_L)
    term2 = delta_C / (K_C * S_C)
    term3 = delta_H / (K_H * S_H)

    return math.sqrt(term1**2 + term2**2 + term3**2)


def delta_e_cie2000(color1, color2, Kl=1, Kc=1, Kh=1):
    """
    Calculates the Delta E (CIE2000) of two colors.
    Более сложная формула с корректировками на тон и насыщенность.
    """
    _validate_lab_colors(color1, color2)

    L1, a1, b1 = color1.lab_l, color1.lab_a, color1.lab_b
    L2, a2, b2 = color2.lab_l, color2.lab_a, color2.lab_b

    # Средние значения
    L_bar_prime = (L1 + L2) / 2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    if abs(h1_prime - h2_prime) <= 180:
        h_bar_prime = (h1_prime + h2_prime) / 2
    else:
        h_bar_prime = (
            (h1_prime + h2_prime + 360) / 2
            if (h1_prime + h2_prime) < 360
            else (h1_prime + h2_prime - 360) / 2
        )

    T = (
        1
        - 0.17 * math.cos(math.radians(h_bar_prime - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar_prime))
        + 0.32 * math.cos(math.radians(3 * h_bar_prime + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar_prime - 63))
    )

    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) <= 180:
        pass
    elif h2_prime <= h1_prime:
        delta_h_prime += 360
    else:
        delta_h_prime -= 360

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = (
        2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
    )

    S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / math.sqrt(
        20 + (L_bar_prime - 50) ** 2
    )
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    delta_theta = 30 * math.exp(-(((h_bar_prime - 275) / 25) ** 2))
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))

    term1 = delta_L_prime / (Kl * S_L)
    term2 = delta_C_prime / (Kc * S_C)
    term3 = delta_H_prime / (Kh * S_H)
    term4 = R_T * term2 * term3

    return math.sqrt(term1**2 + term2**2 + term3**2 + term4)


def delta_e_cmc(color1, color2, pl=2, pc=1):
    """
    Calculates the Delta E (CMC) of two colors.
    """
    _validate_lab_colors(color1, color2)

    L1, a1, b1 = color1.lab_l, color1.lab_a, color1.lab_b
    L2, a2, b2 = color2.lab_l, color2.lab_a, color2.lab_b

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)

    delta_L = L2 - L1
    delta_C = C2 - C1
    delta_a = a2 - a1
    delta_b = b2 - b1

    delta_H_squared = delta_a**2 + delta_b**2 - delta_C**2
    delta_H = math.sqrt(max(0, delta_H_squared))

    S_L = 0.040975 * L1 / (1 + 0.01765 * L1) if L1 >= 16 else 0.511
    S_C = 0.0638 * C1 / (1 + 0.0131 * C1) + 0.638
    F = math.sqrt(C1**4 / (C1**4 + 1900))

    if 164 <= (math.degrees(math.atan2(b1, a1)) % 360) <= 345:
        T = 0.56 + abs(
            0.2 * math.cos(math.radians(168 + math.degrees(math.atan2(b1, a1))))
        )
    else:
        T = 0.36 + abs(
            0.4 * math.cos(math.radians(35 + math.degrees(math.atan2(b1, a1))))
        )

    S_H = S_C * (F * T + 1 - F)

    term1 = delta_L / (pl * S_L)
    term2 = delta_C / (pc * S_C)
    term3 = delta_H / S_H

    return math.sqrt(term1**2 + term2**2 + term3**2)


def main():
    # 1. Задаем исходный цвет в LAB
    print(f"Исходный цвет LAB: {original_lab}")

    # 2. Преобразование LAB → LCH
    lch = lab_to_lch(original_lab)
    print(f"LCH: {lch}")

    # 3. Преобразование LAB → RGB
    rgb = lab_to_rgb(original_lab)
    print(f"RGB: {rgb}")

    # 4. Преобразование RGB → HSB и HSI
    hsb = rgb_to_hsb(rgb)
    print(f"HSB: {hsb}")

    hsi = rgb_to_hsi(rgb)
    print(f"HSI: {hsi}")

    # 5. Обратное преобразование в LAB

    # LCH → LAB
    lab_from_lch = lch_to_lab(lch)
    print(f"LAB из LCH: {lab_from_lch}")

    # RGB → LAB
    lab_from_rgb = rgb_to_lab(rgb)
    print(f"LAB из RGB: {lab_from_rgb}")

    # HSB → RGB → LAB
    rgb_from_hsb = hsb_to_rgb(hsb)
    lab_from_hsb = rgb_to_lab(rgb_from_hsb)
    print(f"LAB из HSB: {lab_from_hsb}")

    # HSI → RGB → LAB
    rgb_from_hsi = hsi_to_rgb(hsi)
    lab_from_hsi = rgb_to_lab(rgb_from_hsi)
    print(f"LAB из HSI: {lab_from_hsi}")

    # 6. Расчет ΔE, ΔE94, ΔE00
    de_lch_76 = delta_e_cie1976(LabColor(*original_lab), LabColor(*lab_from_lch))
    de_lch_94 = delta_e_cie1994(LabColor(*original_lab), LabColor(*lab_from_lch))
    de_lch_00 = delta_e_cie2000(LabColor(*original_lab), LabColor(*lab_from_lch))

    de_rgb_76 = delta_e_cie1976(LabColor(*original_lab), LabColor(*lab_from_rgb))
    de_rgb_94 = delta_e_cie1994(LabColor(*original_lab), LabColor(*lab_from_rgb))
    de_rgb_00 = delta_e_cie2000(LabColor(*original_lab), LabColor(*lab_from_rgb))

    de_hsb_76 = delta_e_cie1976(LabColor(*original_lab), LabColor(*lab_from_hsb))
    de_hsb_94 = delta_e_cie1994(LabColor(*original_lab), LabColor(*lab_from_hsb))
    de_hsb_00 = delta_e_cie2000(LabColor(*original_lab), LabColor(*lab_from_hsb))

    de_hsi_76 = delta_e_cie1976(LabColor(*original_lab), LabColor(*lab_from_hsi))
    de_hsi_94 = delta_e_cie1994(LabColor(*original_lab), LabColor(*lab_from_hsi))
    de_hsi_00 = delta_e_cie2000(LabColor(*original_lab), LabColor(*lab_from_hsi))

    # 7. Вывод результатов в виде таблиц

    print("\nТаблица 3: Результаты обратного пересчета")
    print("| Метод      | L        | a        | b        |")
    print("|------------|----------|----------|----------|")
    print(
        f"| LCH→LAB    | {lab_from_lch[0]:<8.2f} | {lab_from_lch[1]:<8.2f} | {lab_from_lch[2]:<8.2f} |"
    )
    print(
        f"| RGB→LAB    | {lab_from_rgb[0]:<8.2f} | {lab_from_rgb[1]:<8.2f} | {lab_from_rgb[2]:<8.2f} |"
    )

    print("\nТаблица 4: Результаты обратного пересчета")
    print("| Метод      | L        | a        | b        |")
    print("|------------|----------|----------|----------|")
    print(
        f"| HSB→LAB    | {lab_from_hsb[0]:<8.2f} | {lab_from_hsb[1]:<8.2f} | {lab_from_hsb[2]:<8.2f} |"
    )
    print(
        f"| HSI→LAB    | {lab_from_hsi[0]:<8.2f} | {lab_from_hsi[1]:<8.2f} | {lab_from_hsi[2]:<8.2f} |"
    )

    # 7. Вывод результатов в виде таблиц
    print("\nТаблица 5: Цветовые различия")
    print("| Метрика    | LCH→LAB  | RGB→LAB  | HSB→LAB  | HSI→LAB  |")
    print("|------------|----------|----------|----------|----------|")
    print(
        f"| ΔE (1976)  | {de_lch_76:<8.2f} | {de_rgb_76:<8.2f} | {de_hsb_76:<8.2f} | {de_hsi_76:<8.2f} |"
    )
    print(
        f"| ΔE (1994)  | {de_lch_94:<8.2f} | {de_rgb_94:<8.2f} | {de_hsb_94:<8.2f} | {de_hsi_94:<8.2f} |"
    )
    print(
        f"| ΔE (2000)  | {de_lch_00:<8.2f} | {de_rgb_00:<8.2f} | {de_hsb_00:<8.2f} | {de_hsi_00:<8.2f} |"
    )


if __name__ == "__main__":
    main()


# Точность обратного преобразования

# Наиболее точное обратное преобразование в цветовое пространство LAB обеспечивает метод LCH→LAB, так как LCH является полярным представлением LAB и не требует промежуточных конвертаций.

# Методы RGB→LAB, HSB→LAB и HSI→LAB дают близкие, но не идентичные исходному LAB значения, что объясняется погрешностями при конвертации между цветовыми моделями.

# Сравнение цветовых различий

# Наименьшее отклонение от исходного цвета наблюдается при LCH→LAB (ΔE = 0 по всем метрикам), что подтверждает корректность преобразования.

# Для остальных методов различия незначительны, но присутствуют:

# ΔE (1976): ~1.45

# ΔE (1994): ~0.63–0.64

# ΔE (2000): ~0.55–0.56

# Это говорит о том, что преобразования через RGB, HSB и HSI вносят небольшие, но заметные искажения.

# Сравнение RGB, HSB и HSI

# Методы RGB→LAB, HSB→LAB и HSI→LAB дают практически идентичные результаты (разница в ΔE менее 0.02), что объясняется их тесной взаимосвязью (все они основаны на RGB).

# Небольшие расхождения между ними могут быть связаны с разными алгоритмами пересчета яркости и насыщенности.

# Рекомендации

# Для максимальной точности преобразований в LAB следует использовать LCH, если это возможно.

# При работе с RGB, HSB или HSI следует учитывать возможные погрешности (~1.5 ΔE по CIEDE1976).

# Для критичных к цвету задач рекомендуется использовать более точные метрики (например, ΔE 2000), так как они лучше учитывают воспринимаемую разницу.

# Заключение
# Результаты подтверждают, что обратное преобразование в LAB из LCH является наиболее точным, тогда как конвертация через RGB и его производные (HSB, HSI) приводит к незначительным, но измеримым отклонениям. Выбор метода зависит от требуемой точности и исходных данных.
