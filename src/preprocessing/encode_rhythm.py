import os
import sys
import re

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants.music_constants import (
    ALLOWED_SIGNATURES,
    NOTES,
    MODE_INTERVALS,
    TICKS_PER_BEAT,
    BASE_TIME_SIGNATURE,
    CHROMATIC_SCALE_SIZE,
    FLAT_TO_SHARP,
    GLOBAL_LENGHT_LIMIT
)

# Разрешенные делители в дроби длительности аккорда
ALLOWED_DENOMINATORS_FOR_SIGNATURE_4 = {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}
ALLOWED_DENOMINATORS_FOR_SIGNATURE_3 = {1, 2, 3, 4, 6, 9, 12, 18, 36}

NOTE_PATTERN = '|'.join(sorted(NOTES, key=len, reverse=True))
MODE_PATTERN = '|'.join(sorted([m for m in MODE_INTERVALS.keys() if m], key=len, reverse=True))

CHORD_PATTERN = rf'(?:{NOTE_PATTERN})(?:{MODE_PATTERN})?-(?:\d+(?:/\d+)?)'
FULL_PATTERN = rf'^{CHORD_PATTERN}(?:,{CHORD_PATTERN})*$'

def encode_rhythm(rhythm_string, signature, tune_difference=0):
    """
    Кодирует строку аккордов ритм-гитары в дискретную временную последовательность.

    Args:
        rhythm_string (str): строка аккордов (например "C-1/2,G-1,Am-2")
        signature (int): размер такта (3 или 4)
        tune_difference (int): сдвиг по полутонам для транспонирования

    Returns:
        encoded_rhythm (list): закодированная ритм-последовательность

    Raises:
        ValueError: неверный формат входных аккордов
    """

    if signature not in ALLOWED_SIGNATURES:
        raise ValueError("signature должен быть 3 или 4")
    else:
        if signature == 3:
            allowed_denominators = ALLOWED_DENOMINATORS_FOR_SIGNATURE_3
        elif signature == 4:
            allowed_denominators = ALLOWED_DENOMINATORS_FOR_SIGNATURE_4
    encoded_rhythm = []

    rhythm_string = rhythm_string.replace(" ","")

    # Замена записей с бемолями на диезы
    for k, v in FLAT_TO_SHARP.items():
        if k in rhythm_string:
            rhythm_string = rhythm_string.replace(k, v)

    if not re.fullmatch(FULL_PATTERN, rhythm_string.replace(" ", "")):
        raise ValueError(f"Некорректный формат аккордов: {rhythm_string}")

    # Разделяем строку на отдельные аккорды
    chords = [chord.strip() for chord in rhythm_string.split(',')]

    for chord in chords:

        # =========================
        # ПАРСИНГ АККОРДОВ
        # =========================

        name, duration_text = chord.split('-')

        # Обработка длительности (дробной и целой)
        if '/' in duration_text:
            numerator, denominator = map(int, duration_text.split('/'))

            if denominator not in allowed_denominators:
                raise ValueError(f"Недопустимый знаменатель: {denominator}")

            # Количество временных шагов (1/48 доля)
            num_subdivisions = int(TICKS_PER_BEAT / BASE_TIME_SIGNATURE * signature / denominator * numerator)
        else:
            num_subdivisions = int(TICKS_PER_BEAT / BASE_TIME_SIGNATURE * signature * int(duration_text))

        # Бинарное представление аккорда (12 полутонов)
        encoded_chord = [0] * CHROMATIC_SCALE_SIZE

        # =========================
        # Построение аккорда
        # =========================

        if name:

            # Определяем тонику и тип аккорда
            if name[:2] in NOTES:
                root_note = name[:2]
                mode = name[2:]
            elif name[0] in NOTES:
                root_note = name[0]
                mode = name[1:]
            else:
                raise ValueError(f"Некорректный аккорд: {name}")

            # Индекс тоники с учётом транспонирования
            root_index = (NOTES.index(root_note) + tune_difference) % CHROMATIC_SCALE_SIZE

            # Интервалы нот в аккорде
            intervals = MODE_INTERVALS.get(mode, [])

            # Добавление тоники
            encoded_chord[root_index] = 1

            # Построение аккорда по интервалам
            current_index = root_index
            for interval in intervals:
                current_index = (current_index + interval) % CHROMATIC_SCALE_SIZE
                encoded_chord[current_index] = 1

        # =========================
        # РАЗВЕРТКА АККОРДА ВО ВРЕМЕНИ
        # =========================

        for _ in range(num_subdivisions):
            encoded_rhythm.append(encoded_chord)

    if len(encoded_rhythm) > GLOBAL_LENGHT_LIMIT:
        current_lenght = len(encoded_rhythm) / TICKS_PER_BEAT / BASE_TIME_SIGNATURE * signature
        lenght_limit = GLOBAL_LENGHT_LIMIT / TICKS_PER_BEAT / BASE_TIME_SIGNATURE * signature
        raise ValueError(f"Слишком длинная последовательность аккордов: {current_lenght} > {lenght_limit}")

    return encoded_rhythm