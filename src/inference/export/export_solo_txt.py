import os, sys

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from constants.music_constants import (
    ALLOWED_SIGNATURES, 
    NOTE_STATES,
    TICKS_PER_BEAT,
    BASE_TIME_SIGNATURE
)

# Стартовое состояние табулатуры для нового такта (6 струн гитары)
EMPTY_STRINGS = {
    'e': "e| -",
    'B': "B| -",
    'G': "G| -",
    'D': "D| -",
    'A': "A| -",
    'E': "E| -"
}

TAB_STEP_SEPARATOR = '-'
REPEAT_NOTE_THRESHOLD = 3
BAR_DIVISOR = TICKS_PER_BEAT // BASE_TIME_SIGNATURE

E_STRING_RANGE = (1, 8)
A_STRING_RANGE = (9, 15)
D_STRING_RANGE = (16, 22)
G_STRING_RANGE = (23, 29)
B_STRING_RANGE = (30, 35)
HIGH_E_STRING_RANGE = (36, 47)

E_STRING_OFFSET = 1
A_STRING_OFFSET = 6
D_STRING_OFFSET = 11
G_STRING_OFFSET = 16
B_STRING_OFFSET = 20
HIGH_E_STRING_OFFSET = 26


def recognize_string(pitch, state):
    """
    Преобразует числовую ноту в:
    (струна, лад)

    Args:
        pitch (int): индекс ноты на грифе гитары
        state (int): состояние ноты в передаваемый момент времени
    
    Returns:
        string (str): нота открытой струны, на которой воспроизводится нота
        fret (str): лад, на котором зажата струна
    """

    # 0 - это пауза (нет ноты)
    if pitch == 0:
        return None

    # =========================
    # Оформление струны и лад
    # =========================

    if E_STRING_RANGE[0] <= pitch <= E_STRING_RANGE[1]:
        string, fret = 'E', str(pitch - E_STRING_OFFSET)
    elif A_STRING_RANGE[0] <= pitch <= A_STRING_RANGE[1]:
        string, fret = 'A', str(pitch - A_STRING_OFFSET)
    elif D_STRING_RANGE[0] <= pitch <= D_STRING_RANGE[1]:
        string, fret = 'D', str(pitch - D_STRING_OFFSET)
    elif G_STRING_RANGE[0] <= pitch <= G_STRING_RANGE[1]:
        string, fret = 'G', str(pitch - G_STRING_OFFSET)
    elif B_STRING_RANGE[0] <= pitch <= B_STRING_RANGE[1]:
        string, fret = 'B', str(pitch - B_STRING_OFFSET)
    else:
        string, fret = 'e', str(pitch - HIGH_E_STRING_OFFSET)

    # =========================
    # Отображение техники извлечения ноты
    # =========================

    if state == NOTE_STATES["slide"]:
        fret = '/' + fret
    elif state == NOTE_STATES["hammer-on"]:
        fret = 'h' + fret
    elif state == NOTE_STATES["pull-off"]:
        fret = fret + 'p'
    elif state == NOTE_STATES["bend"]:
        fret = fret + 'b'

    return string, fret


def add_note_to_tab(strings, note_1, note_2):
    """
    Добавляет один временной шаг в табулатуру.

    Args:
        strings (list):
        note_1 (list|None): первая нота с информацией о струне и ладе, на которой она звучит
        note_2 (list|None): вторая нота с информацией о струне и ладе, на которой она звучит
    """

    # Обработка случая, когда передана только первая нота
    if note_1 is not None and note_2 is None:
        string, fret = note_1

        other_strings = [s for s in strings if s != string]
 
        for s in other_strings:
            strings[s] += TAB_STEP_SEPARATOR * len(fret)

        strings[string] += fret

    # Обработка случая, когда передана только вторая нота
    elif note_1 is None and note_2 is not None:
        string, fret = note_2

        other_strings = [s for s in strings if s != string]

        for s in other_strings:
            strings[s] += TAB_STEP_SEPARATOR * len(fret)

        strings[string] += fret

    # # Обработка случая, когда переданы обе нота
    elif note_1 is not None and note_2 is not None:
        note_1_string, note_1_fret = note_1
        note_2_string, note_2_fret = note_2

        other_strings = [s for s in strings if s not in (note_1_string, note_2_string)]

        # Если ноты на разных струнах
        if note_1_string != note_2_string:
            # Выравнивание длины для отображения табулатуры без смещения на каждой из струн
            max_len = max(len(note_1_fret), len(note_2_fret))

            note_1_fret = note_1_fret.ljust(max_len, TAB_STEP_SEPARATOR)
            note_2_fret = note_2_fret.ljust(max_len, TAB_STEP_SEPARATOR)

            for s in other_strings:
                strings[s] += TAB_STEP_SEPARATOR * max_len

            strings[note_1_string] += note_1_fret
            strings[note_2_string] += note_2_fret

        else:
            # Если две ноты на одной струне - объединение через "+"
            for s in other_strings:
                strings[s] += TAB_STEP_SEPARATOR * (len(note_1_fret) + len(note_2_fret) + 1)

            strings[note_1_string] += f"{note_1_fret}+{note_2_fret}"

    for s in strings:
        strings[s] += TAB_STEP_SEPARATOR


def export_solo_txt(solo, signature):
    """
    Преобразует последовательность нот соло-гитары в текстовую табулатуру

    Args:
        solo (list):
        signature (int): размер такта (3 или 4)

    Returns:
        final_tab (str): строка с представлением табулатуры соло-гитары в текстовом виде
    """

    if signature not in ALLOWED_SIGNATURES:
        raise ValueError("signature должен быть 3 или 4")

    final_tab = f"Музыкальный размер: {signature}/4.\n\n"

    # Сколько шагов подряд длится одна и та же нота
    last_duration = 0

    # Текущее состояние табулатуры
    strings = EMPTY_STRINGS.copy()

    # Номер текущего такта
    bar = 1

    # Длина такта в шагах
    bar_len = BAR_DIVISOR * signature

    # Флаг: нужно ли продолжить ноту из прошлого такта
    need_bar_continuation = False

    # =========================
    # ОСНОВНОЙ ЦИКЛ
    # =========================

    for i in range(len(solo)):

        pitch_1, state_1, pitch_2, state_2, = solo[i]

        # Обработка первого шага
        if i == 0:
            note_1 = recognize_string(pitch_1, state_1)
            note_2 = recognize_string(pitch_2, state_2)

            add_note_to_tab(strings, note_1, note_2)
            continue

        # Обработка завершающего шага
        if i == len(solo) - 1:
            final_tab += f"Такт {bar}:\n"
            for s in strings.values():
                final_tab += s + '|\n'
            continue

        # Обработка перехода на новый такт
        if i % bar_len == 0:
            final_tab += f"Такт {bar}:\n"
            for s in strings.values():
                final_tab += s + '|\n'
            bar += 1
            final_tab += '\n'
            last_duration = 0
            strings = EMPTY_STRINGS.copy()
            need_bar_continuation = True

        prev_pitch_1, _, prev_pitch_2, _ = solo[i - 1]

        # Проверка соответствия нот на прошлом и текущем шагах
        same_notes = (pitch_1 == prev_pitch_1 and pitch_2 == prev_pitch_2)

        is_sustain = (
            state_1 == NOTE_STATES["sustain"] or state_2 == NOTE_STATES["sustain"]
        )

        same_as_previous = same_notes and is_sustain

        # Отображение продолжения ноты с предыдущего такта
        if need_bar_continuation and same_notes:

            note_1 = recognize_string(pitch_1, state_1)
            note_2 = recognize_string(pitch_2, state_2)

            if note_1 is not None:
                note_1 = (note_1[0], f"({note_1[1]})")
            if note_2 is not None:
                note_2 = (note_2[0], f"({note_2[1]})")

            add_note_to_tab(strings, note_1, note_2)

            need_bar_continuation = False
            last_duration = 0
            continue

        need_bar_continuation = False

        # Проверка продолжительности ноты с предыдешго шага
        if same_as_previous:
            last_duration += 1

            # Вставляем паузу для визуального разделения
            if last_duration == REPEAT_NOTE_THRESHOLD:
                add_note_to_tab(strings, None, None)
                last_duration = 0

        # Обработка только что извлечённой ноты
        else:
            note_1 = recognize_string(pitch_1, state_1)
            note_2 = recognize_string(pitch_2, state_2)

            add_note_to_tab(strings, note_1, note_2)

            last_duration = 0

    return final_tab