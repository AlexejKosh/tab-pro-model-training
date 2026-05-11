from constants.music_constants import (
    ALLOWED_SIGNATURES,
    NOTE_STATES,
    TICKS_PER_BEAT,
    BASE_TIME_SIGNATURE
)

def classify_note_state(note):
    """
    Определяет состояние музыкального события (ноты).

    Args:
        note (list): данные ноты из GuitarPro файла

    Returns:
        int: состояние ноты
            0 - пауза
            1 - обычная атака
            2 - slide
            3 - hammer-on
            4 - pull-off / grace
            5 - bend
            6 - sustain / tie (продолжение ноты)
    """

    if note is None:
        return NOTE_STATES["pause"]

    note_state = note[1]
    note_type = note[2]

    # Проверка ноты на продолжение
    if str(note_type).endswith("tie"):
        return NOTE_STATES["sustain"]

    # Обозначение способа извлечения ноты
    if note_state is not None:
        if note_state.hammer:
            return NOTE_STATES["hammer-on"]
        elif len(note_state.slides) > 0:
            return NOTE_STATES["slide"]
        elif note_state.bend is not None:
            return NOTE_STATES["bend"]
        elif note_state.grace is not None:
            return NOTE_STATES["pull-off"]

    return NOTE_STATES["attack"]


def process_new_event(raw_solo_data, note_slot, note_sequence, note_index):
    """
    Обрабатывает переход к новому событию (ноте) во временной сетке.

    Args:
        raw_solo_data (list): исходные события из GuitarPro файла
        note_slot (list): текущее состояние [index, start_time, end_time]
        note_sequence (list): выходая последовательность
        note_index (int): индекс ноты (1 или 2)
    """

    # =========================
    # ПЕРЕХОД К НОВОМУ СОБЫТИЮ
    # =========================
    
    # Переход к новому событию в дорожке
    note_slot[0] += 1
    # Объявляем время начала события
    note_slot[1] += raw_solo_data[note_slot[0]][0]

    # Коррекция накопленной ошибки float
    if abs(note_slot[1] - int(note_slot[1])) < 1e-6:
        note_slot[1] = int(note_slot[1])

    # Загрузка данных о конкретной ноте [pitch, state, type, durationPercent]
    note_data = raw_solo_data[note_slot[0]][note_index]

    # =========================
    # РАСЧЕТ ДЛИТЕЛЬНОСТИ СОБЫТИЯ
    # =========================
    
    if note_data is not None:
        note_slot[2] = note_slot[1] + raw_solo_data[note_slot[0]][0] * note_data[3]
    else:
        note_slot[2] = note_slot[1] + raw_solo_data[note_slot[0]][0]


    note_state = classify_note_state(note_data)

    # =========================
    # ЗАПИСЬ В ПОСЛЕДОВАТЕЛЬНОСТЬ
    # =========================
    # Запись паузы
    if note_state == NOTE_STATES["pause"]:
        note_sequence.append([0, NOTE_STATES["pause"]])
    # Запись продолжения нот, которые до этого начали звучать
    elif note_state == NOTE_STATES["sustain"]:
        if note_sequence:
            note_sequence.append([note_sequence[-1][0], NOTE_STATES["sustain"]])
        else:
            if note_data is not None:
                note_sequence.append([note_data[0], NOTE_STATES["sustain"]])
            else:
                note_sequence.append([0, NOTE_STATES["pause"]])
    # Запись нот, которые только что начали звучать
    else:
        note_sequence.append([note_data[0], note_state])


def encode_solo(raw_solo_data, signature):
    """
    Кодирует последовательность событий GuitarPro соло-гитары в дискретную временную последовательность.

    Args:
        raw_solo_data (list): список событий [duration, note1, note2]
        signature (int): размер такта (3 или 4)

    Returns:
        list: последовательность вида
              [[pitch1, state1, pitch2, state2], ...]
    """

    if signature not in ALLOWED_SIGNATURES:
        raise ValueError("signature должен быть 3 или 4")

    if not raw_solo_data:
        return []

    # =========================
    # ПОДГОТОВКА ПЕРЕМЕННЫХ
    # =========================
    
    total_duration = sum(i[0] for i in raw_solo_data)

    # Последовательности нот в закодированном виде
    # - формат который будет понимать обучаемая модель
    note_1_sequence = []
    note_2_sequence = []
    # Объединение последовательностей для первых и вторых нот
    output_sequence = []

    # [index, start_time, end_time]
    note_1_slot = [-1, 0, 0]
    note_2_slot = [-1, 0, 0]

    # =========================
    # ВРЕМЕННАЯ ДИСКРЕТИЗАЦИЯ
    # =========================

    # Вычисление количества шагов на дорожке соло-гитары с заданной временной дискретизацией
    total_steps = int(round(total_duration * TICKS_PER_BEAT // BASE_TIME_SIGNATURE * signature))

    # Полностью пустое соло
    if total_steps <= 0:
        return []

    # =========================
    # ОСНОВНОЙ ЦИКЛ
    # =========================
    for step in range(total_steps):
        # Начало шага в тактах относительно начала соло
        t = step / ((TICKS_PER_BEAT // BASE_TIME_SIGNATURE) * signature)

        # Обработка нового события
        if t >= note_1_slot[1]:
            process_new_event(raw_solo_data, note_1_slot, note_1_sequence, 1)
            process_new_event(raw_solo_data, note_2_slot, note_2_sequence, 2)

        else:
            # Обработка продолжения события
            # Обработка ноты 1
            if t < note_1_slot[2]:
                if note_1_sequence:
                    if note_1_sequence[-1][1] != 0:
                        note_1_sequence.append([note_1_sequence[-1][0], 6])  # sustain
                    else:
                        note_1_sequence.append(note_1_sequence[-1])
                else:
                    note_1_sequence.append([0, 0])
            else:
                note_1_sequence.append([0, 0])

            # Обработка ноты 2
            if t < note_2_slot[2]:
                if note_2_sequence:
                    if note_2_sequence[-1][1] != 0:
                        note_2_sequence.append([note_2_sequence[-1][0], 6])
                    else:
                        note_2_sequence.append(note_2_sequence[-1])
                else:
                    note_2_sequence.append([0, 0])
            else:
                note_2_sequence.append([0, 0])

    # =========================
    # ОБЪЕДИНЕНИЕ НОТ
    # =========================
    for i in range(len(note_1_sequence)):
        output_sequence.append(note_1_sequence[i] + note_2_sequence[i])

    return output_sequence