import os
import sys
import json
import collections
import math

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants.music_constants import NOTES, CHROMATIC_SCALE_SIZE, NOTE_INDEX_SHIFT, NOTE_STATES

# Сопоставление id состояния -> имя состояния, берём из NOTE_STATES
STATE_NAMES = {value: name for name, value in NOTE_STATES.items()}

def calculate_entropy(counter):
    """
    Вычисляет энтропию распределения (энтропия Шеннона).

    Args:
        counter (collections.Counter): распределение значений

    Returns:
        entropy (float): значение энтропии
    """
    total = sum(counter.values())

    if total == 0:
        return 0

    entropy = 0.0

    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def analyze_dataset_file(path):
    """
    Анализирует датасет и собирает статистику.

    Args:
        path (str): путь к JSON-файлу датасета

    Returns:
        tuple:
            result (dict): распределение нот по тональностям
            state_probs (list): распределение состояний (%)
            key_probs (list): распределение тональностей (%)
            notes_entropy (float): энтропия нот
            states_entropy (float): энтропия состояний

    Raises:
        IOError: ошибка чтения файла
        json.JSONDecodeError: некорректный JSON
    """

    # =========================
    # ЗАГРУЗКА ДАННЫХ
    # =========================
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    note_stats = {}                    # key -> Counter(note)
    totals = collections.Counter()    # key -> total notes
    key_distribution = collections.Counter()
    state_counter = collections.Counter()

    # =========================
    # ОСНОВНОЙ ПРОХОД ПО ДАННЫМ
    # =========================
    for song in data:
        key = song.get('key')

        if key is None:
            continue

        key_distribution[key] += 1

        if key not in note_stats:
            note_stats[key] = collections.Counter()

        solo = song.get('solo', [])

        for unit in solo:
            if not isinstance(unit, (list, tuple)) or len(unit) < 4:
                continue

            # =========================
            # ОБРАБОТКА НОТ
            # =========================
            for pitch_index in (0, 2):
                pitch = unit[pitch_index]

                if not pitch:
                    continue

                try:
                    idx = int(pitch) + NOTE_INDEX_SHIFT
                except Exception:
                    continue

                note_name = NOTES[idx % CHROMATIC_SCALE_SIZE]

                note_stats[key][note_name] += 1
                totals[key] += 1

            # =========================
            # ОБРАБОТКА СОСТОЯНИЙ СИГНАЛА
            # =========================
            for state_index in (1, 3):
                state = unit[state_index]

                try:
                    state_value = int(state)
                except Exception:
                    continue

                state_counter[state_value] += 1

    # =========================
    # РАСЧЁТ РАСПРЕДЕЛЕНИЯ НОТ
    # =========================
    result = {}

    for key, counter in note_stats.items():
        total = totals[key] if totals[key] > 0 else 1

        notes_probs = [
            (note, (count / total) * 100)
            for note, count in counter.items()
        ]

        notes_probs.sort(key=lambda x: -x[1])
        result[key] = notes_probs

    # =========================
    # РАСЧЁТ РАСПРЕДЕЛЕНИЯ СОСТОЯНИЙ
    # =========================
    total_states = sum(state_counter.values()) or 1

    state_probs = [
        (STATE_NAMES.get(sid, str(sid)), (count / total_states) * 100)
        for sid, count in state_counter.items()
    ]
    state_probs.sort(key=lambda x: -x[1])

    # =========================
    # РАСЧЁТ РАСПРЕДЕЛЕНИЯ ТОНАЛЬНОСТЕЙ
    # =========================
    total_keys = sum(key_distribution.values()) or 1

    key_probs = [
        (k, (v / total_keys) * 100)
        for k, v in key_distribution.items()
    ]
    key_probs.sort(key=lambda x: -x[1])

    # =========================
    # РАСЧЁТ ЭНТРОПИИ
    # =========================
    global_note_counter = collections.Counter()

    for counter in note_stats.values():
        global_note_counter.update(counter)

    notes_entropy = calculate_entropy(global_note_counter)
    states_entropy = calculate_entropy(state_counter)

    return result, state_probs, key_probs, notes_entropy, states_entropy


def print_analysis(result, filename,
                   state_probs=None,
                   key_distribution=None,
                   notes_entropy=None,
                   states_entropy=None):
    """
    Выводит статистику анализа датасета в консоль.

    Args:
        result (dict): распределение нот
        filename (str): имя файла
        state_probs (list, optional): распределение состояний
        key_distribution (list, optional): распределение тональностей
        notes_entropy (float, optional): энтропия нот
        states_entropy (float, optional): энтропия состояний
    """

    print(f"\nФайл: {filename}")

    if not result:
        print("  Нет найденных нот.")
        return

    # =========================
    # ВЫВОД НОТ
    # =========================
    for key in sorted(result.keys()):
        key_name = NOTES[key] if 0 <= key < len(NOTES) else str(key)

        print(f"  Тональность {key} ({key_name}):")
        for note, prob in result[key]:
            print(f"    {note}: {prob:.2f}%")

    # =========================
    # ВЫВОД ТОНАЛЬНОСТЕЙ
    # =========================
    if key_distribution:
        print("  Распределение тональностей:")
        for k, prob in key_distribution:
            key_name = NOTES[k] if 0 <= k < len(NOTES) else str(k)
            print(f"    {key_name}: {prob:.2f}%")

    # =========================
    # ВЫВОД СОСТОЯНИЙ
    # =========================
    if state_probs:
        print("  Состояния сигнала:")
        for name, prob in state_probs:
            print(f"    {name}: {prob:.2f}%")

    # =========================
    # ВЫВОД ЭНТРОПИИ
    # =========================
    if notes_entropy is not None:
        print(f"  Энтропия нот: {notes_entropy:.4f}")

    if states_entropy is not None:
        print(f"  Энтропия состояний: {states_entropy:.4f}")


if __name__ == "__main__":
    
    base_path = os.path.join(os.getcwd(), 'data', 'processed')

    if not os.path.exists(base_path):
        print("Папка data/processed не найдена.")
        sys.exit(1)

    # =========================
    # ОБХОД ВСЕХ JSON-ФАЙЛОВ
    # =========================
    for root, _, files in os.walk(base_path):
        for filename in sorted(files):

            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(root, filename)

            try:
                result, state_probs, key_probs, notes_entropy, states_entropy = analyze_dataset_file(file_path)

                print_analysis(
                    result,
                    os.path.relpath(file_path),
                    state_probs,
                    key_probs,
                    notes_entropy,
                    states_entropy
                )

            except Exception as e:
                print(f"\nОшибка при обработке {file_path}: {e}")