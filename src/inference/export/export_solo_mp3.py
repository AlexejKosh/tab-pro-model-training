import os
import sys
from pathlib import Path

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from constants.music_constants import (
    NOTES,
    MODE_INTERVALS,
    FLAT_TO_SHARP,
    GENRES,
    ALLOWED_SIGNATURES,
    CHROMATIC_SCALE_SIZE,
    NOTE_INDEX_SHIFT,
    TICKS_PER_BEAT,
    BASE_TIME_SIGNATURE,
    GLOBAL_LENGHT_LIMIT
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

IS_WINDOWS = os.name == 'nt'

# Добавление бинарный файлов в PATH для работы ffmpeg/ffprobe через pydub
os.environ["PATH"] = (
    str(PROJECT_ROOT / "assets" / "bin" / ("windows" if IS_WINDOWS else "unix"))
    + os.pathsep
    + os.environ.get("PATH", "")
)

BIN_PATH = PROJECT_ROOT / "assets" / "bin" / ("windows" if IS_WINDOWS else "unix")

FFMPEG_PATH = BIN_PATH / ("ffmpeg.exe" if IS_WINDOWS else "ffmpeg")
FFPROBE_PATH = BIN_PATH / ("ffprobe.exe" if IS_WINDOWS else "ffprobe")

if not FFMPEG_PATH.exists() or not FFPROBE_PATH.exists():
    raise FileNotFoundError(
        f"ffmpeg или ffprobe не найдены по пути: {PROJECT_ROOT / 'assets' / 'bin'}"
    )

from pydub import AudioSegment

ALLOWED_CHORD_MODES = list(MODE_INTERVALS.keys())
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]

MIN_NOTE_INDEX = 1
MAX_NOTE_INDEX = 47
BASE_OCTAVE = 2

DEFAULT_FADE_OUT_MS = 50

MS_IN_MINUTE = 60000

SOUND_EXTENSION_MS = 100
TECHNIQUE_ATTACK_MS = 100
BEND_START_MS = 80
BEND_SEGMENTS = 10

MIN_BPM = 50
MAX_BPM = 200

AUDIO_TAIL_BUFFER_MS = 1000
NOTE_RELEASE_EXTENSION_MS = 150

audio_cache = {}

def get_root_note_and_mode(chord_name):
    """
    Разбирает аккорд на тонику и тип аккорда.

    Args:
        chord_name (str): строка аккорда (например "C-1/2,G-1,Am-2")

    Returns:
        tuple (root, mode): тоника и тип аккорда

    Raises:
        ValueError: если аккорд некорректный
    """
    chord_name = chord_name.strip()
    if not chord_name:
        raise ValueError("Пустое имя аккорда")

    if len(chord_name) >= 2 and chord_name[:2] in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[chord_name[:2]]
        mode = chord_name[2:]
    elif len(chord_name) >= 2 and chord_name[:2] in NOTES:
        root = chord_name[:2]
        mode = chord_name[2:]
    elif chord_name[0] in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[chord_name[0]]
        mode = chord_name[1:]
    elif chord_name[0] in NOTES:
        root = chord_name[0]
        mode = chord_name[1:]
    else:
        raise ValueError(f"Некорректное имя аккорда: {chord_name}")

    if mode not in ALLOWED_CHORD_MODES:
        raise ValueError(f"Недопустимый тип аккорда: {mode}")

    return root, mode


def get_note_name_from_index(idx):
    """
    Преобразует числовой индекс ноты в её строковое представление.

    Args:
        idx (int): индекс ноты

    Returns:
        str: название ноты (например E2)

    Raises:
        ValueError: если индекс вне диапазона
    """

    if idx < MIN_NOTE_INDEX or idx > MAX_NOTE_INDEX:
        raise ValueError(f"Индекс ноты вне диапазона [{MIN_NOTE_INDEX}, {MAX_NOTE_INDEX}]")

    octave = BASE_OCTAVE + (idx + NOTE_INDEX_SHIFT) // CHROMATIC_SCALE_SIZE
    name = NOTES[(idx + NOTE_INDEX_SHIFT) % CHROMATIC_SCALE_SIZE]
    return f"{name}{octave}"


def get_scale_pitch_classes(key):
    """
    Возвращает набор нот мажорной гаммы для заданной тональности.

    Args:
        key (int): тональность

    Returns:
        list: список индексов нот гаммы
    """
    return [(key + interval) % CHROMATIC_SCALE_SIZE for interval in MAJOR_SCALE_INTERVALS]


def get_lower_scale_index(idx, scale):
    """
    Находит ближайшую нижнюю ноту в гамме.

    Args:
        idx (int): текущий индекс
        scale (list): гамма

    Returns:
        int: индекс ноты
    """

    if idx <= MIN_NOTE_INDEX:
        return MIN_NOTE_INDEX

    for candidate in range(idx - 1, idx - 3, -1):
        if (candidate + NOTE_INDEX_SHIFT) % CHROMATIC_SCALE_SIZE in scale:
            return candidate

    return max(MIN_NOTE_INDEX, idx - 1)


def get_upper_scale_index(idx, scale):
    """
    Находит ближайшую верхнюю ноту в гамме.

    Args:
        idx (int): текущий индекс
        scale (list): гамма

    Returns:
        int: индекс ноты
    """

    if idx >= MAX_NOTE_INDEX:
        return MAX_NOTE_INDEX

    for candidate in range(idx + 1, idx + 3, -1):
        if (candidate + NOTE_INDEX_SHIFT) % CHROMATIC_SCALE_SIZE in scale:
            return candidate

    return min(MAX_NOTE_INDEX, idx + 1)

def pitch_shift(sound, semitones):
    """
    Сдвигает высоту звука без изменения длительности.

    Args:
        sound (AudioSegment): аудиосегмент
        semitones (int): сдвиг в полутонах

    Returns:
        AudioSegment: обработанный звук
    """

    if semitones == 0:
        return sound

    new_frame_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
    return shifted.set_frame_rate(sound.frame_rate)


def apply_fade_out(sound):
    """
    Применяет fade-out к аудио.

    Args:
        sound (AudioSegment): аудиосегмент

    Returns:
        AudioSegment: обработанный звук
    """

    if len(sound) <= DEFAULT_FADE_OUT_MS:
        return sound.fade_out(len(sound))

    return sound.fade_out(DEFAULT_FADE_OUT_MS)


def load_genre_samples(genre):
    """
    Загружает аудио-файлы для выбранного жанра.

    Args:
        genre (str): жанр музыки

    Returns:
        tuple: (rhythm_samples, solo_samples): набор звуков для ритм- и соло-гитары

    Raises:
        FileNotFoundError: если папки не существуют
    """

    rhythm_dir = PROJECT_ROOT / "assets" / "audio" / genre / "rhythm"
    solo_dir = PROJECT_ROOT / "assets" / "audio" / genre / "solo"

    if not rhythm_dir.exists():
        raise FileNotFoundError(f"Не найдена папка: {rhythm_dir}")
    if not solo_dir.exists():
        raise FileNotFoundError(f"Не найдена папка: {solo_dir}")

    rhythm_samples = {}
    solo_samples = {}

    for file_name in os.listdir(rhythm_dir):
        if file_name.lower().endswith(".mp3"):
            rhythm_samples[file_name[:-4]] = AudioSegment.from_mp3(rhythm_dir / file_name)

    for file_name in os.listdir(solo_dir):
        if file_name.lower().endswith(".mp3"):
            solo_samples[file_name[:-4]] = AudioSegment.from_mp3(solo_dir / file_name)

    return rhythm_samples, solo_samples

def validate_rhythm_string(rhythm_string):
    """
    Проверяет строку аккордов и преобразует её в структуру данных.

    Args:
        rhythm_string (str): строка аккордов

    Returns:
        items (list): список (аккорд, длительность)

    Raises:
        ValueError: если формат некорректный
    """

    if not isinstance(rhythm_string, str) or not rhythm_string.strip():
        raise ValueError("rhythm_string должен быть непустой строкой")

    items = []

    # Разбиваем строку на отдельные аккорды и чистим пробелы
    chunks = [chunk.strip() for chunk in rhythm_string.split(",") if chunk.strip()]

    if not chunks:
        raise ValueError("Список аккордов пуст")

    for chunk in chunks:

        if "-" not in chunk:
            raise ValueError(f"Некорректный формат аккорда: {chunk}")

        chord_name, duration_text = chunk.split("-", 1)
        chord_name = chord_name.strip()
        duration_text = duration_text.strip()

        # Проверка корректности именования аккорда 
        get_root_note_and_mode(chord_name)

        if "/" in duration_text:
            # Дробная длительность (например 1/2)
            parts = duration_text.split("/", 1)

            if len(parts) != 2:
                raise ValueError(f"Некорректный формат длительности: {duration_text}")

            numerator, denominator = parts

            if not numerator.isdigit() or not denominator.isdigit():
                raise ValueError(f"Длительность должна содержать целые числа: {duration_text}")

            numerator = int(numerator)
            denominator = int(denominator)

            if numerator <= 0 or denominator <= 0:
                raise ValueError(f"Длительность должна быть положительной: {duration_text}")

            duration = numerator / denominator

        else:
            # Целая длительность (например 1, 2, 4)
            if not duration_text.isdigit():
                raise ValueError(f"Некорректная длительность: {duration_text}")

            duration = int(duration_text)

        if duration <= 0:
            raise ValueError(f"Некорректная длительность аккорда: {chunk}")

        items.append((chord_name, duration))

    return items


def validate_solo_list(solo_list, rhythm_length):
    """
    Проверяет корректность соло последовательности.

    Args:
        solo_list (list): список событий соло
        rhythm_length (int): ожидаемая длина

    Raises:
        ValueError: при ошибке в данных
    """

    if not isinstance(solo_list, list):
        raise ValueError("solo_list должен быть списком")
    if len(solo_list) != rhythm_length:
        raise ValueError(f"Длина solo_list должна совпадать с rhythm: {len(solo_list)} != {rhythm_length}")
    # Жёсткое ограничение длины последовательности (защита от перегруза памяти)
    if len(solo_list) > GLOBAL_LENGHT_LIMIT:
        raise ValueError(f"Длина solo_list не должна быть больше {GLOBAL_LENGHT_LIMIT} шагов")

    last_pitch_1 = 0
    last_pitch_2 = 0

    for row in solo_list:
        # Проверка структуры шага (обязательно 4 значения)
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("Каждая строка solo_list должна быть списком из 4 чисел")

        pitch_1, state_1, pitch_2, state_2 = row

        for value in [pitch_1, state_1, pitch_2, state_2]:
            if not isinstance(value, int):
                raise ValueError("Все значения solo_list должны быть целыми числами")
            
        if pitch_1 < 0 or pitch_1 > 47 or pitch_2 < 0 or pitch_2 > 47:
            raise ValueError("Высота ноты должна быть в диапазоне [0, 47]")
        if state_1 < 0 or state_1 > 6 or state_2 < 0 or state_2 > 6:
            raise ValueError("Состояние ноты должно быть в диапазоне [0, 6]")

        # Проверка на связанность pitch и state (либо оба 0, либо оба активны)
        if (pitch_1 == 0 and state_1 != 0) or (pitch_1 != 0 and state_1 == 0):
            raise ValueError("Если pitch равен 0, state тоже должен быть 0, и наоборот")
        if (pitch_2 == 0 and state_2 != 0) or (pitch_2!= 0 and state_2 == 0):
            raise ValueError("Если pitch равен 0, state тоже должен быть 0, и наоборот")

        # Проверка на то, что state = 6 (сустейн) не начинает ноту
        if state_1 == 6:
            if last_pitch_1 == 0 or pitch_1 != last_pitch_1:
                raise ValueError("Нота не может начинаться со state 6")
        if state_2 == 6:
            if last_pitch_2 == 0 or pitch_2 != last_pitch_2:
                raise ValueError("Нота не может начинаться со state 6")

        # Обновление состояния последней сыгранной ноты
        if pitch_1 != 0 and state_1 != 6:
            last_pitch_1 = pitch_1
        if pitch_2 != 0 and state_2 != 6:
            last_pitch_2 = pitch_2

def make_chord_clip(chord_name, bars_duration, signature, bpm, rhythm_samples):
    """
    Формирует аудио аккорда с учетом длительности, темпа и размера такта.

    Args:
        chord_name (str): название аккорда
        bars_duration (float): длительность аккорда в долях такта
        signature (int): размер такта (3 или 4)
        bpm (int): темп
        rhythm_samples (dict): аудио ритм-гитары

    Returns:
        AudioSegment: готовый аудиоклип аккорда

    Raises:
        FileNotFoundError: если аккорд отсутствует в базе
    """

    if chord_name not in rhythm_samples:
        raise FileNotFoundError(f"Нет аудио для аккорда: {chord_name}")

    chord_audio = rhythm_samples[chord_name]

    target_duration_ms = int(round(bars_duration * signature * MS_IN_MINUTE / bpm))
    extended_duration_ms = target_duration_ms + SOUND_EXTENSION_MS

    # Обрезка аудио до расчётной длительности
    if len(chord_audio) > extended_duration_ms:
        chord_clip = chord_audio[:extended_duration_ms]
    else:
        chord_clip = chord_audio

    return apply_fade_out(chord_clip)

def build_technique_clip(sound, state, pitch_index, key):
    """
    Применяет исполнительскую технику (hammer-on, pull-off, bend) к ноте.

    Args:
        sound (AudioSegment): исходный звук ноты
        state (int): состояние ноты (1–5)
        pitch_index (int): индекс ноты
        key (int): тональность

    Returns:
        AudioSegment: обработанный звук с техникой исполнения
    """

    # Обычные атаки и слайды не изменяются
    if state in [1, 2]:
        return sound

    scale = get_scale_pitch_classes(key)

    # =========================
    # HAMMER-ON
    # =========================
    if state == 3:
        lower_index = get_lower_scale_index(pitch_index, scale)
        lower_name = get_note_name_from_index(lower_index)

        if lower_name not in audio_cache:
            raise FileNotFoundError(f"Нет аудио для ноты: {lower_name}")

        start_sound = audio_cache[lower_name]

        first_part = start_sound[:TECHNIQUE_ATTACK_MS] if len(start_sound) > TECHNIQUE_ATTACK_MS else start_sound
        second_part = sound[len(first_part):] if len(sound) > len(first_part) else AudioSegment.silent(duration=0)

        return first_part + second_part

    # =========================
    # PULL-OFF
    # =========================
    if state == 4:
        upper_index = get_upper_scale_index(pitch_index, scale)
        upper_name = get_note_name_from_index(upper_index)

        if upper_name not in audio_cache:
            raise FileNotFoundError(f"Нет аудио для ноты: {upper_name}")

        start_sound = audio_cache[upper_name]

        first_part = start_sound[:TECHNIQUE_ATTACK_MS] if len(start_sound) > TECHNIQUE_ATTACK_MS else start_sound
        second_part = sound[len(first_part):] if len(sound) > len(first_part) else AudioSegment.silent(duration=0)

        return first_part + second_part

    # =========================
    # BEND (ПЛАВНОЕ ИЗМЕНЕНИЕ ВЫСОТЫ)
    # =========================
    if state == 5:
        upper_index = get_upper_scale_index(pitch_index, scale)
        semitones = upper_index - pitch_index

        first_part = sound[:BEND_START_MS] if len(sound) > BEND_START_MS else sound
        rest = sound[BEND_START_MS:]

        if len(rest) == 0:
            return first_part

        piece_len = max(1, len(rest) // BEND_SEGMENTS)

        pieces = [first_part]

        for i in range(BEND_SEGMENTS):
            begin = i * piece_len
            if begin >= len(rest):
                break

            end = min(len(rest), begin + piece_len)
            chunk = rest[begin:end]

            # Постепенный рост питча (имитация бенда)
            current_shift = semitones * ((i + 1) / BEND_SEGMENTS)

            pieces.append(pitch_shift(chunk, current_shift))

        result = pieces[0]
        for piece in pieces[1:]:
            result += piece

        return result

    return sound


# =========================
# ЭКСПОРТ АУДИО СОЛО
# =========================

def export_solo_mp3(rhythm_string, solo_list, signature, key, genre, bpm):
    """
    Генерирует итоговый mp3 файл соло-гитары с наложением ритма и техник исполнения.

    Args:
        rhythm_string (str): строка аккордов
        solo_list (list): закодированная последовательность соло
        signature (int): размер такта
        key (int | str): тональность
        genre (str): жанр музыки
        bpm (int): темп

    Returns:
        AudioSegment: сформированный аудиофайл

    Raises:
        ValueError: при некорректных входных данных
    """

    for k, v in FLAT_TO_SHARP.items():
        if k in rhythm_string:
            rhythm_string = rhythm_string.replace(k, v)

    # =========================
    # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
    # =========================

    if genre not in GENRES:
        raise ValueError(f"genre должен быть одним из {GENRES}")

    if signature not in ALLOWED_SIGNATURES:
        raise ValueError("signature должен быть 3 или 4")

    if not isinstance(bpm, int) or bpm < MIN_BPM or bpm > MAX_BPM:
        raise ValueError(f"bpm должен быть целым числом от {MIN_BPM} до {MAX_BPM}")

    if not isinstance(key, int) or key < 0 or key > len(NOTES) - 1:
        raise ValueError(f"key должен быть числом от 0 до {len(NOTES)-1}")

    # =========================
    # ПРОВЕРКА ДАННЫХ СОЛО И РИТМА
    # =========================

    rhythm_items = validate_rhythm_string(rhythm_string)

    expected_length = sum(
        int(round(duration * TICKS_PER_BEAT / BASE_TIME_SIGNATURE * signature))
        for _, duration in rhythm_items
    )

    validate_solo_list(solo_list, expected_length)

    # =========================
    # ЗАГРУЗКА АУДИО-ФАЙЛОВ
    # =========================

    rhythm_samples, solo_samples = load_genre_samples(genre)

    audio_cache.clear()
    audio_cache.update(solo_samples)

    # =========================
    # ФОРМИРОВАНИЕ ОСНОВНОЙ ДОРОЖКИ
    # =========================

    total_ms = 0
    for _, duration in rhythm_items:
        total_ms += int(round(duration * signature * MS_IN_MINUTE / bpm))

    result_audio = AudioSegment.silent(duration=total_ms + AUDIO_TAIL_BUFFER_MS)

    # =========================
    # НАЛОЖЕНИЕ РИТМ-ГИТАРЫ
    # =========================

    current_ms = 0

    for chord_name, duration in rhythm_items:
        clip = make_chord_clip(chord_name, duration, signature, bpm, rhythm_samples)

        result_audio = result_audio.overlay(clip, position=current_ms)

        current_ms += int(round(duration * signature * MS_IN_MINUTE / bpm))

    # Баланс громкости ритм-гитары для жанра 'рок'
    if genre == "rock":
        result_audio = result_audio + 6

    # =========================
    # НАЛОЖЕНИЕ СОЛО (ДВЕ ГОЛОСА)
    # =========================

    step_ms = MS_IN_MINUTE / bpm / TICKS_PER_BEAT * 4

    # Хранение состояния двух независимых голосов
    notes = [
        {"current": None, "start": 0},
        {"current": None, "start": 0}
    ]

    # Добавление фиктивного нулевого шаг для завершения последней ноты
    for i, row in enumerate(solo_list + [[0, 0, 0, 0]]):

        # Разбиение строки на две независимые ноты (два голоса)
        pairs = [(row[0], row[1]), (row[2], row[3])]

        for idx in range(2):
            pitch, state = pairs[idx]
            note = notes[idx]

            # =========================
            # СТАРТ НОТЫ
            # =========================
            if note["current"] is None:
                if pitch != 0 and state != 0:
                    note["current"] = (pitch, state)
                    note["start"] = i

            else:
                cur_pitch, cur_state = note["current"]

                # =========================
                # ЗАВЕРШЕНИЕ НОТЫ
                # =========================
                if pitch == 0 or state == 0 or pitch != cur_pitch:
                    duration_steps = i - note["start"]

                    duration_ms = int(
                        round(duration_steps * step_ms + NOTE_RELEASE_EXTENSION_MS)
                    )

                    note_name = get_note_name_from_index(cur_pitch)

                    if note_name not in audio_cache:
                        raise FileNotFoundError(f"Нет аудио для ноты: {note_name}")

                    clip = audio_cache[note_name]

                    if len(clip) > duration_ms:
                        clip = clip[:duration_ms]

                    clip = build_technique_clip(clip, cur_state, cur_pitch, key)
                    clip = apply_fade_out(clip)

                    result_audio = result_audio.overlay(
                        clip,
                        position=int(round(note["start"] * step_ms))
                    )

                    note["current"] = None

                # =========================
                # СМЕНА ТЕХНИКИ ИСПОЛНЕНИЯ
                # =========================
                elif state != 6 and state != cur_state and state != 0:
                    duration_steps = i - note["start"]

                    duration_ms = int(
                        round(duration_steps * step_ms + SOUND_EXTENSION_MS)
                    )

                    note_name = get_note_name_from_index(cur_pitch)

                    clip = audio_cache[note_name]

                    if len(clip) > duration_ms:
                        clip = clip[:duration_ms]

                    clip = build_technique_clip(clip, cur_state, cur_pitch, key)
                    clip = apply_fade_out(clip)

                    result_audio = result_audio.overlay(
                        clip,
                        position=int(round(note["start"] * step_ms))
                    )

                    note["current"] = (pitch, state)
                    note["start"] = i

    return result_audio