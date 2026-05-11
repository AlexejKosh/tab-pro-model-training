import os
import sys
import csv
import json
import guitarpro
from pathlib import Path

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants.music_constants import NOTES, GENRES, CHROMATIC_SCALE_SIZE, BASE_TIME_SIGNATURE
from preprocessing.encode_solo import encode_solo
from preprocessing.encode_rhythm import encode_rhythm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GP_TICKS_BAR = 3840
LOWEST_NOTE_IN_MIDI = 39
HIGH_LIMIT = 45
LOW_LIMIT = 1
 

if __name__ == "__main__":

    # Общий контейнер для всех обработанных песен одного жанра
    encoded_songs = []

    # =========================
    # ОБРАБОТКА ЖАНРОВ
    # =========================
    for genre in GENRES:

        try:
            genre_path = PROJECT_ROOT / "data" / "raw" / genre

            if not os.path.exists(str(genre_path)):
                print(f"Папка не найдена: {genre_path}")
                continue

            errors = []
            songs = []

            # =========================
            # ЗАГРУЗКА CSV ИНСТРУКЦИЙ
            # =========================
            with open(f"{genre_path}/{genre}_instructions.csv", newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Пропуск заголовка

                for row in reader:
                    songs.append(row)

        except Exception as e:
            print(f"Ошибка при загрузке инструкций для жанра {genre}: {e}")
            continue

        # =========================
        # ОБРАБОТКА ПЕСЕН
        # =========================
        for song_info in songs:
            try:
                song_path = os.path.join(genre_path, song_info[0], song_info[1])
                solo_track = int(song_info[2]) - 1
                interval = [int(song_info[3]), int(song_info[4])]
                signature = int(song_info[5])

                # Длина такта в тиках GuitarPro
                tick = GP_TICKS_BAR * signature / BASE_TIME_SIGNATURE

                song = guitarpro.parse(song_path)

                # =========================
                # НОРМАЛИЗАЦИЯ ВРЕМЕНИ
                # =========================
                shift_interval = min(
                    track.measures[interval[0] - 1].voices[0].beats[0].start
                    for track in song.tracks
                ) - (interval[0] - 1) * tick

                track = song.tracks[solo_track]
                strings = track.strings

                # =========================
                # ПРОВЕРКА СТРОЯ ГИТАРЫ
                # =========================
                if not (
                    strings[5].value - strings[4].value == -5 and
                    strings[4].value - strings[3].value == -5 and
                    strings[3].value - strings[2].value == -5 and
                    strings[2].value - strings[1].value == -4 and
                    strings[1].value - strings[0].value == -5
                ):
                    raise Exception(
                        f"Нестандартный строй: {', '.join(str(s.value) for s in strings)}"
                    )

                # Смещение нот относительно стандарта
                tune_diffecence = LOWEST_NOTE_IN_MIDI + 1 - strings[5].value

                # Определение тональности
                key = (NOTES.index(song_info[6]) + tune_diffecence) % CHROMATIC_SCALE_SIZE

                # =========================
                # КОДИРОВАНИЕ РИТМА
                # =========================
                encoded_rhythm = encode_rhythm(
                    song_info[7],
                    signature,
                    tune_diffecence
                )

                # Набор событий, который будет иметь вид
                # [длительность_нот, информация_о_ноте_1, информация_о_ноте_2]
                raw_solo_data = []
                # Диапазон высоты нот в песне, для дальшего вычисления
                # максимальной тпранспозиции
                max_note, min_note = None, None

                # =========================
                # ИЗВЛЕЧЕНИЕ СОЛО
                # =========================

                # Цикл по заданному диапозону тактов
                for bar in range(interval[0] - 1, interval[1]):
                    # Цикл по всем голосам выбранной соло-дорожки
                    for voice in track.measures[bar].voices:
                        # Цикл по всем событиям в данном такте
                        for beat in voice.beats:

                            start_time = round(
                                (beat.start - shift_interval) / tick - (interval[0] - 1),
                                5
                            )

                            # Защита от дубликатов (баг GuitarPro)
                            if start_time in [rec[0] for rec in raw_solo_data]:
                                continue

                            # Обработка максимум 2 нот
                            if len(beat.notes) >= 2:
                                note_1, note_2 = beat.notes[-1], beat.notes[0]
                            elif len(beat.notes) == 1:
                                note_1, note_2 = beat.notes[0], None
                            else:
                                note_1 = note_2 = None

                            def note_to_list(note):
                                """
                                Преобразует ноту GuitarPro в компактный формат.

                                Args:
                                    note (guitarpro.models.Note): объект ноты GuitarPro

                                Returns:
                                    (list): [pitch, state / effect, type, durationPercent]
                                """
                                string_value = note.string
                                return [
                                    strings[string_value - 1].value + note.value + tune_diffecence - LOWEST_NOTE_IN_MIDI,
                                    note.effect,
                                    note.type,
                                    note.durationPercent
                                ]

                            note_1 = note_to_list(note_1) if note_1 else None
                            note_2 = note_to_list(note_2) if note_2 else None

                            # Обновление диапазона нот
                            for note in (note_1, note_2):
                                if note is not None:
                                    value = note[0]

                                    if max_note is None or value > max_note:
                                        max_note = value
                                    if min_note is None or value < min_note:
                                        min_note = value

                            raw_solo_data.append([start_time, note_1, note_2])

                # =========================
                # ПЕРЕВОД В ДЛИТЕЛЬНОСТИ
                # =========================
                for i in range(len(raw_solo_data)):
                    # Вычисление длительности не последнего события на дорожке
                    if i != len(raw_solo_data) - 1:
                        raw_solo_data[i][0] = round(
                            raw_solo_data[i + 1][0] - raw_solo_data[i][0], 5
                        )
                    # Вычисление длительности последнего события на дорожке
                    else:
                        raw_solo_data[i][0] = round(
                            (interval[1] - interval[0] + 1) - raw_solo_data[i][0], 5
                        )

                # =========================
                # КОДИРОВАНИЕ СОЛО
                # =========================
                encoded_solo = encode_solo(raw_solo_data, signature)

                # Проверка синхронности по длиннам
                if len(encoded_rhythm) != len(encoded_solo):
                    raise Exception(
                        f"Размеры rhythm и solo не совпадают: "
                        f"{len(encoded_rhythm)} и {len(encoded_solo)}"
                    )

                # =========================
                # DATA AUGMENTATION (ТРАНСПОЗИЦИЯ)
                # =========================

                # Расширение датасета за счёт сдвига (транспонирования) всех нот вверх и вниз
                # в пределах допустимого диапазона инструмента
                pitch_limit = 11    # Максимум сдвига (в полутонах)
                pitch_to_high = 0   # Насколько можно поднять
                pitch_to_low = 0    # Насколько можно опустить

                # Определение допустимых границ транспонирования
                while True:
                    if max_note + pitch_to_high < HIGH_LIMIT and pitch_to_high < pitch_limit:
                        pitch_to_high += 1
                    elif min_note + pitch_to_low > LOW_LIMIT and abs(pitch_to_low) < pitch_limit - pitch_to_high:
                        pitch_to_low -= 1
                    else:
                        break

                # Применение транспозиции в найденном диапазоне
                for pitch in range(pitch_to_low, pitch_to_high + 1):

                    if pitch != 0:
                        # Сдвиг тональности
                        transposed_key = (key + pitch) % CHROMATIC_SCALE_SIZE

                        # Сдвиг ритм-гитары
                        transposed_rhythm = [
                            row[-pitch:] + row[:-pitch]
                            for row in encoded_rhythm
                        ]

                        # Сдвиг нот соло-гитары
                        transposed_solo = []
                        for note in encoded_solo:
                            transposed_solo.append([
                                note[0] + pitch if note[0] != 0 else 0,
                                note[1],
                                note[2] + pitch if note[2] != 0 else 0,
                                note[3]
                            ])

                        encoded_songs.append({
                            "key": transposed_key,
                            "rhythm": transposed_rhythm,
                            "solo": transposed_solo
                        })

                    else:
                        # Оригинальные данные без транспозиции
                        encoded_songs.append({
                            "key": key,
                            "rhythm": encoded_rhythm,
                            "solo": encoded_solo
                        })

                print(f"Успешно обработано: {song_info[1]}")
                print(f"Индекс тональности: {key}; диапазон нот: {min_note}-{max_note}; транспонирование вниз/вверх: {pitch_to_low}, {pitch_to_high}")

            except Exception as e:
                errors.append(f"{song_info[1]} - {e}")

        # =========================
        # ЛОГ ОШИБОК
        # =========================
        if errors:
            print("\nОшибки обработки:")
            for er in errors:
                print(er)

        # Сортировка по тональности
        encoded_songs.sort(key=lambda x: x['key'])

        # =========================
        # СОХРАНЕНИЕ ДАТАСЕТА
        # =========================
        total_subdivisions = sum(len(song['rhythm']) for song in encoded_songs)
        output_path = f"data/processed/{genre}_dataset.json"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(encoded_songs, f, ensure_ascii=False, indent=4)

            print(
                f"\nДатасет '{genre}' сохранён\n"
                f"Путь: {output_path}\n"
                f"Размер: {total_subdivisions}\n"
            )

        except Exception as e:
            print(f"\nОшибка сохранения '{genre}': {e}\n")

        # Очистка перед следующим жанром
        encoded_songs.clear()