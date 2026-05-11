import os
from pathlib import Path
from datetime import datetime

from preprocessing.encode_rhythm import encode_rhythm
from inference.core.generate_solo import generate_solo
from inference.export.export_solo_txt import export_solo_txt
from inference.export.export_solo_mp3 import export_solo_mp3
from inference.ui.menu_text import *

cls = lambda: os.system('cls' if os.name == 'nt' else 'clear')

EXPORT_FOLDER = Path(__file__).resolve().parents[1] / "data" / "generated"

if __name__ == "__main__":
    while True:
        cls()
        print(main_menu_text)
        choice = input("> ").strip()

        match choice:

            # =========================
            # ГЕНЕРАЦИЯ СОЛО
            # =========================
            case "1":

                # Выбор размера такта
                while True:
                    cls()
                    print(enter_signature_text)
                    signature_choice = input("> ").strip()

                    signature = None
                    match signature_choice:
                        case "1":
                            signature = 4
                        case "2":
                            signature = 3
                        case _:
                            continue
                    break

                # Выбор жанра
                while True:
                    cls()
                    print(enter_genre_text)
                    genre_choice = input("> ").strip()

                    genre = None
                    match genre_choice:
                        case "1":
                            genre = "blues"
                        case "2":
                            genre = "metal"
                        case "3":
                            genre = "rock"
                        case _:
                            continue
                    break

                # Ввод BPM
                while True:
                    cls()
                    print(enter_bpm_text)
                    try:
                        bpm = int(input("> ").strip())
                        if 50 <= bpm <= 200:
                            break
                    except Exception:
                        continue

                # Выбор тональности
                while True:
                    cls()
                    print(enter_key_text)
                    try:
                        key_input = int(input("> ").strip())
                        key = key_input - 1
                        if 0 <= key <= 11:
                            break
                    except Exception:
                        continue

                # Ввод аккордов (ритм-партия)
                while True:
                    cls()
                    try:
                        print(enter_chords_example_text)
                        rhythm_string = input("> ").strip()

                        chords = encode_rhythm(rhythm_string, signature)
                        break

                    except Exception as e:
                        print("Ошибка при обработке аккордов:", e)
                        input("Нажмите Enter...")
                        continue

                # =========================
                # ГЕНЕРАЦИЯ СОЛО МОДЕЛЬЮ
                # =========================

                while True:
                    try:
                        solo_notes = generate_solo(chords, key, genre)
                        break
                    except Exception as e:
                        print("Ошибка генерации соло:", e)
                        input("Нажмите Enter...")
                        continue

                # =========================
                # ЭКСПОРТ В ТЕКСТ И АУДИО
                # =========================

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = EXPORT_FOLDER / f"solo_{genre}_{timestamp}"
                export_dir.mkdir(parents=True, exist_ok=True)

                solo_text = export_solo_txt(solo_notes, signature)
                solo_audio = export_solo_mp3(
                    rhythm_string,
                    solo_notes,
                    signature,
                    key,
                    genre,
                    bpm
                )

                # Сохранение txt
                txt_path = export_dir / "solo_text.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(solo_text)

                # Сохранение mp3
                mp3_path = export_dir / "solo_audio.mp3"
                solo_audio.export(mp3_path, format="mp3", bitrate="128k")

                print("Соло сгенерировано успешно!")
                print("Папка:", export_dir.name)
                input("Нажмите Enter...")

            # =========================
            # ИНФОРМАЦИЯ О ПРОЕКТЕ
            # =========================
            case "2":
                cls()
                print(project_info)
                input("\nНажмите Enter...")

            # =========================
            # ВЫХОД ИЗ ПРОГРАММЫ
            # =========================
            case "3":
                break