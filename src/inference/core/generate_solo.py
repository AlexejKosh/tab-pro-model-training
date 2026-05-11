import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.constants.music_constants import (
    CHROMATIC_SCALE_SIZE,
    GLOBAL_LENGHT_LIMIT,
)


MODELS_PATH = "models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHORD_DIM = CHROMATIC_SCALE_SIZE
POS_DIM = 3
INPUT_DIM = CHORD_DIM * 2 + POS_DIM

NOTE_CLASSES = 48
STATE_CLASSES = 7

NOTE_PAD = NOTE_CLASSES
NOTE_BOS = NOTE_CLASSES + 1
NOTE_VOCAB = NOTE_CLASSES + 2

STATE_PAD = STATE_CLASSES
STATE_BOS = STATE_CLASSES + 1
STATE_VOCAB = STATE_CLASSES + 2

D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.1

EMB_SIZE = 32

# Количество последних шагов, помечаемых как конец соло
END_ZONE = 48

# Размер окна генерации
MAX_LEN = 192

# Размер перекрытия между окнами
OVERLAP_LEN = 48

# Шаг смещения окна
WINDOW_STEP = MAX_LEN - OVERLAP_LEN


def build_chord_features(chords, key):
    """
    Формирует входные признаки для модели на основе аккордов и тональности

    Для каждого шага создаются:
    - абсолютное представление аккорда
    - относительное представление аккорда относительно тональности
    - флаги позиции соло (начало, середина, конец)

    Args:
        chords (list): последовательность аккордов
        key (int): тональность композиции

    Returns:
        list: список входных признаков для модели
    """

    total_len = len(chords)

    # Начало зоны завершения соло
    end_start = max(total_len - END_ZONE, 0)

    features = []

    for i, chord in enumerate(chords):

        # Абсолютное представление аккорда
        abs_chord = np.asarray(chord, dtype=np.float32)

        # Сдвиг аккорда относительно тональности
        rel_chord = np.roll(abs_chord, -key)

        # Введения флагов начала соло, его конца и остальной части,
        # чтобы модель понимала границы соло
        start_flag = 1.0 if i == 0 else 0.0
        middle_flag = 1.0 if 0 < i < end_start else 0.0
        end_flag = 1.0 if i >= end_start else 0.0

        features.append(
            np.concatenate([
                abs_chord,
                rel_chord,
                np.array(
                    [start_flag, middle_flag, end_flag],
                    dtype=np.float32
                )
            ])
        )

    return features


class PositionalEncoding(nn.Module):
    """
    Добавляет позиционную информацию к последовательности

    Transformer не понимает порядок элементов сам по себе,
    поэтому позиции кодируются синусами и косинусами
    """

    def __init__(self, d_model, max_len):
        """
        Инициализирует позиционное кодирование

        Args:
            d_model (int): размер скрытого пространства модели
            max_len (int): максимальная длина последовательности
        """

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        """
        Добавляет позиционное кодирование к входным данным

        Args:
            x (torch.Tensor): входной тензор

        Returns:
            torch.Tensor: тензор с добавленным позиционным кодированием
        """

        return x + self.pe[:, :x.size(1)]


class SoloTransformer(nn.Module):
    """
    Transformer-модель для генерации гитарных соло
    """

    def __init__(self):
        """
        Инициализирует архитектуру модели
        """

        super().__init__()

        # Проекция входных признаков аккордов
        self.src_proj = nn.Linear(INPUT_DIM, D_MODEL)

        # Embedding-слои для нот и состояний исполнения
        self.note_emb = nn.Embedding(NOTE_VOCAB, EMB_SIZE, padding_idx=NOTE_PAD)

        self.state_emb = nn.Embedding(STATE_VOCAB, EMB_SIZE, padding_idx=STATE_PAD)

        # Объединение embeddings в единое пространство
        self.decoder_proj = nn.Linear(EMB_SIZE * 4, D_MODEL)

        self.pos_enc = PositionalEncoding(D_MODEL, GLOBAL_LENGHT_LIMIT)

        # Основной Transformer
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )

        # Выходные головы предсказания
        self.note1_head = nn.Linear(D_MODEL, NOTE_CLASSES)
        self.state1_head = nn.Linear(D_MODEL, STATE_CLASSES)

        self.note2_head = nn.Linear(D_MODEL, NOTE_CLASSES)
        self.state2_head = nn.Linear(D_MODEL, STATE_CLASSES)

    def encode(self, features):
        """
        Кодирует входные признаки аккордов

        Args:
            features (torch.Tensor): входные признаки аккордов

        Returns:
            torch.Tensor: закодированное представление последовательности
        """

        return self.pos_enc(self.src_proj(features))

    def decode(self, y):
        """
        Преобразует ранее сгенерированные токены в embeddings

        Args:
            y (torch.Tensor): последовательность ранее сгенерированных нот

        Returns:
            torch.Tensor: decoder embeddings
        """

        n1 = y[:, :, 0]
        s1 = y[:, :, 1]

        n2 = y[:, :, 2]
        s2 = y[:, :, 3]

        emb = torch.cat([
            self.note_emb(n1),
            self.state_emb(s1),
            self.note_emb(n2),
            self.state_emb(s2)
        ], dim=-1)

        emb = self.decoder_proj(emb)

        return self.pos_enc(emb)


def sample(logits, temperature=0.7):
    """
    Выполняет случайный выбор следующего токена
    на основе вероятностного распределения

    Args:
        logits (torch.Tensor): логиты модели
        temperature (float): температура генерации

    Returns:
        int: индекс выбранного токена
    """

    # Защита от деления на ноль
    temperature = max(float(temperature), 1e-6)

    probs = torch.softmax(logits / temperature, dim=-1)

    # Защита от NaN и бесконечностей
    probs = torch.nan_to_num(
        probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Нормализация вероятностей
    probs = probs / probs.sum().clamp_min(1e-8)

    return torch.multinomial(probs, 1).item()


def fix_pause_state(note):
    """
    Удаляет состояния у пауз

    Args:
        note (list): Нота
    """

    if note[0] == 0:
        note[1] = 0

    if note[2] == 0:
        note[3] = 0


def fix_first_note_state(note):
    """
    Исправляет состояние первой ноты соло

    Args:
        note (list): Первая нота соло
    """

    if note[0] != 0 and note[1] in [0, 6]:
        note[1] = 1

    if note[2] != 0 and note[3] in [0, 6]:
        note[3] = 1

    fix_pause_state(note)


def fix_continue_state(curr, prev):
    """
    Исправляет состояния продолжения нот

    Args:
        curr (list): Текущая нота
        prev (list): Предыдущая нота
    """

    # Если нота повторяется — помечаем как продолжение
    if curr[0] == prev[0]:
        curr[1] = 6

    if curr[2] == prev[2]:
        curr[3] = 6

    # Если нота изменилась — убираем продолжение
    if curr[0] != prev[0] and curr[1] in [0, 6]:
        curr[1] = 1

    if curr[2] != prev[2] and curr[3] in [0, 6]:
        curr[3] = 1

    fix_pause_state(curr)


def edit_solo(solo):
    """
    Выполняет постобработку сгенерированного соло

    Args:
        solo (list): последовательность нот соло

    Returns:
        list: исправленная последовательность нот
    """

    solo = [row[:] for row in solo]

    if len(solo) > 0:
        fix_first_note_state(solo[0])

    for i in range(1, len(solo)):
        fix_continue_state(solo[i], solo[i - 1])

    return solo


def generate_window(model, memory, warm_tokens, new_tokens_count, temperature):
    """
    Генерирует одно окно соло с учетом перекрытия предыдущего окна

    Args:
        model (SoloTransformer): модель генерации
        memory (torch.Tensor): encoder-представление окна
        warm_tokens (list): последние токены предыдущего окна
        new_tokens_count (int): количество новых токенов
        temperature (float): температура генерации

    Returns:
        list: сгенерированные токены текущего окна
    """

    # BOS-токен начала последовательности
    start_seq = [[NOTE_BOS, STATE_BOS, NOTE_BOS, STATE_BOS]]

    # Добавляем последние токены предыдущего окна
    start_seq.extend(warm_tokens)

    generated = torch.tensor(start_seq, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(new_tokens_count):

        tgt = model.decode(generated)

        # Маска запрещает смотреть в будущее
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)

        out = model.transformer(src=memory, tgt=tgt, tgt_mask=tgt_mask)

        last = out[:, -1, :]

        next_token = torch.tensor([[
            sample(model.note1_head(last)[0], temperature),
            sample(model.state1_head(last)[0], temperature),
            sample(model.note2_head(last)[0], temperature),
            sample(model.state2_head(last)[0], temperature),
        ]], dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Добавляем новую ноту к последовательности
        generated = torch.cat([generated, next_token], dim=1)

    # Убираем BOS и warm-start часть
    return generated[:, 1 + len(warm_tokens):, :].squeeze(0).cpu().tolist()


def generate_solo(chords, key, genre, temperature=0.7):
    """
    Генерирует гитарное соло по последовательности аккордов

    Args:
        chords (list): последовательность аккордов
        key (int): тональность композиции
        genre (str): название жанра модели
        temperature (float): температура генерации

    Returns:
        list: сгенерированное соло
    """

    model = SoloTransformer().to(DEVICE)

    model.load_state_dict(torch.load(MODELS_PATH + f"{genre}_transformer.pt", map_location=DEVICE))

    model.eval()

    # Формирование входных признаков
    features = build_chord_features(chords, key)

    total_len = len(features)

    if total_len == 0:
        return []

    result = []

    with torch.no_grad():

        start = 0

        # Генерация последовательности по окнам
        while start < total_len:

            end = min(start + MAX_LEN, total_len)

            chunk_features = torch.tensor(
                np.array(features[start:end], dtype=np.float32),
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            chunk_len = end - start

            memory = model.encode(chunk_features)

            # Первое окно генерируется с нуля
            if start == 0:
                warm_tokens = []
                new_tokens_count = chunk_len

            else:
                # Передаем последние токены предыдущего окна
                warm_tokens = result[-OVERLAP_LEN:]

                # Генерируем только недостающую часть окна
                new_tokens_count = chunk_len - len(warm_tokens)

                if new_tokens_count <= 0:
                    break

            window_out = generate_window(
                model=model,
                memory=memory,
                warm_tokens=warm_tokens,
                new_tokens_count=new_tokens_count,
                temperature=temperature
            )

            result.extend(window_out)

            # Сдвигаем окно вперед
            start += WINDOW_STEP

    return edit_solo(result[:total_len])