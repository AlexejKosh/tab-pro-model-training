import json
import math
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Добавление корня проекта в PYTHONPATH (для корректных импортов)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants.music_constants import (
    TICKS_PER_BEAT,
    CHROMATIC_SCALE_SIZE,
    GLOBAL_LENGHT_LIMIT
)

GENRE = "rock"

DATA_PATH = f"data/processed/{GENRE}_dataset.json"
MODEL_FOLDER = "models/"

BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Максимальная длина окна обучения
MAX_LEN = TICKS_PER_BEAT * 4

# Шаг смещения скользящего окна
STRIDE = 12

# Размер бинарного массива аккорда
CHORD_DIM = CHROMATIC_SCALE_SIZE

# Размер массива фаз соло:
# [начало, середина, конец]
POS_DIM = 3

# Полный размер входных признаков:
# абсолютный аккорд + относительный аккорд + фазы
INPUT_DIM = CHORD_DIM * 2 + POS_DIM

# Количество классов нот
NOTE_CLASSES = 48

# Количество классов состояний/эффектов
STATE_CLASSES = 7

# Специальные токены для нот
NOTE_PAD = NOTE_CLASSES
NOTE_BOS = NOTE_CLASSES + 1
NOTE_VOCAB = NOTE_CLASSES + 2

# Специальные токены для состояний
STATE_PAD = STATE_CLASSES
STATE_BOS = STATE_CLASSES + 1
STATE_VOCAB = STATE_CLASSES + 2

# Размер скрытого пространства Transformer
D_MODEL = 256

# Количество attention-head
NHEAD = 4

# Количество encoder/decoder слоев
NUM_LAYERS = 3

# Dropout для регуляризации
DROPOUT = 0.1

# Размер эмбеддингов нот и состояний
EMB_SIZE = 32

# Количество шагов до конца соло,
# которое считается "зоной завершения"
END_ZONE = 48


def build_chord_features(chords, key):
    """
    Формирует расширенные признаки аккордов для модели

    Для каждого шага создаются:
    - абсолютный аккорд
    - аккорд относительно тональности
    - флаги положения внутри соло

    Args:
        chords (list): последовательность аккордов
        key (int): тональность композиции

    Returns:
        list: список объединенных признаков аккордов
    """

    total_len = len(chords)

    # Начало зоны окончания соло
    end_start = max(total_len - END_ZONE, 0)

    features = []

    for i, chord in enumerate(chords):

        # Абсолютное представление аккорда
        abs_chord = np.asarray(chord, dtype=np.float32)

        # Представление аккорда относительно тональности
        rel_chord = np.roll(abs_chord, -key)

        # Введение флагов начала, середины и конца соло,
        # чтобы модель понимала структуру композиции
        start_flag = 1.0 if i == 0 else 0.0
        middle_flag = 1.0 if 0 < i < end_start else 0.0
        end_flag = 1.0 if i >= end_start else 0.0

        # Объединение всех признаков в один вектор
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


class GuitarDataset(Dataset):
    """
    Dataset для обучения Transformer-модели генерации соло

    Разбивает композиции на скользящие окна,
    подготавливает teacher forcing и pad-маски
    """

    def __init__(self, dataset, max_len, stride):

        self.samples = []
        self.max_len = max_len

        for item in dataset:

            key = item["key"]
            chords = item["rhythm"]
            solo = item["solo"]

            # Выравнивание длин ритма и соло
            total_len = min(len(chords), len(solo))

            chords = chords[:total_len]
            solo = solo[:total_len]

            # Построение признаков аккордов
            features = build_chord_features(chords, key)

            # Если композиция помещается в одно окно
            if total_len <= max_len:
                self.samples.append({"features": features, "solo": solo})
                continue

            # =========================
            # РАЗБИЕНИЕ НА СКОЛЬЗЯЩИЕ ОКНА
            # =========================
            start = 0

            while start < total_len:

                end = start + max_len

                chunk_features = features[start:end]
                chunk_solo = solo[start:end]

                # Не берем слишком короткие окна
                if len(chunk_features) < 32:
                    break

                self.samples.append({"features": chunk_features, "solo": chunk_solo})

                start += stride

    def __len__(self):
        """
        Возвращает количество обучающих окон

        Returns:
            int: количество примеров
        """
        return len(self.samples)

    def pad_features(self, seq):
        """
        Дополняет признаки аккордов до фиксированной длины

        Args:
            seq (list): последовательность признаков

        Returns:
            list: дополненная последовательность
        """

        pad_len = self.max_len - len(seq)

        if pad_len <= 0:
            return seq[:self.max_len]

        return seq + [[0.0] * INPUT_DIM for _ in range(pad_len)]

    def pad_solo(self, seq):
        """
        Дополняет последовательность соло до фиксированной длины

        Args:
            seq (list): последовательность нот

        Returns:
            list: дополненная последовательность
        """

        pad_len = self.max_len - len(seq)

        if pad_len <= 0:
            return seq[:self.max_len]

        return seq + [[0, 0, 0, 0] for _ in range(pad_len)]

    def __getitem__(self, idx):
        """
        Возвращает подготовленный пример для обучения

        Args:
            idx (int): индекс примера

        Returns:
            dict: подготовленные данные для модели
        """

        item = self.samples[idx]

        features = item["features"]
        solo = item["solo"]

        seq_len = min(len(features), self.max_len)

        # Дополнение последовательностей до MAX_LEN
        features = self.pad_features(features)
        solo = self.pad_solo(solo)

        # Преобразование в tensor
        features = torch.tensor(np.array(features), dtype=torch.float32)

        solo = torch.tensor(solo, dtype=torch.long)

        # =========================
        # TEACHER FORCING
        # =========================

        # Создание входной последовательности decoder
        y_in = torch.full((self.max_len, 4), fill_value=0, dtype=torch.long)

        # Заполнение PAD-токенами
        y_in[:, 0] = NOTE_PAD
        y_in[:, 1] = STATE_PAD
        y_in[:, 2] = NOTE_PAD
        y_in[:, 3] = STATE_PAD

        # Вставка BOS-токенов в начало последовательности
        if seq_len > 0:
            y_in[0, 0] = NOTE_BOS
            y_in[0, 1] = STATE_BOS
            y_in[0, 2] = NOTE_BOS
            y_in[0, 3] = STATE_BOS

        # Сдвиг последовательности на 1 шаг вправо
        # чтобы модель училась предсказывать следующую ноту
        if seq_len > 1:
            y_in[1:seq_len] = solo[:seq_len - 1]

        # Оригинальная последовательность для target
        y_out = solo.clone()

        # =========================
        # СОЗДАНИЕ PAD-МАСКИ
        # =========================

        # True = PAD-область
        # False = реальные данные
        pad_mask = torch.ones(self.max_len, dtype=torch.bool)

        pad_mask[:seq_len] = False

        return {
            "features": features,
            "y_in": y_in,
            "y_out": y_out,
            "pad_mask": pad_mask
        }


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    """
    Добавляет позиционную информацию к последовательности

    Transformer не понимает порядок элементов сам по себе,
    поэтому используется синусоидальное позиционное кодирование
    """

    def __init__(self, d_model, max_len):
        """
        Создает таблицу позиционных кодировок

        Args:
            d_model (int): размер скрытого пространства
            max_len (int): максимальная длина последовательности
        """

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Сохраняем как buffer, а не как обучаемый параметр
        # чтобы не обновлять во время обучения
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        """
        Добавляет позиционное кодирование

        Args:
            x (torch.Tensor): входная последовательность

        Returns:
            torch.Tensor: последовательность с позиционным кодированием
        """

        return x + self.pe[:, :x.size(1)]


class SoloTransformer(nn.Module):
    """
    Transformer-модель генерации гитарных соло
    """

    def __init__(self):

        super().__init__()

        # Проекция входных признаков аккордов
        self.src_proj = nn.Linear(INPUT_DIM, D_MODEL)

        # Эмбеддинги нот
        self.note_emb = nn.Embedding(NOTE_VOCAB, EMB_SIZE, padding_idx=NOTE_PAD)

        # Эмбеддинги состояний нот
        self.state_emb = nn.Embedding(STATE_VOCAB, EMB_SIZE, padding_idx=STATE_PAD)

        # Объединение эмбеддингов decoder
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

        # Выходные слои предсказания
        self.note1_head = nn.Linear(D_MODEL, NOTE_CLASSES)
        self.state1_head = nn.Linear(D_MODEL, STATE_CLASSES)

        self.note2_head = nn.Linear(D_MODEL, NOTE_CLASSES)
        self.state2_head = nn.Linear(D_MODEL, STATE_CLASSES)

    def encode(self, features):
        """
        Кодирует входные признаки аккордов

        Args:
            features (torch.Tensor): признаки аккордов

        Returns:
            torch.Tensor: encoder-представление
        """

        x = self.src_proj(features)

        return self.pos_enc(x)

    def decode(self, y):
        """
        Кодирует входную последовательность decoder

        Args:
            y (torch.Tensor): вход decoder

        Returns:
            torch.Tensor: decoder-представление
        """

        # Разделение компонентов нот
        n1 = y[:, :, 0]
        s1 = y[:, :, 1]

        n2 = y[:, :, 2]
        s2 = y[:, :, 3]

        # Объединение эмбеддингов
        emb = torch.cat([
            self.note_emb(n1),
            self.state_emb(s1),
            self.note_emb(n2),
            self.state_emb(s2)
        ], dim=-1)

        emb = self.decoder_proj(emb)

        return self.pos_enc(emb)

    def forward(self, features, y_in, pad_mask):
        """
        Выполняет полный проход модели

        Args:
            features (torch.Tensor): признаки аккордов
            y_in (torch.Tensor): вход decoder
            pad_mask (torch.Tensor): маска PAD-токенов

        Returns:
            tuple: предсказания всех компонентов ноты
        """

        src = self.encode(features)
        tgt = self.decode(y_in)

        tgt_len = tgt.size(1)

        # Маска запрещает decoder смотреть в будущее
        tgt_mask = torch.triu(
            torch.ones(
                tgt_len,
                tgt_len,
                device=tgt.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=pad_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask
        )

        return (
            self.note1_head(out),
            self.state1_head(out),
            self.note2_head(out),
            self.state2_head(out)
        )


# =========================
def masked_ce_loss(logits, targets, mask):
    """
    Считает CrossEntropyLoss только по реальным данным,
    игнорируя PAD-области

    Args:
        logits (torch.Tensor): предсказания модели
        targets (torch.Tensor): целевые значения
        mask (torch.Tensor): PAD-маска

    Returns:
        torch.Tensor: значение функции потерь
    """

    # Получаем размеры батча и длину последовательности
    b, l, c = logits.shape

    # Вычисляем CrossEntropyLoss для всех позиций,
    loss = F.cross_entropy(
        logits.reshape(-1, c),
        targets.reshape(-1),
        reduction="none"
    ).view(b, l)

    # Инвертируем маску, чтобы True стало 0, а False стало 1
    mask = (~mask).float()

    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def train():
    """
    Запускает полный цикл обучения модели
    """

    # =========================
    # ЗАГРУЗКА ДАТАСЕТА
    # =========================
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = GuitarDataset(data, max_len=MAX_LEN, stride=STRIDE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # =========================
    # СОЗДАНИЕ МОДЕЛИ
    # =========================
    model = SoloTransformer().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_batches = len(loader)

    # =========================
    # ЦИКЛ ОБУЧЕНИЯ
    # =========================
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        epoch_start_time = time.time()

        progress = tqdm(
            enumerate(loader, start=1),
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            dynamic_ncols=True,
            leave=True
        )

        # =========================
        # ПРОХОД ПО БАТЧАМ
        # =========================
        for batch_idx, batch in progress:

            # Перенос данных на GPU/CPU
            features = batch["features"].to(DEVICE)

            y_in = batch["y_in"].to(DEVICE)
            y_out = batch["y_out"].to(DEVICE)

            pad_mask = batch["pad_mask"].to(DEVICE)

            # Предсказание модели
            pred_note1, pred_state1, pred_note2, pred_state2 = model(features, y_in, pad_mask)

            # =========================
            # ВЫЧИСЛЕНИЕ LOSS
            # =========================
            loss = 0.0

            loss += masked_ce_loss(pred_note1, y_out[:, :, 0], pad_mask)
            loss += masked_ce_loss(pred_state1, y_out[:, :, 1], pad_mask)
            loss += masked_ce_loss(pred_note2, y_out[:, :, 2], pad_mask)
            loss += masked_ce_loss(pred_state2, y_out[:, :, 3], pad_mask)

            # Обнуление градиентов
            optimizer.zero_grad(set_to_none=True)

            # Backpropagation
            loss.backward()

            # Ограничение градиентов
            # для защиты от взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Обновление весов
            optimizer.step()
            total_loss += loss.item()

            # =========================
            # ОБНОВЛЕНИЕ PROGRESS BAR
            # =========================
            avg_loss_so_far = total_loss / batch_idx
            elapsed = time.time() - epoch_start_time
            batches_left = num_batches - batch_idx

            eta_seconds = (
                (elapsed / batch_idx) * batches_left
                if batch_idx > 0 else 0.0
            )

            progress.set_postfix({
                "batch": f"{batch_idx}/{num_batches}",
                "loss": f"{avg_loss_so_far:.4f}",
                "eta": f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            })

        # =========================
        # ИТОГИ ЭПОХИ
        # =========================
        avg_loss = total_loss / max(num_batches, 1)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} finished | "
            f"loss = {avg_loss:.4f} | "
            f"time = {int(epoch_time // 60)}m "
            f"{int(epoch_time % 60)}s"
        )

        # =========================
        # СОХРАНЕНИЕ МОДЕЛИ
        # =========================
        if (epoch + 1) % 20 == 0:

            model_path = (MODEL_FOLDER + f"{GENRE}_transformer_{epoch + 1}.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена: {model_path}")

    print("\nОбучение завершено")


if __name__ == "__main__":
    train()