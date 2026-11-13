import multiprocessing as mp
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

CLASSES = ["Круг", "Квадрат", "Треугольник"]
DEFAULT_IMAGE_SIZE = 128


def generate_single_image(
    cls: int, augment: bool = False, size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Генерация одного изображения (3, size, size) в формате float32 [0, 1]."""
    if augment:
        bg = (
            random.randint(0, 70),
            random.randint(0, 70),
            random.randint(0, 70),
        )
        fg = (
            random.randint(180, 255),
            random.randint(180, 255),
            random.randint(180, 255),
        )
    else:
        bg = (0, 0, 0)
        fg = (255, 255, 255)

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    margin = size // 6
    if cls == 0:  # Круг
        draw.ellipse([margin, margin, size - margin, size - margin], fill=fg)
    elif cls == 1:  # Квадрат
        draw.rectangle([margin, margin, size - margin, size - margin], fill=fg)
    elif cls == 2:  # Треугольник
        draw.polygon(
            [
                (size // 2, margin),
                (margin, size - margin),
                (size - margin, size - margin),
            ],
            fill=fg,
        )

    if augment:
        # Поворот
        angle = random.uniform(0.0, 360.0)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=bg)

        arr = np.array(img, dtype=np.float32)
        # Аддитивный шум
        noise = np.random.normal(0.0, 15.0, arr.shape)
        arr = arr + noise

        # Битые пиксели (salt-and-pepper)
        dead_mask = np.random.rand(size, size) < 0.03
        arr[dead_mask] = random.choice([0.0, 255.0])

        arr = np.clip(arr, 0.0, 255.0)
    else:
        arr = np.array(img, dtype=np.float32)

    # Транспонирование в (C, H, W) и нормализация в [0, 1]
    tensor_arr = arr.transpose(2, 0, 1) / 255.0
    return tensor_arr.astype(np.float32)


def _worker_gen(args: tuple[int, int, bool, int]) -> tuple[np.ndarray, np.ndarray]:
    """Вспомогательная функция для генерации в пуле процессов."""
    cls, count, augment, size = args
    imgs = [generate_single_image(cls, augment, size) for _ in range(count)]
    labels = [cls] * count
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.int64)


def create_dataset(
    counts: list[int],
    augment: bool = False,
    test_per_class: int = 600,
    n_jobs: int = 1,
    size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Создает обучающую и тестовую выборки.

    counts: количество обучающих примеров для каждого класса [cls0, cls1, cls2]
    test_per_class: количество тестовых примеров на класс
    n_jobs: количество параллельных процессов для генерации
    size: размер стороны квадратного изображения (по умолчанию 128)
    """
    train_tasks = [(c, cnt, augment, size) for c, cnt in enumerate(counts)]
    test_tasks = [(c, test_per_class, augment, size) for c in range(len(counts))]

    all_tasks = train_tasks + test_tasks

    if n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(_worker_gen, all_tasks)
    else:
        results = [_worker_gen(t) for t in all_tasks]

    num_classes = len(counts)
    train_res = results[:num_classes]
    test_res = results[num_classes:]

    x_train = np.concatenate([r[0] for r in train_res], axis=0)
    y_train = np.concatenate([r[1] for r in train_res], axis=0)
    x_test = np.concatenate([r[0] for r in test_res], axis=0)
    y_test = np.concatenate([r[1] for r in test_res], axis=0)

    # Перемешивание обучающей выборки
    perm = np.random.permutation(len(y_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def ascii_art(img_tensor: torch.Tensor, width: int = 24, height: int = 12) -> str:
    """Визуализирует изображение в ASCII-символах для консоли."""
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        # Перевод в grayscale (0.299 R + 0.587 G + 0.114 B)
        gray = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
    else:
        gray = arr.squeeze()

    pil_img = Image.fromarray((gray * 255.0).astype(np.uint8))
    pil_resized = pil_img.resize((width, height), Image.Resampling.BILINEAR)
    grid = np.array(pil_resized, dtype=int)

    chars = " .:-=+*#%@"
    lines = []
    for row in grid:
        line = "".join(chars[int(val) * (len(chars) - 1) // 255] for val in row)
        lines.append(line)
    return "\n".join(lines)


def save_image_file(tensor: torch.Tensor, path: str) -> None:
    """Сохранение тензора (3, H, W) [0..1] на диск в формате PNG/JPG."""
    t = tensor.detach().cpu().squeeze()
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
    arr = (t.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_image_file(path: str, size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Загрузка изображения с диска в тензор (3, size, size) [0..1]."""
    img = (
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    )
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)
