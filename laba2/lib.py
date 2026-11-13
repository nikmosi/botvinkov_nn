import time
from typing import Any

import torch
from torch import nn

from dataset import CLASSES, DEFAULT_IMAGE_SIZE, load_image_file

DEFAULT_INPUT_SIZE = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE * 3


def create_model(sizes: list[int] | None = None) -> nn.Sequential:
    """Создает перцептрон в PyTorch с сигмоидами (как в laba1/lib.py)."""
    if sizes is None:
        sizes = [DEFAULT_INPUT_SIZE, 256, 3]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def train(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 0.5,
    verbose: bool = False,
) -> float:
    """Обучение перцептрона PyTorch (SGD + MSELoss) для прямого сравнения с my.py."""
    num_samples = len(y_train)
    num_classes = len(CLASSES)
    y_one_hot = torch.zeros((num_samples, num_classes), dtype=torch.float32)
    y_one_hot[torch.arange(num_samples), y_train] = 1.0
    x_flat = x_train.reshape(num_samples, -1)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")
    model.train()

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        for i in range(0, num_samples, batch_size):
            b_idx = perm[i : i + batch_size]
            bx = x_flat[b_idx]
            by = y_one_hot[b_idx]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            (loss * 0.5).backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        if verbose:
            print(f"Эпоха {epoch}/{epochs} | MSE: {total_loss / num_samples:.5f}")

    return time.perf_counter() - t0


def evaluate(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    verbose: bool = False,
) -> dict[str, Any]:
    """Оценка точности модели PyTorch."""
    model.eval()
    x_flat = x_test.reshape(len(x_test), -1)
    with torch.no_grad():
        out = model(x_flat)
    preds = torch.argmax(out, dim=1)
    acc = (preds == y_test).float().mean().item()
    if verbose:
        print(f"PyTorch Точность: {acc * 100:.2f}%")
    return {"overall_accuracy": acc}


def predict_single(
    model: nn.Module, img_tensor: torch.Tensor
) -> tuple[int, list[float]]:
    """Предсказание класса и вероятностей для одиночного изображения."""
    model.eval()
    x = img_tensor.reshape(1, -1)
    with torch.no_grad():
        out = model(x).squeeze(0)
    s = out.sum().item()
    probs = (out / s).tolist() if s > 0 else [1.0 / len(out)] * len(out)
    return int(torch.argmax(out).item()), probs


def predict_image_file(
    model: nn.Module, image_path: str, size: int = DEFAULT_IMAGE_SIZE
) -> tuple[int, list[float]]:
    """Предсказание класса для изображения по пути к файлу."""
    tensor = load_image_file(image_path, size=size)
    return predict_single(model, tensor)


def save_model(
    model: nn.Module, path: str = "model_torch.pt", verbose: bool = False
) -> None:
    """Сохранение весов PyTorch на диск."""
    torch.save(model.state_dict(), path)
    if verbose:
        print(f"Сохранено в '{path}'.")


def load_model(
    model: nn.Module, path: str = "model_torch.pt", verbose: bool = False
) -> None:
    """Загрузка весов PyTorch с диска."""
    model.load_state_dict(torch.load(path, weights_only=True))
    if verbose:
        print(f"Загружено из '{path}'.")
