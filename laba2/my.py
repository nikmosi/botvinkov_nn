import json
import time
from typing import Any

import torch

from dataset import CLASSES, DEFAULT_IMAGE_SIZE, load_image_file

DEFAULT_INPUT_SIZE = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE * 3  # 128*128*3 = 49152


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Сигмоидальная функция активации (с ограничением диапазона)."""
    return torch.sigmoid(torch.clamp(x, -50.0, 50.0))


def sigmoid_derivative(a: torch.Tensor) -> torch.Tensor:
    """Производная сигмоиды через ее выходное значение a = sigmoid(z)."""
    return (1.0 - a) * a


def init_network(sizes: list[int] | None = None) -> dict[str, Any]:
    """Инициализация простого перцептрона: веса и смещения по слоям (как в laba1)."""
    if sizes is None:
        sizes = [
            DEFAULT_INPUT_SIZE,
            256,
            3,
        ]  # 128*128*3 -> 256 скрытых -> 3 класса (~1 мин обучения)

    weights = []
    biases = []
    for i in range(len(sizes) - 1):
        limit = (6.0 / (sizes[i] + sizes[i + 1])) ** 0.5
        w = torch.empty(sizes[i], sizes[i + 1]).uniform_(-limit, limit)
        b = torch.zeros(sizes[i + 1])
        weights.append(w)
        biases.append(b)

    return {"weights": weights, "biases": biases, "sizes": sizes}


def forward(network: dict[str, Any], inputs: torch.Tensor) -> list[torch.Tensor]:
    """Прямой проход перцептрона. Сохраняет память активаций для обратного хода."""
    x = inputs.reshape(inputs.shape[0], -1) if inputs.ndim > 2 else inputs
    if x.ndim == 1:
        x = x.unsqueeze(0)

    memory = [x]
    for w, b in zip(network["weights"], network["biases"], strict=True):
        z = memory[-1] @ w + b
        a = sigmoid(z)
        memory.append(a)
    return memory


def backward(
    network: dict[str, Any],
    memory: list[torch.Tensor],
    target: torch.Tensor,
) -> tuple[list[torch.Tensor], float]:
    """Обратное распространение ошибки (backprop) через градиенты и цепное правило."""
    weights = network["weights"]
    output = memory[-1]

    # Ошибка и дельта выходного слоя
    error = target - output
    d_o = error * sigmoid_derivative(output)

    deltas = [d_o]
    for back_step in range(1, len(weights)):
        from_ = deltas[-1]
        w = weights[-back_step]
        mem = memory[-back_step - 1]
        to_ = sigmoid_derivative(mem) * (from_ @ w.T)
        deltas.append(to_)

    deltas.reverse()
    mse_loss = torch.mean(torch.sum(error**2, dim=1)).item()
    return deltas, mse_loss


def train(
    network: dict[str, Any],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 0.5,
    verbose: bool = False,
) -> float:
    """Обучение перцептрона методом градиентного спуска по батчам."""
    num_samples = len(y_train)
    num_classes = network["sizes"][-1]

    # One-hot кодирование целевых меток классов
    y_one_hot = torch.zeros((num_samples, num_classes), dtype=torch.float32)
    y_one_hot[torch.arange(num_samples), y_train] = 1.0

    x_flat = x_train.reshape(num_samples, -1)

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        correct = 0

        for i in range(0, num_samples, batch_size):
            b_idx = perm[i : i + batch_size]
            x_b = x_flat[b_idx]
            y_b = y_one_hot[b_idx]

            memory = forward(network, x_b)
            deltas, b_loss = backward(network, memory, y_b)

            total_loss += b_loss * len(x_b)
            preds = torch.argmax(memory[-1], dim=1)
            correct += (preds == y_train[b_idx]).sum().item()

            # Обновление весов и смещений (градиентный спуск)
            for layer in range(len(network["weights"])):
                grad_w = memory[layer].T @ deltas[layer] / len(x_b)
                grad_b = torch.mean(deltas[layer], dim=0)
                network["weights"][layer] += lr * grad_w
                network["biases"][layer] += lr * grad_b

        avg_loss = total_loss / max(num_samples, 1)
        acc = correct / max(num_samples, 1) * 100.0
        if verbose:
            print(
                f"Эпоха {epoch}/{epochs} | MSE: {avg_loss:.5f} | Точность: {acc:.2f}%"
            )

    elapsed = time.perf_counter() - t0
    if verbose:
        throughput = num_samples * epochs / elapsed
        print(f"Обучение завершено за {elapsed:.2f} сек ({throughput:.0f} изобр/сек).")
    return elapsed


def evaluate(
    network: dict[str, Any],
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    verbose: bool = False,
) -> dict[str, Any]:
    """Проверка качества работы перцептрона с расчетом метрик для каждого класса."""
    x_flat = x_test.reshape(len(x_test), -1)
    memory = forward(network, x_flat)
    output = memory[-1]
    preds = torch.argmax(output, dim=1)

    overall_acc = (preds == y_test).float().mean().item()

    num_classes = len(CLASSES)
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(y_test.tolist(), preds.tolist(), strict=True):
        conf_matrix[true][pred] += 1

    per_class = []
    f1_list = []

    for c in range(num_classes):
        tp = conf_matrix[c][c]
        fp = sum(conf_matrix[i][c] for i in range(num_classes) if i != c)
        fn = sum(conf_matrix[c][j] for j in range(num_classes) if j != c)
        total_c = sum(conf_matrix[c])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_list.append(f1)

        per_class.append(
            {
                "class_idx": c,
                "name": CLASSES[c],
                "samples": total_c,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = sum(f1_list) / max(len(f1_list), 1)

    if verbose:
        print("\n" + "=" * 68)
        print(
            f"{'Класс':<16} | {'Примеров':<9} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
        )
        print("-" * 68)
        for stat in per_class:
            print(
                f"{stat['name']:<16} | {stat['samples']:<9} | {stat['precision']:<10.4f} | "
                f"{stat['recall']:<10.4f} | {stat['f1']:<10.4f}"
            )
        print("-" * 68)
        print(
            f"Общая точность (Accuracy): {overall_acc * 100:.2f}% | Macro F1: {macro_f1:.4f}"
        )
        print("=" * 68)

        print("\nМатрица ошибок (строки — истинный класс, столбцы — предсказанный):")
        header = f"{'Истинный \\ Предск':<18} | " + " | ".join(
            f"{name[:8]:<8}" for name in CLASSES
        )
        print(header)
        print("-" * len(header))
        for i, row in enumerate(conf_matrix):
            row_str = " | ".join(f"{cnt:<8}" for cnt in row)
            print(f"{CLASSES[i]:<18} | {row_str}")

    return {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
    }


def predict_single(
    network: dict[str, Any], img_tensor: torch.Tensor
) -> tuple[int, list[float]]:
    """Предсказание класса и вероятностей для одиночного изображения."""
    x = img_tensor.reshape(1, -1)
    memory = forward(network, x)
    out = memory[-1].squeeze(0)

    # Нормализация в псевдовероятности
    sum_out = torch.sum(out).item()
    if sum_out > 0:
        probs = (out / sum_out).tolist()
    else:
        probs = [1.0 / len(out)] * len(out)

    pred_class = int(torch.argmax(out).item())
    return pred_class, probs


def predict_image_file(
    network: dict[str, Any], path: str, size: int = DEFAULT_IMAGE_SIZE
) -> tuple[int, list[float]]:
    """Предсказание класса и вероятностей для изображения по пути к файлу."""
    img_tensor = load_image_file(path, size=size)
    return predict_single(network, img_tensor)


def save_model(
    network: dict[str, Any], path: str = "model.json", verbose: bool = False
) -> None:
    """Сохранение весов перцептрона в формате JSON (как в laba1)."""
    data = {
        "sizes": network["sizes"],
        "weights": [w.tolist() for w in network["weights"]],
        "biases": [b.tolist() for b in network["biases"]],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f"Веса перцептрона сохранены в '{path}'.")


def load_model(path: str = "model.json", verbose: bool = False) -> dict[str, Any]:
    """Загрузка весов перцептрона из JSON (как в laba1)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weights = [torch.tensor(w, dtype=torch.float32) for w in data["weights"]]
    biases = [torch.tensor(b, dtype=torch.float32) for b in data["biases"]]

    if verbose:
        print(f"Веса перцептрона загружены из '{path}'.")
    return {"weights": weights, "biases": biases, "sizes": data["sizes"]}
