import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import CLASSES


class SimpleCNN(nn.Module):
    """Компактная сверточная нейросеть для классификации изображений 32x32."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 4,
    batch_size: int = 128,
    lr: float = 0.003,
    verbose: bool = True,
) -> float:
    """Обучение модели. Возвращает затраченное время (сек)."""
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for x_b, y_b in loader:
            optimizer.zero_grad()
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1) * 100.0

        if verbose:
            print(f"Эпоха {epoch}/{epochs} | Loss: {avg_loss:.4f} | Точность: {acc:.2f}%")

    elapsed = time.perf_counter() - t0
    if verbose:
        throughput = len(y_train) * epochs / elapsed
        print(f"Обучение завершено за {elapsed:.2f} сек ({throughput:.0f} изобр/сек).")
    return elapsed


def evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 128,
    verbose: bool = True,
) -> dict[str, Any]:
    """Автоматическая проверка модели с расчетом метрик для каждого класса."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for x_b, y_b in loader:
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            total_loss += loss.item() * len(y_b)
            all_preds.append(outputs.argmax(dim=1))

    preds = torch.cat(all_preds, dim=0)
    avg_loss = total_loss / len(y_test)
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
            f"Общая точность (Accuracy): {overall_acc * 100:.2f}% | Macro F1: {macro_f1:.4f} | Loss: {avg_loss:.4f}"
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
        "loss": avg_loss,
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
    }


def predict_single(
    model: nn.Module, img_tensor: torch.Tensor
) -> tuple[int, list[float]]:
    """Предсказание класса и вероятностей для одного изображения."""
    model.eval()
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    pred_class = int(torch.tensor(probs).argmax().item())
    return pred_class, probs


def save_model(model: nn.Module, path: str = "model.pt") -> None:
    torch.save(model.state_dict(), path)
    print(f"Веса модели сохранены в '{path}'.")


def load_model(model: nn.Module, path: str = "model.pt") -> None:
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Веса модели загружены из '{path}'.")
