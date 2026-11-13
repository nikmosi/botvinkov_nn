import torch
from torch import nn
from tqdm import tqdm


def create_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 2),
        nn.Sigmoid(),
        nn.Linear(2, 1),
        nn.Sigmoid(),
    )


def train(
    model: nn.Module,
    epochs: int = 10,
    batch_size: int = 16,
    samples_per_epoch: int = 2000,
    lr: float = 0.3,
    steps: int | None = None,
) -> None:
    if steps is not None:
        epochs = max(1, steps // samples_per_epoch)

    print(
        f"Обучение: {epochs} эпох по {samples_per_epoch} примеров (батч {batch_size})..."
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="sum")
    model.train()

    total_batches = (samples_per_epoch + batch_size - 1) // batch_size
    total_samples = epochs * samples_per_epoch
    all_loss = 0.0

    for epoch in range(1, epochs + 1):
        x = torch.rand(samples_per_epoch, 2)
        y = x.mean(dim=1, keepdim=True)

        with tqdm(
            range(0, samples_per_epoch, batch_size),
            total=total_batches,
            desc=f"Эпоха {epoch}/{epochs}",
            unit="батч",
        ) as pbar:
            for iteration, start_idx in enumerate(pbar, 1):
                bx = x[start_idx : start_idx + batch_size]
                by = y[start_idx : start_idx + batch_size]

                optimizer.zero_grad()
                pred = model(bx)
                loss_sum = criterion(pred, by)
                (loss_sum / 2.0).backward()
                optimizer.step()

                batch_loss = loss_sum.item()
                all_loss += batch_loss
                current_mse = batch_loss / len(bx)
                pbar.set_postfix({"MSE": f"{current_mse:.5f}", "итерация": iteration})

    avg_mse = all_loss / total_samples
    print(f"Обучение завершено. Итоговый MSE: {avg_mse:.6f}")


def save_model(model: nn.Module, path: str = "model_torch.pt") -> None:
    torch.save(model.state_dict(), path)
    print(f"Сохранено в '{path}'.")


def load_model(model: nn.Module, path: str = "model_torch.pt") -> None:
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Загружено из '{path}'.")


def test_model(model: nn.Module) -> None:
    try:
        raw = input("Введите два числа от 0 до 1 (например: 0.2 0.8): ").strip()
        nums = [float(val) for val in raw.split()]
        if len(nums) != 2:
            print("Нужно ровно два числа.")
            return
        if not (0.0 <= nums[0] <= 1.0 and 0.0 <= nums[1] <= 1.0):
            print("Внимание: числа должны быть в диапазоне от 0 до 1.")

        model.eval()
        with torch.no_grad():
            tensor_in = torch.tensor([nums], dtype=torch.float32)
            pred = model(tensor_in).item()

        expected = (nums[0] + nums[1]) / 2.0
        print(f"Точное среднее:  {expected:.4f}")
        print(f"Ответ сети:      {pred:.4f}")
        print(f"Разница:         {abs(pred - expected):.4f}")
    except ValueError:
        print("Неверный ввод чисел.")


def main() -> None:
    model = create_model()

    while True:
        print("\n=== Меню (PyTorch) ===")
        print("1. Обучить модель")
        print("2. Сохранить модель")
        print("3. Загрузить модель")
        print("4. Проверить модель")
        print("0. Выход")

        choice = input("Выберите действие: ").strip()

        if choice == "1":
            raw_epochs = input("Количество эпох [10]: ").strip()
            epochs = int(raw_epochs) if raw_epochs.isdigit() else 10
            train(model, epochs=epochs)
        elif choice == "2":
            path = input("Имя файла [model_torch.pt]: ").strip() or "model_torch.pt"
            save_model(model, path)
        elif choice == "3":
            path = input("Имя файла [model_torch.pt]: ").strip() or "model_torch.pt"
            try:
                load_model(model, path)
            except FileNotFoundError:
                print(f"Файл '{path}' не найден.")
            except (RuntimeError, OSError) as e:
                print(f"Не удалось загрузить модель: {e}")
        elif choice == "4":
            test_model(model)
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный ввод.")


if __name__ == "__main__":
    main()
