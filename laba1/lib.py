import torch
from torch import nn


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
    steps: int = 3000,
    batch_size: int = 64,
    lr: float = 0.01,
) -> None:
    print(f"Обучение: {steps} шагов (размер батча: {batch_size})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    total_loss = 0.0
    report_every = steps // 5 or 1

    for step in range(steps):
        x = torch.rand(batch_size, 2)
        y = x.mean(dim=1, keepdim=True)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (step + 1) % report_every == 0:
            avg_loss = total_loss / report_every
            print(f"Шаг {step + 1}/{steps} | Средняя ошибка (MSE): {avg_loss:.6f}")
            total_loss = 0.0

    print("Обучение завершено.")


def save_model(model: nn.Module, path: str = "model_torch.pt") -> None:
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в '{path}'.")


def load_model(model: nn.Module, path: str = "model_torch.pt") -> None:
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Модель успешно загружена из '{path}'.")


def test_model(model: nn.Module) -> None:
    try:
        raw = input(
            "Введите 2 числа от 0 до 1 через пробел (например, 0.2 0.8): "
        ).strip()
        nums = [float(val) for val in raw.split()]
        if len(nums) != 2:
            print("Ошибка: нужно ввести ровно 2 числа.")
            return
        if not (0.0 <= nums[0] <= 1.0 and 0.0 <= nums[1] <= 1.0):
            print("Предупреждение: числа выходят за рекомендуемый диапазон [0, 1].")

        model.eval()
        with torch.no_grad():
            tensor_in = torch.tensor([nums], dtype=torch.float32)
            pred = model(tensor_in).item()

        expected = (nums[0] + nums[1]) / 2.0
        print(f"Истинное среднее: {expected:.4f}")
        print(f"Предсказание сети: {pred:.4f}")
        print(f"Абсолютная разница: {abs(pred - expected):.4f}")
    except ValueError:
        print("Ошибка: введены некорректные числа.")


def main() -> None:
    model = create_model()

    while True:
        print("\n=== Меню нейросети (PyTorch) ===")
        print("1. Обучить модель")
        print("2. Сохранить модель")
        print("3. Загрузить модель")
        print("4. Проверить модель")
        print("0. Выход")

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            raw_steps = input("Количество шагов (Enter для 3000): ").strip()
            steps = int(raw_steps) if raw_steps.isdigit() else 3000
            train(model, steps=steps)
        elif choice == "2":
            path = (
                input("Имя файла (Enter для 'model_torch.pt'): ").strip()
                or "model_torch.pt"
            )
            save_model(model, path)
        elif choice == "3":
            path = (
                input("Имя файла (Enter для 'model_torch.pt'): ").strip()
                or "model_torch.pt"
            )
            try:
                load_model(model, path)
            except FileNotFoundError:
                print(f"Ошибка: файл '{path}' не найден.")
            except (RuntimeError, OSError) as e:
                print(f"Ошибка при загрузке модели: {e}")
        elif choice == "4":
            test_model(model)
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный пункт, попробуйте снова.")


if __name__ == "__main__":
    main()
