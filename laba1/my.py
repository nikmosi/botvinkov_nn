import json
import random
from itertools import batched
from math import exp

from tqdm import tqdm


def sigmoid(x: float) -> float:
    # shortcut: clamp to [-500, 500] to prevent math range overflow
    clamped = max(min(x, 500.0), -500.0)
    return 1.0 / (1.0 + exp(-clamped))


def sigmoid_derivative(x: float) -> float:
    return (1.0 - x) * x


def init_network(sizes: list[int]) -> dict[str, list]:
    return {
        "weights": [
            [
                [random.uniform(-0.5, 0.5) for _ in range(sizes[i + 1])]
                for _ in range(sizes[i])
            ]
            for i in range(len(sizes) - 1)
        ],
        "biases": [
            [random.uniform(-0.5, 0.5) for _ in range(sizes[i + 1])]
            for i in range(len(sizes) - 1)
        ],
    }


def forward(network: dict[str, list], inputs: list[float]) -> list[list[float]]:
    memory = [inputs]
    for weights, biases in zip(network["weights"], network["biases"], strict=True):
        prev = memory[-1]
        nxt = [
            sigmoid(biases[i] + sum(prev[j] * weights[j][i] for j in range(len(prev))))
            for i in range(len(biases))
        ]
        memory.append(nxt)
    return memory


def backward(
    network: dict[str, list],
    memory: list[list[float]],
    target: float,
) -> tuple[list[list[float]], float]:
    weights = network["weights"]
    output = memory[-1][0]
    error = target - output
    d_o = error * sigmoid_derivative(output)

    deltas = [[d_o]]
    for back_step in range(1, len(weights)):
        from_ = deltas[-1]
        w = weights[-back_step]
        mem = memory[-back_step - 1]
        to_ = [
            sigmoid_derivative(mem[i])
            * sum(w[i][j] * from_[j] for j in range(len(from_)))
            for i in range(len(w))
        ]
        deltas.append(to_)

    deltas.reverse()
    return deltas, error**2


def train(
    network: dict[str, list],
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
    weights, biases = network["weights"], network["biases"]

    for epoch in range(1, epochs + 1):
        dataset = [
            ([random.random(), random.random()], None) for _ in range(samples_per_epoch)
        ]
        dataset = [(x, (x[0] + x[1]) / 2.0) for x, _ in dataset]

        total_loss = 0.0
        total_batches = (len(dataset) + batch_size - 1) // batch_size

        with tqdm(
            batched(dataset, batch_size),
            total=total_batches,
            desc=f"Эпоха {epoch}/{epochs}",
            unit="батч",
        ) as pbar:
            for iteration, batch in enumerate(pbar, 1):
                grad_w = [[[0.0 for _ in row] for row in layer] for layer in weights]
                grad_b = [[0.0 for _ in layer] for layer in biases]
                batch_loss = 0.0

                for x, y in batch:
                    memory = forward(network, x)
                    deltas, loss = backward(network, memory, y)
                    batch_loss += loss

                    for l in range(len(weights)):
                        for i, d in enumerate(deltas[l]):
                            grad_b[l][i] += d
                            for j, a in enumerate(memory[l]):
                                grad_w[l][j][i] += d * a

                total_loss += batch_loss

                # Обновление весов и смещений на основе накопленных градиентов батча
                for l in range(len(weights)):
                    for i in range(len(biases[l])):
                        biases[l][i] += lr * grad_b[l][i]
                    for j in range(len(weights[l])):
                        for i in range(len(weights[l][j])):
                            weights[l][j][i] += lr * grad_w[l][j][i]

                current_mse = batch_loss / len(batch)
                pbar.set_postfix({"MSE": f"{current_mse:.5f}", "итерация": iteration})

    avg_mse = total_loss / samples_per_epoch
    print(f"Обучение завершено. Итоговый MSE: {avg_mse:.6f}")


def save_model(network: dict[str, list], path: str = "model.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2)
    print(f"Сохранено в '{path}'.")


def load_model(path: str = "model.json") -> dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {
            "weights": data,
            "biases": [[0.0 for _ in layer[0]] for layer in data],
        }
    print(f"Загружено из '{path}'.")
    return data


def test_model(network: dict[str, list]) -> None:
    try:
        raw = input("Введите два числа от 0 до 1 (например: 0.2 0.8): ").strip()
        nums = [float(val) for val in raw.split()]
        if len(nums) != 2:
            print("Нужно ровно два числа.")
            return
        if not (0.0 <= nums[0] <= 1.0 and 0.0 <= nums[1] <= 1.0):
            print("Внимание: числа должны быть в диапазоне от 0 до 1.")
        pred = forward(network, nums)[-1][0]
        expected = (nums[0] + nums[1]) / 2.0
        print(f"Точное среднее:  {expected:.4f}")
        print(f"Ответ сети:      {pred:.4f}")
        print(f"Разница:         {abs(pred - expected):.4f}")
    except ValueError:
        print("Неверный ввод чисел.")


def main() -> None:
    sizes = [2, 3, 2, 1]
    network = init_network(sizes)

    while True:
        print("\n=== Меню ===")
        print("1. Обучить модель")
        print("2. Сохранить модель")
        print("3. Загрузить модель")
        print("4. Проверить модель")
        print("0. Выход")

        choice = input("Выберите действие: ").strip()

        if choice == "1":
            raw_epochs = input("Количество эпох [10]: ").strip()
            epochs = int(raw_epochs) if raw_epochs.isdigit() else 10
            train(network, epochs=epochs)
        elif choice == "2":
            path = input("Имя файла [model.json]: ").strip() or "model.json"
            save_model(network, path)
        elif choice == "3":
            path = input("Имя файла [model.json]: ").strip() or "model.json"
            try:
                network = load_model(path)
            except FileNotFoundError:
                print(f"Файл '{path}' не найден.")
            except (json.JSONDecodeError, OSError) as e:
                print(f"Не удалось загрузить: {e}")
        elif choice == "4":
            test_model(network)
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный ввод.")


if __name__ == "__main__":
    main()
