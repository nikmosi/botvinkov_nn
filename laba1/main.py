import json
import random
from math import exp


def sigmoid(x: float) -> float:
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
    lr: float = 0.5,
) -> float:
    weights, biases = network["weights"], network["biases"]
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
    for layer_idx in range(len(weights)):
        d_next = deltas[layer_idx]
        a_prev = memory[layer_idx]
        for i in range(len(d_next)):
            biases[layer_idx][i] += lr * d_next[i]
            for j in range(len(a_prev)):
                weights[layer_idx][j][i] += lr * d_next[i] * a_prev[j]

    return error**2


def train(network: dict[str, list], steps: int = 30000, lr: float = 0.5) -> None:
    print(f"Обучение: {steps} шагов...")
    total_loss = 0.0

    for step in range(steps):
        x = [random.random(), random.random()]
        y = (x[0] + x[1]) / 2.0

        memory = forward(network, x)
        total_loss += backward(network, memory, target=y, lr=lr)

        if (step + 1) % (steps // 5 or 1) == 0:
            avg_loss = total_loss / (steps // 5 or 1)
            print(f"Шаг {step + 1}/{steps} | Средняя ошибка (MSE): {avg_loss:.6f}")
            total_loss = 0.0
    print("Обучение завершено.")


def save_model(network: dict[str, list], path: str = "model.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2)
    print(f"Модель сохранена в '{path}'.")


def load_model(path: str = "model.json") -> dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {
            "weights": data,
            "biases": [[0.0 for _ in layer[0]] for layer in data],
        }
    print(f"Модель загружена '{path}'.")
    return data


def test_model(network: dict[str, list]) -> None:
    try:
        raw = input("Введите 2 числа (например, 0.2 0.8): ").strip()
        nums = [float(val) for val in raw.split()]
        if len(nums) != 2:
            print("Ошибка: нужно ввести ровно 2 числа.")
            return
        if not (0.0 <= nums[0] <= 1.0 and 0.0 <= nums[1] <= 1.0):
            print("Предупреждение: числа выходят за [0, 1].")
        pred = forward(network, nums)[-1][0]
        expected = (nums[0] + nums[1]) / 2.0
        print(f"Истинное среднее: {expected:.4f}")
        print(f"Предсказание сети: {pred:.4f}")
        print(f"Абсолютная разница: {abs(pred - expected):.4f}")
    except ValueError:
        print("Ошибка: введены некорректные числа.")


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

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            raw_steps = input("Количество шагов (Enter для 30000): ").strip()
            steps = int(raw_steps) if raw_steps.isdigit() else 30000
            train(network, steps=steps)
        elif choice == "2":
            path = input("Имя файла (Enter для 'model.json'): ").strip() or "model.json"
            save_model(network, path)
        elif choice == "3":
            path = input("Имя файла (Enter для 'model.json'): ").strip() or "model.json"
            try:
                network = load_model(path)
            except FileNotFoundError:
                print(f"Ошибка: файл '{path}' не найден.")
            except (json.JSONDecodeError, OSError) as e:
                print(f"Ошибка при загрузке: {e}")
        elif choice == "4":
            test_model(network)
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный пункт, попробуйте снова.")


if __name__ == "__main__":
    main()
