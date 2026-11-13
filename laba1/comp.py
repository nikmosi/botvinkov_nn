import time

import torch

import lib
import my


def compare_single_input(my_net: dict, lib_net: torch.nn.Module) -> None:
    try:
        raw = input("Введите два числа от 0 до 1 (например: 0.2 0.8): ").strip()
        nums = [float(val) for val in raw.split()]
        if len(nums) != 2:
            print("Нужно ровно два числа.")
            return

        expected = (nums[0] + nums[1]) / 2.0

        pred_my = my.forward(my_net, nums)[-1][0]
        err_my = abs(pred_my - expected)

        lib_net.eval()
        with torch.no_grad():
            tensor_in = torch.tensor([nums], dtype=torch.float32)
            pred_lib = lib_net(tensor_in).item()
        err_lib = abs(pred_lib - expected)

        print("\n--- Результаты ---")
        print(f"Входные числа:   {nums}")
        print(f"Точное среднее:  {expected:.4f}")
        print(f"my.py (чистый):  {pred_my:.4f} (разница: {err_my:.4f})")
        print(f"lib.py (PyTorch):{pred_lib:.4f} (разница: {err_lib:.4f})")
    except ValueError:
        print("Неверный ввод чисел.")


def compare_accuracy(
    my_net: dict, lib_net: torch.nn.Module, n_samples: int = 1000
) -> None:
    print(f"\nТестируем на {n_samples} случайных примерах...")
    torch.manual_seed(42)
    x_test = torch.rand(n_samples, 2)
    y_test = x_test.mean(dim=1).tolist()
    samples = x_test.tolist()

    t0 = time.perf_counter()
    preds_my = [my.forward(my_net, s)[-1][0] for s in samples]
    t_my = time.perf_counter() - t0

    lib_net.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        preds_lib = lib_net(x_test).squeeze().tolist()
    t_lib = time.perf_counter() - t0

    diffs_my = [abs(p - y) for p, y in zip(preds_my, y_test, strict=True)]
    diffs_lib = [abs(p - y) for p, y in zip(preds_lib, y_test, strict=True)]

    mse_my = sum(d**2 for d in diffs_my) / n_samples
    mse_lib = sum(d**2 for d in diffs_lib) / n_samples
    mae_my = sum(diffs_my) / n_samples
    mae_lib = sum(diffs_lib) / n_samples

    print("\n" + "=" * 54)
    print(f"{'Метрика':<22} | {'my.py (чистый)':<14} | {'lib.py (PyTorch)':<14}")
    print("-" * 54)
    print(f"{'MSE (ср. кв. ошибка)':<22} | {mse_my:<14.6f} | {mse_lib:<14.6f}")
    print(f"{'MAE (средняя ошибка)':<22} | {mae_my:<14.6f} | {mae_lib:<14.6f}")
    print(
        f"{'Максимальная ошибка':<22} | {max(diffs_my):<14.6f} | {max(diffs_lib):<14.6f}"
    )
    print(f"{'Время теста (сек)':<22} | {t_my:<14.4f} | {t_lib:<14.4f}")
    print("=" * 54)


def train_both(my_net: dict, lib_net: torch.nn.Module) -> None:
    raw = input("Количество шагов [20000]: ").strip()
    steps = int(raw) if raw.isdigit() else 20000

    print("\n--- 1. Обучаем my.py (чистый Python) ---")
    t0 = time.perf_counter()
    my.train(my_net, steps=steps)
    t_my = time.perf_counter() - t0

    print("\n--- 2. Обучаем lib.py (PyTorch) ---")
    t0 = time.perf_counter()
    lib.train(lib_net, steps=steps)
    t_lib = time.perf_counter() - t0

    print("\n" + "=" * 55)
    print(f"Время обучения my.py:  {t_my:.3f} сек ({steps} примеров)")
    print(f"Время обучения lib.py: {t_lib:.3f} сек ({steps * 64} примеров)")
    print(
        f"Скорость обработки:    my.py = {steps / t_my:.0f} обр/сек | lib.py = {(steps * 64) / t_lib:.0f} обр/сек"
    )
    print("=" * 55)


def load_both(my_net: dict, lib_net: torch.nn.Module) -> dict:
    path_my = input("Файл my.py [model.json]: ").strip() or "model.json"
    try:
        my_net = my.load_model(path_my)
    except FileNotFoundError:
        print(f"Файл '{path_my}' не найден.")
    except (my.json.JSONDecodeError, OSError) as e:
        print(f"Не удалось загрузить '{path_my}': {e}")

    path_lib = input("Файл lib.py [model_torch.pt]: ").strip() or "model_torch.pt"
    try:
        lib.load_model(lib_net, path_lib)
    except FileNotFoundError:
        print(f"Файл '{path_lib}' не найден.")
    except (RuntimeError, OSError) as e:
        print(f"Не удалось загрузить '{path_lib}': {e}")

    return my_net


def main() -> None:
    sizes = [2, 3, 2, 1]
    my_net = my.init_network(sizes)
    lib_net = lib.create_model()

    while True:
        print("\n=== Сравнение (my.py vs lib.py) ===")
        print("1. Обучить обе модели (с замером времени)")
        print("2. Сравнить точность на 1000 примерах (MSE, MAE, время теста)")
        print("3. Сравнить предсказание на конкретных числах")
        print("4. Загрузить обе модели из файлов")
        print("0. Выход")

        choice = input("Выберите действие: ").strip()

        if choice == "1":
            train_both(my_net, lib_net)
        elif choice == "2":
            compare_accuracy(my_net, lib_net)
        elif choice == "3":
            compare_single_input(my_net, lib_net)
        elif choice == "4":
            my_net = load_both(my_net, lib_net)
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный ввод.")


if __name__ == "__main__":
    main()
