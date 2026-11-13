import argparse
import os
import random
import time

import torch

import dataset
import model

DEFAULT_COUNTS = [3000, 3000, 3000]


def print_banner(cores: int, counts: list[int], augment: bool) -> None:
    mode_str = "Сбалансированный" if len(set(counts)) == 1 else "Дисбаланс"
    aug_str = "ВКЛ (шум, повороты, пиксели, цвета)" if augment else "ВЫКЛ (чистые)"
    print("\n" + "=" * 68)
    print("      Лабораторная работа №2: Классификация изображений")
    print(
        f"  Датасет: {counts} ({mode_str}) | Искажения: {aug_str}"
    )
    print(f"  Активные ядра CPU: {cores} из {os.cpu_count() or 1}")
    print("=" * 68)


def benchmark_cores(
    counts: list[int], augment: bool, max_cores: int
) -> None:
    """Бенчмарк производительности: 1 ядро vs N ядер (по образцу laba1)."""
    print("\n--- Запуск сравнительного бенчмарка (1 ядро vs N ядер) ---")
    bench_counts = [min(c, 1500) for c in counts]
    print(f"Тестовая выборка для замера: {bench_counts}")

    # 1. Замер генерации данных
    print("\n1. Тест многопроцессорной генерации данных (multiprocessing.Pool):")
    t0 = time.perf_counter()
    dataset.create_dataset(bench_counts, augment=augment, test_per_class=300, n_jobs=1)
    t_gen_1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dataset.create_dataset(
        bench_counts, augment=augment, test_per_class=300, n_jobs=max_cores
    )
    t_gen_n = time.perf_counter() - t0

    # 2. Замер обучения нейросети
    print("\n2. Тест параллельного обучения PyTorch (torch.set_num_threads):")
    x_tr, y_tr, _, _ = dataset.create_dataset(
        bench_counts, augment=augment, test_per_class=100, n_jobs=max_cores
    )

    # 1 ядро
    torch.set_num_threads(1)
    net_1 = model.SimpleCNN()
    t0 = time.perf_counter()
    model.train_model(net_1, x_tr, y_tr, epochs=1, verbose=False)
    t_train_1 = time.perf_counter() - t0

    # N ядер
    torch.set_num_threads(max_cores)
    net_n = model.SimpleCNN()
    t0 = time.perf_counter()
    model.train_model(net_n, x_tr, y_tr, epochs=1, verbose=False)
    t_train_n = time.perf_counter() - t0

    # Сводная таблица
    print("\n" + "=" * 68)
    print(f"{'Операция':<26} | {'1 ядро':<11} | {f'{max_cores} ядер':<11} | {'Ускорение':<10}")
    print("-" * 68)
    speedup_gen = t_gen_1 / max(t_gen_n, 1e-6)
    speedup_train = t_train_1 / max(t_train_n, 1e-6)
    print(
        f"{'Генерация данных':<26} | {t_gen_1:<10.3f}с | {t_gen_n:<10.3f}с | {speedup_gen:<9.2f}x"
    )
    print(
        f"{'Обучение 1 эпохи':<26} | {t_train_1:<10.3f}с | {t_train_n:<10.3f}с | {speedup_train:<9.2f}x"
    )
    print("=" * 68)


def menu_cores_config(current_cores: int, max_cores: int) -> int:
    """Подменю настройки распараллеливания на ядра CPU."""
    while True:
        print("\n--- Опция распараллеливания на ядра CPU ---")
        print(f"Всего ядер в системе: {max_cores}")
        print(f"Текущее число рабочих потоков / процессов: {current_cores}")
        print("1. Использовать 1 ядро (однопоточный режим)")
        print(f"2. Использовать все доступные ядра ({max_cores})")
        print("3. Задать произвольное число ядер")
        print("4. Запустить бенчмарк (1 ядро vs все ядра)")
        print("0. Назад в главное меню")

        sub = input("Выберите пункт: ").strip()
        if sub == "1":
            current_cores = 1
            torch.set_num_threads(1)
            print("Установлено: 1 ядро.")
            break
        elif sub == "2":
            current_cores = max_cores
            torch.set_num_threads(max_cores)
            print(f"Установлено: все {max_cores} ядер.")
            break
        elif sub == "3":
            raw = input(f"Введите число ядер (1..{max_cores}): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= max_cores:
                current_cores = int(raw)
                torch.set_num_threads(current_cores)
                print(f"Установлено: {current_cores} ядер.")
                break
            print("Некорректное число ядер.")
        elif sub == "4":
            benchmark_cores(DEFAULT_COUNTS, augment=False, max_cores=max_cores)
        elif sub == "0":
            break
        else:
            print("Неверный ввод.")
    return current_cores


def run_random_check(
    net: torch.nn.Module, x_test: torch.Tensor, y_test: torch.Tensor
) -> None:
    """Проверка случайного рисунка с визуализацией в консоли."""
    if len(x_test) == 0:
        print("Сначала необходимо сгенерировать датасет!")
        return

    idx = random.randint(0, len(x_test) - 1)
    sample = x_test[idx]
    true_cls = int(y_test[idx].item())

    pred_cls, probs = model.predict_single(net, sample)

    print("\n" + "-" * 40)
    print(f"Случайный тестовый образец #{idx}:")
    print(dataset.ascii_art(sample, width=22, height=11))
    print("-" * 40)
    print(f"Истинный класс:      {dataset.CLASSES[true_cls]}")
    print(f"Предсказанный класс: {dataset.CLASSES[pred_cls]}")
    status = "ВЕРНО" if pred_cls == true_cls else "ОШИБКА"
    print(f"Результат:           [{status}]")
    print("Вероятности классов:")
    for c_i, name in enumerate(dataset.CLASSES):
        bar = "█" * int(probs[c_i] * 20)
        print(f"  {name:<12}: {probs[c_i] * 100:5.1f}% | {bar}")
    print("-" * 40)


def run_auto_test(max_cores: int) -> None:
    """Автоматическая проверка всех этапов лабораторной работы."""
    print("\n" + "#" * 68)
    print("     ПОЛНАЯ АВТОМАТИЧЕСКАЯ ПРОВЕРКА ВСЕХ ЭТАПОВ ЛАБОРАТОРНОЙ 2")
    print("#" * 68)

    torch.set_num_threads(max_cores)

    # ЭТАП 1: 3 класса по 3000 картинок (чистые)
    print("\n>>> ЭТАП 1: Генерация 3 классов по 3000 картинок (чистые)")
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        [3000, 3000, 3000], augment=False, test_per_class=600, n_jobs=max_cores
    )
    print(f"Обучающая выборка: {x_tr.shape[0]} шт. | Тестовая: {x_te.shape[0]} шт.")

    net = model.SimpleCNN()
    print("\n>>> Обучение модели на сбалансированных чистых данных:")
    model.train_model(net, x_tr, y_tr, epochs=3, verbose=True)

    print("\n>>> Оценка качества по каждому классу (сбалансированные чистые):")
    model.evaluate_model(net, x_te, y_te, verbose=True)

    # ЭТАП 2: Дисбаланс классов (уменьшение одного класса до 200)
    print("\n>>> ЭТАП 2: Дисбаланс классов (Класс 'Треугольник' уменьшен до 200 шт.)")
    x_tr_imb, y_tr_imb, x_te_imb, y_te_imb = dataset.create_dataset(
        [3000, 3000, 200], augment=False, test_per_class=600, n_jobs=max_cores
    )
    print(f"Обучающая выборка с дисбалансом: {x_tr_imb.shape[0]} шт. [3000, 3000, 200]")

    net_imb = model.SimpleCNN()
    print("\n>>> Обучение модели на несбалансированных данных:")
    model.train_model(net_imb, x_tr_imb, y_tr_imb, epochs=3, verbose=True)

    print("\n>>> Оценка качества по каждому классу при дисбалансе:")
    model.evaluate_model(net_imb, x_te_imb, y_te_imb, verbose=True)

    # ЭТАП 3: Искажения (шумы, повороты, битые пиксели, цвета)
    print("\n>>> ЭТАП 3: Добавление искажений (шум, поворот, битые пиксели, цвета)")
    x_tr_aug, y_tr_aug, x_te_aug, y_te_aug = dataset.create_dataset(
        [3000, 3000, 3000], augment=True, test_per_class=600, n_jobs=max_cores
    )
    net_aug = model.SimpleCNN()
    print("\n>>> Обучение модели на данных с искажениями:")
    model.train_model(net_aug, x_tr_aug, y_tr_aug, epochs=4, verbose=True)

    print("\n>>> Оценка качества по каждому классу на искаженных данных:")
    model.evaluate_model(net_aug, x_te_aug, y_te_aug, verbose=True)

    # ЭТАП 4: Проверка случайных рисунков
    print("\n>>> ЭТАП 4: Проверка случайных рисунков с ASCII-отрисовкой:")
    for _ in range(2):
        run_random_check(net_aug, x_te_aug, y_te_aug)

    # ЭТАП 5: Бенчмарк распараллеливания на ядра
    print("\n>>> ЭТАП 5: Замер параллелизма CPU (1 ядро vs N ядер):")
    benchmark_cores([1500, 1500, 1500], augment=False, max_cores=max_cores)

    print("\n[OK] Полная автоматическая проверка завершена успешно!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Лабораторная работа 2")
    parser.add_argument(
        "--auto", action="store_true", help="Запустить полный авто-тест всех этапов"
    )
    args = parser.parse_args()

    max_cores = os.cpu_count() or 4
    current_cores = max_cores
    torch.set_num_threads(current_cores)

    if args.auto:
        run_auto_test(max_cores)
        return

    # Состояние приложения
    counts = list(DEFAULT_COUNTS)
    augment = False
    net = model.SimpleCNN()
    x_train, y_train, x_test, y_test = None, None, None, None

    while True:
        print_banner(current_cores, counts, augment)
        print("1. Сгенерировать датасет (3 класса, размер 32x32)")
        print("2. Обучить модель на текущем датасете")
        print("3. Автоматическая проверка на тестовой выборке (метрики по классам)")
        print("4. Проверить случайный рисунок (ASCII-визуализация в консоли)")
        print("5. Уменьшить количество одного класса (дисбаланс: 100-300)")
        print("6. Переключить искажения (шум, повороты, битые пиксели, цвета)")
        print("7. Настройка распараллеливания на ядра CPU (выбор ядер + бенчмарк)")
        print("8. Полный авто-тест всех этапов задания (режим демонстрации)")
        print("9. Сохранить / Загрузить веса модели")
        print("0. Выход")

        choice = input("Выберите действие: ").strip()

        if choice == "1":
            print(
                f"\nГенерация датасета {counts} (искажения: {augment}, ядер: {current_cores})..."
            )
            t0 = time.perf_counter()
            x_train, y_train, x_test, y_test = dataset.create_dataset(
                counts, augment=augment, test_per_class=600, n_jobs=current_cores
            )
            dt = time.perf_counter() - t0
            print(
                f"Готово за {dt:.2f} сек. Обучающих: {len(y_train)}, тестовых: {len(y_test)}."
            )

        elif choice == "2":
            if x_train is None:
                print("\nСначала сгенерируйте датасет (пункт 1)!")
                continue
            raw_ep = input("Количество эпох [4]: ").strip()
            epochs = int(raw_ep) if raw_ep.isdigit() else 4
            print(f"\nОбучение модели ({epochs} эпох, ядер: {current_cores})...")
            model.train_model(net, x_train, y_train, epochs=epochs, verbose=True)

        elif choice == "3":
            if x_test is None:
                print("\nСначала сгенерируйте датасет (пункт 1)!")
                continue
            print("\nАвтоматическая проверка модели на тестовой выборке:")
            model.evaluate_model(net, x_test, y_test, verbose=True)

        elif choice == "4":
            if x_test is None:
                print("\nСначала сгенерируйте датасет (пункт 1)!")
                continue
            run_random_check(net, x_test, y_test)

        elif choice == "5":
            print("\nНастройка дисбаланса классов:")
            for i, name in enumerate(dataset.CLASSES):
                print(f"  {i}: {name}")
            raw_c = input("Выберите класс для уменьшения [2 - Треугольник]: ").strip()
            cls_idx = (
                int(raw_c) if raw_c.isdigit() and 0 <= int(raw_c) <= 2 else 2
            )
            raw_n = input("Количество примеров для этого класса (100..300) [200]: ").strip()
            cnt = int(raw_n) if raw_n.isdigit() and 50 <= int(raw_n) <= 1000 else 200

            counts = list(DEFAULT_COUNTS)
            counts[cls_idx] = cnt
            print(f"Новый размер выборки: {counts}")
            print("Перегенерируем датасет с дисбалансом...")
            t0 = time.perf_counter()
            x_train, y_train, x_test, y_test = dataset.create_dataset(
                counts, augment=augment, test_per_class=600, n_jobs=current_cores
            )
            print(f"Датасет обновлен за {time.perf_counter() - t0:.2f} сек.")

        elif choice == "6":
            augment = not augment
            status_str = "ВКЛЮЧЕНЫ" if augment else "ВЫКЛЮЧЕНЫ"
            print(f"\nИскажения (шум, повороты, битые пиксели, цвета): {status_str}")
            ans = input("Перегенерировать текущий датасет? (д/н) [д]: ").strip().lower()
            if ans != "н":
                t0 = time.perf_counter()
                x_train, y_train, x_test, y_test = dataset.create_dataset(
                    counts, augment=augment, test_per_class=600, n_jobs=current_cores
                )
                print(f"Датасет перегенерирован за {time.perf_counter() - t0:.2f} сек.")

        elif choice == "7":
            current_cores = menu_cores_config(current_cores, max_cores)

        elif choice == "8":
            run_auto_test(max_cores)

        elif choice == "9":
            sub = input("1. Сохранить веса в model.pt\n2. Загрузить веса из model.pt\nВыберите: ").strip()
            if sub == "1":
                model.save_model(net, "model.pt")
            elif sub == "2":
                try:
                    model.load_model(net, "model.pt")
                except FileNotFoundError:
                    print("Файл 'model.pt' не найден.")
                except (RuntimeError, OSError) as e:
                    print(f"Ошибка загрузки: {e}")

        elif choice == "0":
            print("\nВыход.")
            break
        else:
            print("Неверный ввод, повторите попытку.")


if __name__ == "__main__":
    main()
