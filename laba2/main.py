import argparse
import os
import random
import time
from typing import Any

import torch

import comp
import dataset
import my

DEFAULT_COUNTS = [3000, 3000, 3000]
IMAGE_SIZE = dataset.DEFAULT_IMAGE_SIZE  # 128x128


def print_banner(cores: int, counts: list[int], augment: bool) -> None:
    mode_str = "Сбалансированный" if len(set(counts)) == 1 else "Дисбаланс"
    aug_str = "ВКЛ (шум, повороты, пиксели, цвета)" if augment else "ВЫКЛ (чистые)"
    print("\n" + "=" * 68)
    print("      Лабораторная работа №2: Простой перцептрон (MLP)")
    print(
        f"  Датасет: {counts} ({mode_str}) | Размер: {IMAGE_SIZE}x{IMAGE_SIZE} | Искажения: {aug_str}"
    )
    print(f"  Активные ядра CPU: {cores} из {os.cpu_count() or 1}")
    print("=" * 68)


def benchmark_cores(counts: list[int], augment: bool, max_cores: int) -> None:
    """Бенчмарк производительности: 1 ядро vs N ядер (по образцу laba1)."""
    print("\n--- Запуск сравнительного бенчмарка (1 ядро vs N ядер) ---")
    bench_counts = [min(c, 800) for c in counts]
    print(f"Тестовая выборка для замера: {bench_counts}")

    # 1. Замер генерации данных
    print("\n1. Тест многопроцессорной генерации данных (multiprocessing.Pool):")
    t0 = time.perf_counter()
    dataset.create_dataset(
        bench_counts, augment=augment, test_per_class=200, n_jobs=1, size=IMAGE_SIZE
    )
    t_gen_1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dataset.create_dataset(
        bench_counts,
        augment=augment,
        test_per_class=200,
        n_jobs=max_cores,
        size=IMAGE_SIZE,
    )
    t_gen_n = time.perf_counter() - t0

    # 2. Замер обучения перцептрона
    print("\n2. Тест параллельного обучения перцептрона (torch.set_num_threads):")
    x_tr, y_tr, _, _ = dataset.create_dataset(
        bench_counts,
        augment=augment,
        test_per_class=50,
        n_jobs=max_cores,
        size=IMAGE_SIZE,
    )

    d_in = x_tr.shape[1] * x_tr.shape[2] * x_tr.shape[3]

    # 1 ядро
    torch.set_num_threads(1)
    net_1 = my.init_network([d_in, 256, 3])
    t0 = time.perf_counter()
    my.train(net_1, x_tr, y_tr, epochs=1, batch_size=32, verbose=False)
    t_train_1 = time.perf_counter() - t0

    # N ядер (оптимальное число потоков)
    opt_cores = min(max_cores, 4)
    torch.set_num_threads(opt_cores)
    net_n = my.init_network([d_in, 256, 3])
    t0 = time.perf_counter()
    my.train(net_n, x_tr, y_tr, epochs=1, batch_size=32, verbose=False)
    t_train_n = time.perf_counter() - t0

    torch.set_num_threads(max_cores)

    # Сводная таблица
    print("\n" + "=" * 68)
    print(
        f"{'Операция':<26} | {'1 ядро':<11} | {f'{opt_cores} ядер':<11} | {'Ускорение':<10}"
    )
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
        print(f"Текущее число рабочих потоков: {current_cores}")
        print("1. Использовать 1 ядро (однопоточный режим)")
        print(f"2. Использовать все доступные ядра ({max_cores})")
        print("3. Задать произвольное число ядер")
        print("4. Запустить бенчмарк (1 ядро vs N ядер)")
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
    network: dict[str, Any], x_test: torch.Tensor, y_test: torch.Tensor
) -> None:
    """Проверка случайного рисунка с визуализацией в консоли."""
    if len(x_test) == 0:
        print("Сначала необходимо сгенерировать датасет!")
        return

    idx = random.randint(0, len(x_test) - 1)
    sample = x_test[idx]
    true_cls = int(y_test[idx].item())

    pred_cls, probs = my.predict_single(network, sample)

    print("\n" + "-" * 40)
    print(f"Случайный тестовый образец #{idx} ({IMAGE_SIZE}x{IMAGE_SIZE}):")
    print(dataset.ascii_art(sample, width=24, height=12))
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


def run_custom_image_check(
    network: dict[str, Any], image_path: str, quiet: bool = False
) -> int:
    """Распознавание произвольной картинки по пути к файлу с диска."""
    if not os.path.exists(image_path):
        if not quiet:
            print(f"Ошибка: файл '{image_path}' не найден.")
        return -1

    tensor = dataset.load_image_file(image_path, size=IMAGE_SIZE)
    pred_cls, probs = my.predict_single(network, tensor)

    if not quiet:
        print("\n" + "-" * 40)
        print(f"Пользовательское изображение: {image_path}")
        print(dataset.ascii_art(tensor, width=24, height=12))
        print("-" * 40)
        print(f"Предсказанный класс: {dataset.CLASSES[pred_cls]}")
        print("Вероятности классов:")
        for c_i, name in enumerate(dataset.CLASSES):
            bar = "█" * int(probs[c_i] * 20)
            print(f"  {name:<12}: {probs[c_i] * 100:5.1f}% | {bar}")
        print("-" * 40)
    else:
        print(dataset.CLASSES[pred_cls])

    return pred_cls


def run_auto_test(max_cores: int) -> None:
    """Автоматическая проверка всех этапов лабораторной работы на перцептроне."""
    print("\n" + "#" * 68)
    print(f"   ПОЛНАЯ АВТОМАТИЧЕСКАЯ ПРОВЕРКА: ПЕРЦЕПТРОН ({IMAGE_SIZE}x{IMAGE_SIZE})")
    print("#" * 68)

    torch.set_num_threads(min(max_cores, 4))
    d_in = IMAGE_SIZE * IMAGE_SIZE * 3

    # ЭТАП 1: 3 класса по 3000 картинок (чистые)
    print(
        f"\n>>> ЭТАП 1: Генерация 3 классов по 3000 картинок ({IMAGE_SIZE}x{IMAGE_SIZE})"
    )
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        [3000, 3000, 3000],
        augment=False,
        test_per_class=600,
        n_jobs=max_cores,
        size=IMAGE_SIZE,
    )
    print(f"Обучающая выборка: {x_tr.shape[0]} шт. | Тестовая: {x_te.shape[0]} шт.")

    net = my.init_network([d_in, 256, 3])
    print(f"\n>>> Обучение перцептрона ([{d_in}, 256, 3], ~1 мин)...")
    my.train(net, x_tr, y_tr, epochs=6, batch_size=32, verbose=True)

    print("\n>>> Оценка качества по каждому классу (сбалансированные чистые):")
    my.evaluate(net, x_te, y_te, verbose=True)

    # ЭТАП 2: Дисбаланс классов (уменьшение одного класса до 200)
    print("\n>>> ЭТАП 2: Дисбаланс классов (Класс 'Треугольник' уменьшен до 200 шт.)")
    x_tr_imb, y_tr_imb, x_te_imb, y_te_imb = dataset.create_dataset(
        [3000, 3000, 200],
        augment=False,
        test_per_class=600,
        n_jobs=max_cores,
        size=IMAGE_SIZE,
    )
    print(f"Обучающая выборка с дисбалансом: {x_tr_imb.shape[0]} шт. [3000, 3000, 200]")

    net_imb = my.init_network([d_in, 256, 3])
    print("\n>>> Обучение перцептрона на несбалансированных данных:")
    my.train(net_imb, x_tr_imb, y_tr_imb, epochs=6, batch_size=32, verbose=True)

    print("\n>>> Оценка качества по каждому классу при дисбалансе:")
    my.evaluate(net_imb, x_te_imb, y_te_imb, verbose=True)

    # ЭТАП 3: Искажения (шумы, повороты, битые пиксели, цвета)
    print("\n>>> ЭТАП 3: Добавление искажений (шум, поворот, битые пиксели, цвета)")
    x_tr_aug, y_tr_aug, x_te_aug, y_te_aug = dataset.create_dataset(
        [3000, 3000, 3000],
        augment=True,
        test_per_class=600,
        n_jobs=max_cores,
        size=IMAGE_SIZE,
    )
    net_aug = my.init_network([d_in, 256, 3])
    print("\n>>> Обучение перцептрона на данных с искажениями:")
    my.train(net_aug, x_tr_aug, y_tr_aug, epochs=6, batch_size=32, verbose=True)

    print("\n>>> Оценка качества по каждому классу на искаженных данных:")
    my.evaluate(net_aug, x_te_aug, y_te_aug, verbose=True)

    # ЭТАП 4: Проверка случайных рисунков
    print("\n>>> ЭТАП 4: Проверка случайных рисунков с ASCII-отрисовкой:")
    for _ in range(2):
        run_random_check(net_aug, x_te_aug, y_te_aug)

    # ЭТАП 5: Бенчмарк распараллеливания на ядра
    print("\n>>> ЭТАП 5: Замер параллелизма CPU (1 ядро vs N ядер):")
    benchmark_cores([800, 800, 800], augment=False, max_cores=max_cores)

    print("\n[OK] Полная автоматическая проверка завершена успешно!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа 2: Простой перцептрон"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Запустить полный авто-тест всех этапов"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Путь к файлу изображения для распознавания",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Запустить сравнение с PyTorch (comp.py)"
    )
    parser.add_argument(
        "--save-sample",
        type=str,
        default=None,
        help="Сохранить образец картинки на диск",
    )
    parser.add_argument(
        "--cls", type=int, default=0, help="Класс для сохранения образца (0..2)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Тихий режим без лишнего вывода"
    )
    args = parser.parse_args()

    max_cores = os.cpu_count() or 4
    current_cores = min(max_cores, 4)
    torch.set_num_threads(current_cores)

    d_in = IMAGE_SIZE * IMAGE_SIZE * 3
    if os.path.exists("model.json"):
        try:
            net = my.load_model("model.json", verbose=False)
        except (OSError, KeyError, RuntimeError, my.json.JSONDecodeError):
            net = my.init_network([d_in, 256, 3])
    else:
        net = my.init_network([d_in, 256, 3])

    if args.save_sample:
        raw_img = dataset.generate_single_image(
            cls=args.cls, augment=False, size=IMAGE_SIZE
        )
        dataset.save_image_file(torch.from_numpy(raw_img), args.save_sample)
        if not args.quiet:
            print(
                f"Образец класса '{dataset.CLASSES[args.cls]}' сохранен в '{args.save_sample}'."
            )
        return

    if args.image:
        run_custom_image_check(net, args.image, quiet=args.quiet)
        return

    if args.compare:
        comp.compare_models(
            counts=[400, 400, 400], epochs=3, size=IMAGE_SIZE, verbose=not args.quiet
        )
        return

    if args.auto:
        run_auto_test(max_cores)
        return

    # Состояние приложения: простой перцептрон (как в laba1/my.py)
    counts = list(DEFAULT_COUNTS)
    augment = False
    x_train, y_train, x_test, y_test = None, None, None, None

    while True:
        if not args.quiet:
            print_banner(current_cores, counts, augment)
        print(f"1. Сгенерировать датасет (3 класса, размер {IMAGE_SIZE}x{IMAGE_SIZE})")
        print("2. Обучить на текущем датасете")
        print("3. Автоматическая проверка на тестовой выборке")
        print("4. Проверить случайный рисунок (из памяти)")
        print("5. Распознать свою картинку по пути к файлу")
        print("6. Сохранить образец картинки на диск")
        print("7. Сравнение с PyTorch (lib.py / comp.py)")
        print("8. Уменьшить количество одного класса")
        print("9. Переключить искажения")
        print("10. Настройка распараллеливания на ядра CPU")
        print("11. Полный авто-тест всех этапов задания")
        print("12. Сохранить / Загрузить (model.json)")
        print("0. Выход")

        choice = input("Выберите действие: ").strip()

        if choice == "1":
            print(
                f"\nГенерация датасета {counts} (размер: {IMAGE_SIZE}x{IMAGE_SIZE}, искажения: {augment}, ядер: {current_cores})..."
            )
            t0 = time.perf_counter()
            x_train, y_train, x_test, y_test = dataset.create_dataset(
                counts,
                augment=augment,
                test_per_class=600,
                n_jobs=current_cores,
                size=IMAGE_SIZE,
            )
            dt = time.perf_counter() - t0
            print(
                f"Готово за {dt:.2f} сек. Обучающих: {len(y_train)}, тестовых: {len(y_test)}."
            )

        elif choice == "2":
            if x_train is None:
                print("\nСначала сгенерируйте датасет (пункт 1)!")
                continue
            raw_ep = input("Количество эпох [6]: ").strip()
            epochs = int(raw_ep) if raw_ep.isdigit() else 6
            raw_b = input("Размер батча (1 для обучения без батча) [32]: ").strip()
            batch_sz = int(raw_b) if raw_b.isdigit() and int(raw_b) >= 1 else 32
            print(
                f"\nОбучение перцептрона ({epochs} эпох, батч {batch_sz}, ~1 мин, ядер: {current_cores})..."
            )
            my.train(
                net, x_train, y_train, epochs=epochs, batch_size=batch_sz, verbose=True
            )

        elif choice == "3":
            if x_test is None:
                print("\nСначала сгенерируйте датасет (пункт 1)!")
                continue
            print("\nАвтоматическая проверка перцептрона на тестовой выборке:")
            my.evaluate(net, x_test, y_test, verbose=True)

        elif choice == "4":
            if x_test is None:
                print("\nСначала сгенерируйте датасет (пункт 1)!")
                continue
            run_random_check(net, x_test, y_test)

        elif choice == "5":
            path = input("Введите путь к картинке (PNG/JPG): ").strip()
            if path:
                run_custom_image_check(net, path)

        elif choice == "6":
            for i, name in enumerate(dataset.CLASSES):
                print(f"  {i}: {name}")
            raw_c = input("Выберите класс для сохранения [0]: ").strip()
            cls_idx = int(raw_c) if raw_c.isdigit() and 0 <= int(raw_c) <= 2 else 0
            fname = (
                input(f"Имя файла [sample_{dataset.CLASSES[cls_idx]}.png]: ").strip()
                or f"sample_{dataset.CLASSES[cls_idx]}.png"
            )
            raw_img = dataset.generate_single_image(
                cls=cls_idx, augment=augment, size=IMAGE_SIZE
            )
            dataset.save_image_file(torch.from_numpy(raw_img), fname)
            print(f"Картинка успешно сохранена в '{fname}'.")

        elif choice == "7":
            comp.compare_models(
                counts=[400, 400, 400], epochs=3, size=IMAGE_SIZE, verbose=True
            )

        elif choice == "8":
            print("\nНастройка дисбаланса классов:")
            for i, name in enumerate(dataset.CLASSES):
                print(f"  {i}: {name}")
            raw_c = input("Выберите класс для уменьшения [2 - Треугольник]: ").strip()
            cls_idx = int(raw_c) if raw_c.isdigit() and 0 <= int(raw_c) <= 2 else 2
            raw_n = input(
                "Количество примеров для этого класса (100..300) [200]: "
            ).strip()
            cnt = int(raw_n) if raw_n.isdigit() and 50 <= int(raw_n) <= 1000 else 200

            counts = list(DEFAULT_COUNTS)
            counts[cls_idx] = cnt
            print(f"Новый размер выборки: {counts}")
            print("Перегенерируем датасет с дисбалансом...")
            t0 = time.perf_counter()
            x_train, y_train, x_test, y_test = dataset.create_dataset(
                counts,
                augment=augment,
                test_per_class=600,
                n_jobs=current_cores,
                size=IMAGE_SIZE,
            )
            print(f"Датасет обновлен за {time.perf_counter() - t0:.2f} сек.")

        elif choice == "9":
            augment = not augment
            status_str = "ВКЛЮЧЕНЫ" if augment else "ВЫКЛЮЧЕНЫ"
            print(f"\nИскажения (шум, повороты, битые пиксели, цвета): {status_str}")
            ans = input("Перегенерировать текущий датасет? (д/н) [д]: ").strip().lower()
            if ans != "н":
                t0 = time.perf_counter()
                x_train, y_train, x_test, y_test = dataset.create_dataset(
                    counts,
                    augment=augment,
                    test_per_class=600,
                    n_jobs=current_cores,
                    size=IMAGE_SIZE,
                )
                print(f"Датасет перегенерирован за {time.perf_counter() - t0:.2f} сек.")

        elif choice == "10":
            current_cores = menu_cores_config(current_cores, max_cores)

        elif choice == "11":
            run_auto_test(max_cores)

        elif choice == "12":
            sub = input(
                "1. Сохранить веса в model.json\n2. Загрузить веса из model.json\nВыберите: "
            ).strip()
            if sub == "1":
                my.save_model(net, "model.json", verbose=True)
            elif sub == "2":
                try:
                    net = my.load_model("model.json", verbose=True)
                except FileNotFoundError:
                    print("Файл 'model.json' не найден.")
                except (RuntimeError, OSError, my.json.JSONDecodeError) as e:
                    print(f"Ошибка загрузки: {e}")

        elif choice == "0":
            print("\nВыход.")
            break
        else:
            print("Неверный ввод, повторите попытку.")


if __name__ == "__main__":
    main()
