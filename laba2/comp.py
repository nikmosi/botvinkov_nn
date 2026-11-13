import time
from typing import Any

import dataset
import lib
import my


def compare_models(
    counts: list[int] | None = None,
    epochs: int = 4,
    batch_size: int = 32,
    size: int = dataset.DEFAULT_IMAGE_SIZE,
    verbose: bool = False,
) -> dict[str, Any]:
    """Сравнительное тестирование перцептрона my.py и PyTorch lib.py (как в laba1/comp.py)."""
    if counts is None:
        counts = [600, 600, 600]

    d_in = size * size * 3
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        counts, augment=False, test_per_class=100, n_jobs=1, size=size
    )

    # 1. Обучение my.py
    net_my = my.init_network([d_in, 128, 3])
    t0 = time.perf_counter()
    my.train(net_my, x_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=False)
    t_my = time.perf_counter() - t0
    eval_my = my.evaluate(net_my, x_te, y_te, verbose=False)

    # 2. Обучение PyTorch lib.py
    net_lib = lib.create_model([d_in, 128, 3])
    t0 = time.perf_counter()
    lib.train(net_lib, x_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=False)
    t_lib = time.perf_counter() - t0
    eval_lib = lib.evaluate(net_lib, x_te, y_te, verbose=False)

    speedup = t_my / max(t_lib, 1e-6)
    results = {
        "time_my": t_my,
        "time_torch": t_lib,
        "acc_my": eval_my["overall_accuracy"],
        "acc_torch": eval_lib["overall_accuracy"],
        "speedup": speedup,
        "samples": len(y_tr),
    }

    if verbose:
        print("\n" + "=" * 58)
        print(f"{'Метрика':<24} | {'my.py (чистый)':<14} | {'lib.py (PyTorch)':<14}")
        print("-" * 58)
        print(f"{'Время обучения':<24} | {t_my:<13.3f}с | {t_lib:<13.3f}с")
        print(
            f"{'Точность (Accuracy)':<24} | {results['acc_my'] * 100:<13.2f}% | {results['acc_torch'] * 100:<13.2f}%"
        )
        print(f"{'Ускорение PyTorch':<24} | 1.00x          | {speedup:<13.2f}x")
        print("=" * 58)

    return results


if __name__ == "__main__":
    compare_models(counts=[400, 400, 400], epochs=3, size=64, verbose=True)
