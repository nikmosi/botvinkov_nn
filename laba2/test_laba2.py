import torch

import dataset
import model
import my


def test_dataset_generation():
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        [50, 50, 50], augment=False, test_per_class=20, n_jobs=1
    )
    assert x_tr.shape == (150, 3, 32, 32), f"Unexpected train shape: {x_tr.shape}"
    assert y_tr.shape == (150,), f"Unexpected train labels: {y_tr.shape}"
    assert x_te.shape == (60, 3, 32, 32), f"Unexpected test shape: {x_te.shape}"
    assert y_te.shape == (60,), f"Unexpected test labels: {y_te.shape}"


def test_augmented_generation():
    x_tr, y_tr, x_te, _ = dataset.create_dataset(
        [50, 50, 10], augment=True, test_per_class=15, n_jobs=2
    )
    assert x_tr.shape == (110, 3, 32, 32)
    assert y_tr.shape == (110,)
    assert x_te.shape == (45, 3, 32, 32)


def test_perceptron_training_and_evaluation():
    torch.set_num_threads(2)
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        [60, 60, 60], augment=False, test_per_class=20, n_jobs=1
    )
    net = my.init_network([3072, 32, 3])
    elapsed = my.train(net, x_tr, y_tr, epochs=4, batch_size=32, verbose=False)
    assert elapsed > 0

    metrics = my.evaluate(net, x_te, y_te, verbose=False)
    assert "overall_accuracy" in metrics
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 3
    assert "confusion_matrix" in metrics
    assert metrics["overall_accuracy"] >= 0.80

    pred_cls, probs = my.predict_single(net, x_te[0])
    assert 0 <= pred_cls <= 2
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-4

    # Test backward compatibility in model.py
    pred_c2, _ = model.predict_single(net, x_te[0])
    assert pred_c2 == pred_cls


def test_serialization():
    net = my.init_network([3072, 16, 3])
    my.save_model(net, "test_model.json")
    loaded = my.load_model("test_model.json")
    assert loaded["sizes"] == net["sizes"]
    assert len(loaded["weights"]) == len(net["weights"])
    assert torch.allclose(loaded["weights"][0], net["weights"][0])
    import os

    if os.path.exists("test_model.json"):
        os.remove("test_model.json")


def test_ascii_art():
    sample = torch.rand(3, 32, 32)
    art = dataset.ascii_art(sample, width=16, height=8)
    lines = art.split("\n")
    assert len(lines) == 8
    assert len(lines[0]) == 16


if __name__ == "__main__":
    print("Запуск тестов laba2 (простой перцептрон)...")
    test_dataset_generation()
    print("- test_dataset_generation OK")
    test_augmented_generation()
    print("- test_augmented_generation OK")
    test_perceptron_training_and_evaluation()
    print("- test_perceptron_training_and_evaluation OK")
    test_serialization()
    print("- test_serialization OK")
    test_ascii_art()
    print("- test_ascii_art OK")
    print("Все тесты успешно пройдены!")
