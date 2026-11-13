import contextlib
import io
import os

import torch

import comp
import dataset
import lib
import my


def test_batch_vs_no_batch():
    """Тест: обучение с батчем и без батча (batch_size=1 vs batch_size>1)."""
    x = torch.rand(10, 48)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # С батчем
    net_batch = my.init_network([48, 16, 3])
    t_batch = my.train(net_batch, x, y, epochs=2, batch_size=4, verbose=False)
    assert t_batch > 0

    # Без батча (stochastic gradient descent, batch_size=1)
    net_nobatch = my.init_network([48, 16, 3])
    t_nobatch = my.train(net_nobatch, x, y, epochs=2, batch_size=1, verbose=False)
    assert t_nobatch > 0


def test_multiprocessing():
    """Тест: генерация выборки с мультипроцессингом (n_jobs > 1)."""
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        [10, 10, 10], augment=False, test_per_class=5, n_jobs=2, size=16
    )
    assert x_tr.shape == (30, 3, 16, 16)
    assert y_tr.shape == (30,)
    assert x_te.shape == (15, 3, 16, 16)
    assert y_te.shape == (15,)


def test_no_multiprocessing():
    """Тест: генерация выборки без мультипроцессинга (n_jobs = 1)."""
    x_tr, y_tr, x_te, y_te = dataset.create_dataset(
        [10, 10, 10], augment=False, test_per_class=5, n_jobs=1, size=16
    )
    assert x_tr.shape == (30, 3, 16, 16)
    assert y_tr.shape == (30,)
    assert x_te.shape == (15, 3, 16, 16)
    assert y_te.shape == (15,)


def test_disk_saving():
    """Тест: сохранение на диск весов моделей (my/lib) и изображений."""
    # 1. Сохранение и загрузка my.py
    net = my.init_network([48, 16, 3])
    my.save_model(net, "tmp_my_model.json", verbose=False)
    assert os.path.exists("tmp_my_model.json")
    loaded = my.load_model("tmp_my_model.json", verbose=False)
    assert loaded["sizes"] == net["sizes"]
    os.remove("tmp_my_model.json")

    # 2. Сохранение и загрузка PyTorch lib.py
    torch_model = lib.create_model([48, 16, 3])
    lib.save_model(torch_model, "tmp_torch.pt", verbose=False)
    assert os.path.exists("tmp_torch.pt")
    lib.load_model(torch_model, "tmp_torch.pt", verbose=False)
    os.remove("tmp_torch.pt")

    # 3. Сохранение изображения на диск
    img = torch.rand(3, 32, 32)
    dataset.save_image_file(img, "tmp_sample.png")
    assert os.path.exists("tmp_sample.png")
    os.remove("tmp_sample.png")


def test_custom_image_prediction():
    """Тест: возможность самому указывать картинку (по пути к файлу)."""
    # Создаем тестовую картинку на диске
    raw_img = dataset.generate_single_image(cls=0, augment=False, size=32)
    dataset.save_image_file(torch.from_numpy(raw_img), "tmp_circle.png")

    net = my.init_network([32 * 32 * 3, 16, 3])
    pred_cls, probs = my.predict_image_file(net, "tmp_circle.png", size=32)
    assert 0 <= pred_cls <= 2
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-4

    torch_net = lib.create_model([32 * 32 * 3, 16, 3])
    pred_t, probs_t = lib.predict_image_file(torch_net, "tmp_circle.png", size=32)
    assert 0 <= pred_t <= 2
    assert len(probs_t) == 3

    if os.path.exists("tmp_circle.png"):
        os.remove("tmp_circle.png")


def test_comparison_with_torch():
    """Тест: сравнение перцептрона с PyTorch (время, точность)."""
    res = comp.compare_models(counts=[20, 20, 20], epochs=1, size=16, verbose=False)
    assert "time_my" in res
    assert "time_torch" in res
    assert "acc_my" in res
    assert "acc_torch" in res
    assert res["time_my"] >= 0
    assert res["time_torch"] >= 0


def test_no_console_output():
    """Тест: проверка отсутствия вывода в консоль при verbose=False."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net = my.init_network([48, 16, 3])
        x = torch.rand(6, 48)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        my.train(net, x, y, epochs=1, verbose=False)
        my.evaluate(net, x, y, verbose=False)
        my.save_model(net, "tmp_silent.json", verbose=False)
        my.load_model("tmp_silent.json", verbose=False)
        if os.path.exists("tmp_silent.json"):
            os.remove("tmp_silent.json")

    assert buf.getvalue() == "", f"Ожидался тихий режим, но выведено: {buf.getvalue()}"


def test_ascii_art():
    sample = torch.rand(3, 128, 128)
    art = dataset.ascii_art(sample, width=16, height=8)
    lines = art.split("\n")
    assert len(lines) == 8
    assert len(lines[0]) == 16


if __name__ == "__main__":
    test_batch_vs_no_batch()
    test_multiprocessing()
    test_no_multiprocessing()
    test_disk_saving()
    test_custom_image_prediction()
    test_comparison_with_torch()
    test_no_console_output()
    test_ascii_art()
    print("Все тесты успешно пройдены!")
