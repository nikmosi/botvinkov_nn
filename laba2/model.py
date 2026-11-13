"""Модуль модели: многослойный перцептрон (MLP), совместимый с laba1/my.py."""

from my import (
    backward,
    evaluate,
    forward,
    init_network,
    load_model,
    predict_image_file,
    predict_single,
    save_model,
    sigmoid,
    sigmoid_derivative,
    train,
)

# Алиасы для обратной совместимости
train_model = train
evaluate_model = evaluate

__all__ = [
    "backward",
    "evaluate",
    "evaluate_model",
    "forward",
    "init_network",
    "load_model",
    "predict_image_file",
    "predict_single",
    "save_model",
    "sigmoid",
    "sigmoid_derivative",
    "train",
    "train_model",
]
