"""Веб-интерфейс для Лабораторной работы №2 (MLP классификация геометрических фигур).

Запуск:
    uv run python web.py [--port 8080] [--host 0.0.0.0]
Для доступа через SSH port forward:
    ssh -L 8080:localhost:8080 user@host
    Затем открыть: http://localhost:8080
"""

import argparse
import base64
import io
import json
import os
import random
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image

import comp
import dataset
import my

IMAGE_SIZE = dataset.DEFAULT_IMAGE_SIZE  # 128
DEFAULT_INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE * 3  # 49152


class AppState:
    """Глобальное состояние веб-приложения с потокобезопасной блокировкой."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.max_cores = os.cpu_count() or 4
        self.cores = min(self.max_cores, 4)
        torch.set_num_threads(self.cores)

        if os.path.exists("model.json"):
            try:
                self.net = my.load_model("model.json", verbose=False)
            except (OSError, KeyError, RuntimeError, my.json.JSONDecodeError):
                self.net = my.init_network([DEFAULT_INPUT_SIZE, 256, 3])
        else:
            self.net = my.init_network([DEFAULT_INPUT_SIZE, 256, 3])

        self.counts = [1000, 1000, 1000]
        self.augment = False
        self.x_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.x_test: torch.Tensor | None = None
        self.y_test: torch.Tensor | None = None
        self.last_train_info: dict[str, Any] | None = None


STATE = AppState()


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """Конвертирует PyTorch тензор (3, H, W) [0..1] в base64 PNG data URL."""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def pil_to_base64_png(img: Image.Image) -> str:
    """Конвертирует Pillow Image в base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Лабораторная работа №2: MLP Классификатор фигур</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-dim: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent-circle: #38bdf8;
      --accent-square: #a855f7;
      --accent-triangle: #f97316;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 1.5rem;
      line-height: 1.5;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      margin-bottom: 1.5rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1rem;
    }
    h1 { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .status-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .badge {
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      color: var(--text-dim);
    }
    .badge strong { color: var(--primary); }

    .ssh-notice {
      background: #1e1b4b;
      border: 1px solid #4338ca;
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .ssh-notice code {
      background: #312e81;
      padding: 0.2rem 0.5rem;
      border-radius: 0.25rem;
      font-family: monospace;
      color: #a5b4fc;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 1.5rem;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 600;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
    }
    .btn {
      background: var(--primary);
      color: #0f172a;
      border: none;
      font-weight: 600;
      padding: 0.6rem 1rem;
      border-radius: 0.5rem;
      cursor: pointer;
      font-size: 0.9rem;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
    }
    .btn:hover:not(:disabled) { background: var(--primary-hover); color: #fff; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-secondary {
      background: #334155;
      color: var(--text);
    }
    .btn-secondary:hover:not(:disabled) {
      background: #475569;
    }
    .btn-group { display: flex; gap: 0.5rem; flex-wrap: wrap; }

    /* Drag & drop upload area */
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 0.5rem;
      padding: 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(56, 189, 248, 0.05);
    }
    .dropzone p { color: var(--text-dim); font-size: 0.9rem; margin-top: 0.5rem; }

    /* Prediction view */
    .prediction-box {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
      background: #0f172a;
      border-radius: 0.5rem;
      padding: 1rem;
      border: 1px solid var(--border);
    }
    .preview-img {
      width: 140px;
      height: 140px;
      object-fit: contain;
      image-rendering: pixelated;
      border: 2px solid var(--border);
      border-radius: 0.35rem;
      background: #000;
    }
    .pred-result {
      width: 100%;
      text-align: center;
    }
    .pred-badge {
      display: inline-block;
      font-size: 1.25rem;
      font-weight: 700;
      padding: 0.25rem 1rem;
      border-radius: 9999px;
      margin-bottom: 0.5rem;
    }
    .bar-group { display: flex; flex-direction: column; gap: 0.5rem; width: 100%; }
    .bar-item { display: flex; flex-direction: column; gap: 0.2rem; }
    .bar-label { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-dim); }
    .bar-bg {
      background: #334155;
      height: 10px;
      border-radius: 5px;
      overflow: hidden;
      width: 100%;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      transition: width 0.4s ease-out;
      border-radius: 5px;
    }
    .fill-circle { background: var(--accent-circle); }
    .fill-square { background: var(--accent-square); }
    .fill-triangle { background: var(--accent-triangle); }

    /* Form inputs */
    .form-row {
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }
    label { font-size: 0.85rem; color: var(--text-dim); display: block; margin-bottom: 0.25rem; }
    input[type="number"], select {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.5rem 0.75rem;
      border-radius: 0.35rem;
      width: 100%;
      font-size: 0.9rem;
    }
    .checkbox-label {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.9rem;
      color: var(--text);
      cursor: pointer;
    }

    /* Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }
    th, td {
      padding: 0.5rem 0.6rem;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--text-dim); font-weight: 600; }
    .table-container { overflow-x: auto; }

    /* Output console */
    .console-out {
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 0.75rem;
      font-family: monospace;
      font-size: 0.8rem;
      color: #38bdf8;
      max-height: 180px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
    .spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎨 Классификация фигур (MLP 128×128)</h1>
        <p style="color: var(--text-dim); font-size: 0.85rem;">Лабораторная работа №2 — Многослойный перцептрон с нуля</p>
      </div>
      <div class="status-badges">
        <div class="badge">Ядра CPU: <strong id="coresBadge">-</strong></div>
        <div class="badge">Обучающих: <strong id="trainBadge">0</strong></div>
        <div class="badge">Тестовых: <strong id="testBadge">0</strong></div>
      </div>
    </header>

    <div class="ssh-notice">
      <div>
        <strong>Доступ через SSH Port Forward:</strong>
        <code id="sshCmd">ssh -L 8088:localhost:8088 user@host</code>
        <span style="color: var(--text-dim); margin-left: 0.5rem;">— и откройте <strong id="sshUrl">http://localhost:8088</strong> в локальном браузере.</span>
      </div>
    </div>

    <div class="grid">
      <!-- Блок 1: Распознавание пользовательской картинки -->
      <div class="card" style="grid-column: 1 / -1;">
        <div class="card-title">🔍 Распознавание изображения (Загрузка файла или Случайный рисунок)</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem;">
          <div>
            <div class="dropzone" id="dropArea" onclick="document.getElementById('fileInput').click()">
              <input type="file" id="fileInput" accept="image/*" style="display: none;">
              <div style="font-size: 2rem;">📁</div>
              <strong style="color: var(--primary);">Нажмите или перетащите сюда картинку</strong>
              <p>Поддерживаются PNG, JPG, WebP любого размера (авто-масштабирование до 128×128 RGB). Также работает вставка из буфера (Ctrl+V)!</p>
            </div>
            <div class="btn-group" style="margin-top: 0.75rem;">
              <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">📂 Выбрать файл</button>
              <button class="btn" id="btnSample" onclick="fetchRandomSample()">🎲 Случайный образец</button>
            </div>
          </div>

          <div class="prediction-box">
            <img id="previewImg" class="preview-img" alt="Предпросмотр">
            <div class="pred-result">
              <div id="predBadge" class="pred-badge" style="background: #334155;">Ожидание изображения...</div>
              <div id="trueBadge" style="font-size: 0.85rem; color: var(--text-dim); margin-bottom: 0.5rem;"></div>
              <div class="bar-group">
                <div class="bar-item">
                  <div class="bar-label"><span>Круг (Circle)</span><span id="prob0">0.0%</span></div>
                  <div class="bar-bg"><div id="fill0" class="bar-fill fill-circle"></div></div>
                </div>
                <div class="bar-item">
                  <div class="bar-label"><span>Квадрат (Square)</span><span id="prob1">0.0%</span></div>
                  <div class="bar-bg"><div id="fill1" class="bar-fill fill-square"></div></div>
                </div>
                <div class="bar-item">
                  <div class="bar-label"><span>Треугольник (Triangle)</span><span id="prob2">0.0%</span></div>
                  <div class="bar-bg"><div id="fill2" class="bar-fill fill-triangle"></div></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Блок 2: Генерация выборки -->
      <div class="card">
        <div class="card-title">📦 Генерация датасета (128×128)</div>
        <div>
          <label>Количество примеров по классам (Круг / Квадрат / Треугольник):</label>
          <div class="form-row">
            <input type="number" id="cnt0" value="1000" min="50" max="5000" step="50" title="Круг">
            <input type="number" id="cnt1" value="1000" min="50" max="5000" step="50" title="Квадрат">
            <input type="number" id="cnt2" value="1000" min="50" max="5000" step="50" title="Треугольник">
          </div>
        </div>
        <div>
          <label>Быстрые пресеты:</label>
          <div class="btn-group">
            <button class="btn btn-secondary" onclick="setCounts(1000, 1000, 1000)">1000 (Быстрый)</button>
            <button class="btn btn-secondary" onclick="setCounts(3000, 3000, 3000)">3000 (Стандарт)</button>
            <button class="btn btn-secondary" onclick="setCounts(1000, 1000, 100)">Дисбаланс (1000/100)</button>
          </div>
        </div>
        <label class="checkbox-label">
          <input type="checkbox" id="chkAugment">
          <span>Искажения (шум Гаусса, поворот 0-360°, битые пиксели, цвета)</span>
        </label>
        <button class="btn" id="btnGen" onclick="generateDataset()">⚡ Сгенерировать датасет</button>
        <div id="genStatus" style="font-size: 0.85rem; color: var(--text-dim);"></div>
      </div>

      <!-- Блок 3: Обучение модели -->
      <div class="card">
        <div class="card-title">🧠 Обучение перцептрона (MLP)</div>
        <p style="font-size: 0.85rem; color: var(--text-dim);">
          Архитектура: 49 152 входов (128×128×3) &rarr; 256 скрытых нейронов &rarr; 3 класса.
        </p>
        <div class="form-row">
          <div style="flex: 1;">
            <label>Эпох обучения:</label>
            <input type="number" id="trainEpochs" value="5" min="1" max="50">
          </div>
          <div style="flex: 1;">
            <label>Размер батча:</label>
            <select id="trainBatch">
              <option value="1">1 (Без батча / SGD)</option>
              <option value="16">16</option>
              <option value="32" selected>32 (Оптимально)</option>
              <option value="64">64</option>
            </select>
          </div>
        </div>
        <button class="btn" id="btnTrain" onclick="trainModel()">🚀 Запустить обучение</button>
        <div id="trainStatus" style="font-size: 0.85rem; color: var(--text-dim);"></div>
      </div>

      <!-- Блок 4: Оценка качества и Матрица ошибок -->
      <div class="card">
        <div class="card-title">📊 Оценка качества и Матрица ошибок</div>
        <button class="btn" id="btnEval" onclick="evaluateModel()">📈 Рассчитать метрики на тесте</button>
        <div id="evalSummary" style="font-size: 0.9rem; font-weight: 600; color: var(--primary);"></div>
        <div class="table-container">
          <table id="tblMetrics" style="display: none;">
            <thead>
              <tr><th>Класс</th><th>Выборка</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
            </thead>
            <tbody id="metricsBody"></tbody>
          </table>
        </div>
        <div id="cmContainer" style="display: none;">
          <label style="font-weight: 600; margin-top: 0.5rem;">Матрица ошибок (строки: факт, столбцы: прогноз):</label>
          <table id="tblCM">
            <thead>
              <tr><th>Факт \\ Прогноз</th><th>Круг</th><th>Квадрат</th><th>Треуг.</th></tr>
            </thead>
            <tbody id="cmBody"></tbody>
          </table>
        </div>
      </div>

      <!-- Блок 5: Инструменты, Веса и Сравнение -->
      <div class="card">
        <div class="card-title">⚙️ Инструменты и Сравнение с PyTorch</div>
        <div class="btn-group">
          <button class="btn btn-secondary" onclick="compareTorch()">⚖️ Сравнить с PyTorch</button>
          <button class="btn btn-secondary" onclick="saveModel()">💾 Сохранить веса</button>
          <button class="btn btn-secondary" onclick="loadModel()">📂 Загрузить веса</button>
        </div>
        <div id="toolsOut" class="console-out">Готов к работе.</div>
      </div>
    </div>
  </div>

  <script>
    // Инициализация статуса
    async function updateStatus() {
      try {
        const curPort = window.location.port || '8088';
        const cmdEl = document.getElementById('sshCmd');
        const urlEl = document.getElementById('sshUrl');
        if (cmdEl) cmdEl.textContent = 'ssh -L ' + curPort + ':localhost:' + curPort + ' user@host';
        if (urlEl) urlEl.textContent = 'http://localhost:' + curPort;

        const res = await fetch('/api/status');
        const d = await res.json();
        document.getElementById('coresBadge').textContent = d.cores;
        document.getElementById('trainBadge').textContent = d.train_samples;
        document.getElementById('testBadge').textContent = d.test_samples;
      } catch (e) {
        console.error(e);
      }
    }
    updateStatus();

    function setCounts(c0, c1, c2) {
      document.getElementById('cnt0').value = c0;
      document.getElementById('cnt1').value = c1;
      document.getElementById('cnt2').value = c2;
    }

    // Отображение вероятностей и бейджей
    function renderPrediction(data) {
      if (data.status !== 'ok') {
        alert(data.message || 'Ошибка распознавания');
        return;
      }
      const img = document.getElementById('previewImg');
      if (data.preview) img.src = data.preview;

      const badge = document.getElementById('predBadge');
      badge.textContent = data.class_name;
      if (data.class_name === 'Круг') badge.style.background = 'var(--accent-circle)';
      else if (data.class_name === 'Квадрат') badge.style.background = 'var(--accent-square)';
      else badge.style.background = 'var(--accent-triangle)';
      badge.style.color = '#fff';

      const trueBadge = document.getElementById('trueBadge');
      if (data.true_name !== undefined) {
        const match = data.predicted_class === data.true_class;
        trueBadge.innerHTML = 'Истинный класс: <strong>' + data.true_name + '</strong> ' + 
          (match ? '<span style="color: var(--success);">[ВЕРНО ✓]</span>' : '<span style="color: var(--danger);">[ОШИБКА ✗]</span>');
      } else {
        trueBadge.innerHTML = 'Пользовательское изображение';
      }

      data.probabilities.forEach((p, idx) => {
        const pct = (p.prob * 100).toFixed(1);
        document.getElementById('prob' + idx).textContent = pct + '%';
        document.getElementById('fill' + idx).style.width = pct + '%';
      });
    }

    // Случайный образец
    async function fetchRandomSample() {
      const btn = document.getElementById('btnSample');
      btn.disabled = true;
      try {
        const res = await fetch('/api/sample');
        const data = await res.json();
        renderPrediction(data);
      } catch (err) {
        alert('Ошибка: ' + err);
      } finally {
        btn.disabled = false;
      }
    }

    // Загрузка пользовательского файла
    async function uploadImageFile(file) {
      if (!file) return;
      const badge = document.getElementById('predBadge');
      badge.textContent = 'Обработка...';
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': file.type || 'application/octet-stream' },
          body: file
        });
        const data = await res.json();
        renderPrediction(data);
      } catch (err) {
        alert('Ошибка загрузки: ' + err);
      }
    }

    document.getElementById('fileInput').addEventListener('change', (e) => {
      if (e.target.files.length > 0) uploadImageFile(e.target.files[0]);
    });

    // Drag and Drop
    const dropArea = document.getElementById('dropArea');
    ['dragenter', 'dragover'].forEach(name => {
      dropArea.addEventListener(name, (e) => { e.preventDefault(); dropArea.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropArea.addEventListener(name, (e) => { e.preventDefault(); dropArea.classList.remove('dragover'); });
    });
    dropArea.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length > 0) {
        uploadImageFile(e.dataTransfer.files[0]);
      }
    });

    // Вставка из буфера обмена (Ctrl+V)
    window.addEventListener('paste', (e) => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
          uploadImageFile(item.getAsFile());
          break;
        }
      }
    });

    // Генерация датасета
    async function generateDataset() {
      const btn = document.getElementById('btnGen');
      const st = document.getElementById('genStatus');
      btn.disabled = true;
      st.innerHTML = '<span class="spinner"></span> Генерация изображений в multiprocessing пуле...';
      try {
        const counts = [
          parseInt(document.getElementById('cnt0').value) || 1000,
          parseInt(document.getElementById('cnt1').value) || 1000,
          parseInt(document.getElementById('cnt2').value) || 1000,
        ];
        const augment = document.getElementById('chkAugment').checked;
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ counts, augment })
        });
        const d = await res.json();
        if (d.status === 'ok') {
          st.innerHTML = '✓ Сгенерировано: <strong>' + d.train_count + '</strong> обуч., <strong>' + d.test_count + '</strong> тест. за ' + d.time.toFixed(2) + 'с';
          updateStatus();
        } else {
          st.textContent = 'Ошибка: ' + d.message;
        }
      } catch (err) {
        st.textContent = 'Сетевая ошибка: ' + err;
      } finally {
        btn.disabled = false;
      }
    }

    // Обучение перцептрона
    async function trainModel() {
      const btn = document.getElementById('btnTrain');
      const st = document.getElementById('trainStatus');
      btn.disabled = true;
      st.innerHTML = '<span class="spinner"></span> Обучение перцептрона (Backpropagation)...';
      try {
        const epochs = parseInt(document.getElementById('trainEpochs').value) || 5;
        const batch_size = parseInt(document.getElementById('trainBatch').value) || 32;
        const res = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ epochs, batch_size })
        });
        const d = await res.json();
        if (d.status === 'ok') {
          st.innerHTML = '✓ Обучено ' + d.epochs + ' эпох за <strong>' + d.time.toFixed(2) + 'с</strong> (' + d.speed.toFixed(0) + ' изобр/с)';
          logTool('Обучение завершено: ' + d.epochs + ' эпох за ' + d.time.toFixed(2) + 'с (' + d.speed.toFixed(0) + ' изобр/сек)');
        } else {
          st.textContent = 'Ошибка: ' + d.message;
        }
      } catch (err) {
        st.textContent = 'Сетевая ошибка: ' + err;
      } finally {
        btn.disabled = false;
      }
    }

    // Оценка на тесте
    async function evaluateModel() {
      const btn = document.getElementById('btnEval');
      const sum = document.getElementById('evalSummary');
      btn.disabled = true;
      sum.innerHTML = '<span class="spinner"></span> Расчет метрик...';
      try {
        const res = await fetch('/api/evaluate', { method: 'POST' });
        const d = await res.json();
        if (d.status === 'ok') {
          sum.innerHTML = 'Точность (Accuracy): ' + (d.overall_accuracy * 100).toFixed(2) + '% | Macro F1: ' + d.macro_f1.toFixed(4);
          
          const tbody = document.getElementById('metricsBody');
          tbody.innerHTML = '';
          d.per_class.forEach(c => {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td><strong>' + c.name + '</strong></td><td>' + c.samples + '</td><td>' + (c.precision * 100).toFixed(1) + '%</td><td>' + (c.recall * 100).toFixed(1) + '%</td><td><strong>' + (c.f1 * 100).toFixed(1) + '%</strong></td>';
            tbody.appendChild(tr);
          });
          document.getElementById('tblMetrics').style.display = 'table';

          const cmBody = document.getElementById('cmBody');
          cmBody.innerHTML = '';
          const names = ['Круг', 'Квадрат', 'Треугольник'];
          d.confusion_matrix.forEach((row, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td><strong>' + names[i] + '</strong></td>' + row.map(v => '<td>' + v + '</td>').join('');
            cmBody.appendChild(tr);
          });
          document.getElementById('cmContainer').style.display = 'block';
        } else {
          sum.textContent = 'Ошибка: ' + d.message;
        }
      } catch (err) {
        sum.textContent = 'Ошибка: ' + err;
      } finally {
        btn.disabled = false;
      }
    }

    function logTool(msg) {
      const box = document.getElementById('toolsOut');
      box.textContent = msg + '\\n' + box.textContent;
    }

    // Сравнение с PyTorch
    async function compareTorch() {
      logTool('Запуск сравнения с PyTorch (comp.py)...');
      try {
        const res = await fetch('/api/compare', { method: 'POST' });
        const d = await res.json();
        if (d.status === 'ok') {
          logTool('Сравнение:\\n  Самодельный: ' + d.time_my.toFixed(3) + 'с (acc: ' + (d.acc_my*100).toFixed(1) + '%)\\n  PyTorch:     ' + d.time_torch.toFixed(3) + 'с (acc: ' + (d.acc_torch*100).toFixed(1) + '%)\\n  Ускорение:   ' + d.speedup.toFixed(2) + 'x');
        } else {
          logTool('Ошибка: ' + d.message);
        }
      } catch (err) {
        logTool('Сетевая ошибка: ' + err);
      }
    }

    // Сохранение и загрузка
    async function saveModel() {
      try {
        const res = await fetch('/api/save_model', { method: 'POST' });
        const d = await res.json();
        logTool('Веса модели сохранены в ' + d.path);
      } catch (err) {
        logTool('Ошибка сохранения: ' + err);
      }
    }

    async function loadModel() {
      try {
        const res = await fetch('/api/load_model', { method: 'POST' });
        const d = await res.json();
        if (d.status === 'ok') {
          logTool('Веса модели успешно загружены из model.json');
        } else {
          logTool('Ошибка загрузки: ' + d.message);
        }
      } catch (err) {
        logTool('Ошибка загрузки: ' + err);
      }
    }

    // Загрузка начального случайного примера при открытии
    setTimeout(fetchRandomSample, 300);
  </script>
</body>
</html>
"""


class WebHandler(BaseHTTPRequestHandler):
    """HTTP обработчик запросов для браузерного интерфейса."""

    def log_message(self, format: str, *args: Any) -> None:
        """Минимальный вывод в консоль для чистоты логов."""

    def _send_json(
        self, data: dict[str, Any], status: int = HTTPStatus.OK
    ) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            raw_html = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw_html)))
            self.end_headers()
            self.wfile.write(raw_html)
            return

        if path == "/api/status":
            with STATE.lock:
                self._send_json(
                    {
                        "status": "ok",
                        "cores": STATE.cores,
                        "counts": STATE.counts,
                        "augment": STATE.augment,
                        "train_samples": len(STATE.y_train)
                        if STATE.y_train is not None
                        else 0,
                        "test_samples": len(STATE.y_test)
                        if STATE.y_test is not None
                        else 0,
                    }
                )
            return

        if path == "/api/sample":
            with STATE.lock:
                if STATE.x_test is not None and len(STATE.x_test) > 0:
                    idx = random.randint(0, len(STATE.x_test) - 1)
                    sample = STATE.x_test[idx]
                    true_cls = int(STATE.y_test[idx].item())
                else:
                    # Если датасет еще не создан, генерируем одну фигуру на лету
                    true_cls = random.randint(0, len(dataset.CLASSES) - 1)
                    raw_arr = dataset.generate_single_image(
                        cls=true_cls, augment=STATE.augment, size=IMAGE_SIZE
                    )
                    sample = torch.from_numpy(raw_arr)

                pred_cls, probs = my.predict_single(STATE.net, sample)
                preview = tensor_to_base64_png(sample)

            self._send_json(
                {
                    "status": "ok",
                    "predicted_class": pred_cls,
                    "class_name": dataset.CLASSES[pred_cls],
                    "true_class": true_cls,
                    "true_name": dataset.CLASSES[true_cls],
                    "probabilities": [
                        {
                            "name": dataset.CLASSES[i],
                            "prob": round(float(probs[i]), 4),
                        }
                        for i in range(len(dataset.CLASSES))
                    ],
                    "preview": preview,
                }
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/predict":
            # Пользовательская картинка (любые бинарные байты PNG/JPG/etc)
            body = self._read_body()
            if not body:
                self._send_json(
                    {"status": "error", "message": "Пустое тело запроса"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                img = (
                    Image.open(io.BytesIO(body))
                    .convert("RGB")
                    .resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                )
                arr = (
                    np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                )
                tensor = torch.from_numpy(arr)

                with STATE.lock:
                    pred_cls, probs = my.predict_single(STATE.net, tensor)

                self._send_json(
                    {
                        "status": "ok",
                        "predicted_class": pred_cls,
                        "class_name": dataset.CLASSES[pred_cls],
                        "probabilities": [
                            {
                                "name": dataset.CLASSES[i],
                                "prob": round(float(probs[i]), 4),
                            }
                            for i in range(len(dataset.CLASSES))
                        ],
                        "preview": pil_to_base64_png(img),
                    }
                )
            except (OSError, ValueError, TypeError) as e:
                self._send_json(
                    {"status": "error", "message": f"Не удалось прочитать изображение: {e}"},
                    HTTPStatus.BAD_REQUEST,
                )
            return

        if path == "/api/generate":
            body = self._read_body()
            payload = json.loads(body.decode("utf-8")) if body else {}
            counts = payload.get("counts", [1000, 1000, 1000])
            augment = bool(payload.get("augment", False))

            counts = [max(10, min(10000, int(c))) for c in counts]
            test_per = min(600, min(counts))

            t0 = time.perf_counter()
            with STATE.lock:
                STATE.counts = counts
                STATE.augment = augment
                x_tr, y_tr, x_te, y_te = dataset.create_dataset(
                    counts,
                    augment=augment,
                    test_per_class=test_per,
                    n_jobs=STATE.cores,
                    size=IMAGE_SIZE,
                )
                STATE.x_train = x_tr
                STATE.y_train = y_tr
                STATE.x_test = x_te
                STATE.y_test = y_te
            dt = time.perf_counter() - t0

            self._send_json(
                {
                    "status": "ok",
                    "train_count": len(y_tr),
                    "test_count": len(y_te),
                    "time": round(dt, 3),
                }
            )
            return

        if path == "/api/train":
            body = self._read_body()
            payload = json.loads(body.decode("utf-8")) if body else {}
            epochs = max(1, min(50, int(payload.get("epochs", 5))))
            batch_size = max(1, min(256, int(payload.get("batch_size", 32))))

            with STATE.lock:
                if STATE.x_train is None:
                    self._send_json(
                        {
                            "status": "error",
                            "message": "Сначала сгенерируйте датасет!",
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                x_tr = STATE.x_train
                y_tr = STATE.y_train
                net = STATE.net

            t0 = time.perf_counter()
            my.train(
                net,
                x_tr,
                y_tr,
                epochs=epochs,
                batch_size=batch_size,
                verbose=False,
            )
            dt = time.perf_counter() - t0
            speed = (len(y_tr) * epochs) / max(dt, 1e-6)

            self._send_json(
                {
                    "status": "ok",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "time": round(dt, 3),
                    "speed": round(speed, 1),
                }
            )
            return

        if path == "/api/evaluate":
            with STATE.lock:
                if STATE.x_test is None:
                    self._send_json(
                        {
                            "status": "error",
                            "message": "Сначала сгенерируйте датасет!",
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                x_te = STATE.x_test
                y_te = STATE.y_test
                net = STATE.net

            stats = my.evaluate(net, x_te, y_te, verbose=False)
            res = {"status": "ok"}
            res.update(stats)
            self._send_json(res)
            return

        if path == "/api/compare":
            res = comp.compare_models(
                counts=[300, 300, 300], epochs=2, size=IMAGE_SIZE, verbose=False
            )
            self._send_json(
                {
                    "status": "ok",
                    "time_my": round(res["time_my"], 3),
                    "time_torch": round(res["time_torch"], 3),
                    "acc_my": round(res["acc_my"], 4),
                    "acc_torch": round(res["acc_torch"], 4),
                    "speedup": round(res["speedup"], 2),
                }
            )
            return

        if path == "/api/save_model":
            with STATE.lock:
                my.save_model(STATE.net, "model.json", verbose=False)
            self._send_json({"status": "ok", "path": "model.json"})
            return

        if path == "/api/load_model":
            try:
                with STATE.lock:
                    STATE.net = my.load_model("model.json", verbose=False)
                self._send_json({"status": "ok"})
            except (OSError, KeyError, RuntimeError, my.json.JSONDecodeError) as e:
                self._send_json(
                    {"status": "error", "message": str(e)},
                    HTTPStatus.BAD_REQUEST,
                )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def run_server(host: str = "0.0.0.0", port: int = 8088) -> None:
    """Запуск многопоточного веб-сервера с автоматическим подбором свободного порта."""
    target_port = port
    httpd = None
    for p in range(target_port, target_port + 50):
        try:
            httpd = ThreadingHTTPServer((host, p), WebHandler)
            port = p
            break
        except OSError:
            continue

    if httpd is None:
        httpd = ThreadingHTTPServer((host, 0), WebHandler)
        port = httpd.server_address[1]

    url_local = f"http://127.0.0.1:{port}"
    print("=" * 68)
    if port != target_port:
        print(
            f"⚠️  Порт {target_port} занят, автоматически выбран свободный порт {port}!"
        )
    print(f"🚀 Веб-интерфейс запущен на {url_local}")
    print("Для доступа через SSH port forward выполните на своем компьютере:")
    print(f"   ssh -L {port}:localhost:{port} <ваш_пользователь>@<ip_сервера>")
    print(f"Затем откройте в браузере: http://localhost:{port}")
    print("=" * 68)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка веб-сервера...")
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Веб-интерфейс для классификации фигур (MLP)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для привязки (по умолчанию 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8088,
        help="Порт для веб-интерфейса (по умолчанию 8088)",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
