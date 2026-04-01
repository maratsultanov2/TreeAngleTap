# 🌳 TreeAngleTap (TAT-7)

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSES/LICENSE_CODE.txt)
[![License: CC BY-NC-ND](https://img.shields.io/badge/Weights-CC%20BY--NC--ND-lightgrey.svg)](LICENSES/LICENSE_WEIGHTS.txt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZAMW7NaNk4NqOQxPjSRJr_5i9OI6UpZR)

**83% на CIFAR-10 | 0.45M параметров | 3–18 мс на телефоне | 5 пирамид**

---

## 🧛 The Giving Vampire

**🇷🇺** TreeAngleTap не заменяет ваш ИИ. Он делает его умнее.

**🇬🇧** TreeAngleTap doesn't replace your AI. It makes it smarter.

**🇷🇺** Он впитывает знания из DeepSeek, ChatGPT, Claude, GigaChat — и отдаёт их, когда нужно.

**🇬🇧** It absorbs knowledge from DeepSeek, ChatGPT, Claude, GigaChat — and gives it back when you need it.

**🇷🇺** Одна память. Любой ИИ.

**🇬🇧** One memory. Any AI.

**🇷🇺** Вампир, который отдаёт.

**🇬🇧** A vampire that gives.

---

## 📌 О проекте / About

**🇷🇺** **TreeAngleTap (TAT-7)** — экспериментальная архитектура, которая снижает забывание при последовательном обучении через 5 пирамид и гармоническое взаимодействие.

**🇬🇧** **TreeAngleTap (TAT-7)** is an experimental architecture that reduces forgetting in continual learning through 5 pyramids and harmonic interaction.

**🇷🇺** **Ключевое открытие:** HARMONY (связи между признаками) оказалась в 4 раза важнее, чем SOLO (сами признаки).

**🇬🇧** **Key discovery:** HARMONY (connections between features) is 4x more important than SOLO (the features themselves).

---

## ✨ Особенности / Features

| 🇷🇺 Особенность | 🇬🇧 Feature | 🇷🇺 Описание | 🇬🇧 Description |
|----------------|------------|--------------|----------------|
| 📱 Мобильная оптимизация | Mobile optimization | 3–18 мс на Infinix Smart 6 HD (2022) | 3–18 ms on Infinix Smart 6 HD (2022) |
| 🧠 Не забывает | Doesn't forget | После 5 задач на CIFAR-10 помнит 71% | After 5 tasks on CIFAR-10 remembers 71% |
| 🏗️ Понятная архитектура | Clear architecture | 5 пирамид — объясняется за 5 минут | 5 pyramids — explained in 5 minutes |
| 🔓 Открытый код | Open source | MIT для кода, веса CC BY-NC-ND | MIT for code, CC BY-NC-ND for weights |

---

## 🧱 Архитектура: 5 пирамид / Architecture: 5 Pyramids

| Пирамида / Pyramid | 🇷🇺 Функция | 🇬🇧 Function | Параметры / Parameters |
|--------------------|------------|-------------|------------------------|
| **Фильтр / Filter** | Разделяет признаки | Splits features | SOLO=0.55, RHYTHM=0.43, HARMONY=2.0 |
| **Компрессор / Compressor** | Сжимает данные | Compresses data | 1 слой, 128 нейронов / 1 layer, 128 units |
| **Головы / Heads** | Параллельная обработка | Parallel processing | 5 голов, 64 размерность / 5 heads, 64 dim |
| **Память / Memory** | Сохраняет контекст | Stores context | T=0.7, энтропия=1.18 / T=0.7, entropy=1.18 |
| **Вывод / Output** | Классифицирует | Classifies | 1x128_silu, dropout=0.3 |

---

## 📊 Результаты / Results

### CIFAR-10 (5 задач, 15 эпох / 5 tasks, 15 epochs)

| 🇷🇺 Метрика | 🇬🇧 Metric | 🇷🇺 Значение | 🇬🇧 Value |
|------------|-----------|-------------|----------|
| Точность | Accuracy | **83.43%** | **83.43%** |
| Память о первой задаче | Memory of first task | **71.54%** | **71.54%** |

### MNIST (5 задач, 15 эпох / 5 tasks, 15 epochs)

| 🇷🇺 Метрика | 🇬🇧 Metric | 🇷🇺 Значение | 🇬🇧 Value |
|------------|-----------|-------------|----------|
| Точность | Accuracy | **99.77%** | **99.77%** |

### Fashion MNIST (5 задач, 15 эпох / 5 tasks, 15 epochs)

| 🇷🇺 Метрика | 🇬🇧 Metric | 🇷🇺 Значение | 🇬🇧 Value |
|------------|-----------|-------------|----------|
| Точность | Accuracy | **99.24%** | **99.24%** |

---

## 🚀 Быстрый старт / Quick Start

### 🇷🇺 Установка
```bash
pip install git+https://github.com/maratsultanov2/TreeAngleTap.git
```

### 🇬🇧 Installation
```bash
pip install git+https://github.com/maratsultanov2/TreeAngleTap.git
```

### 🇷🇺 Использование
```python
from treeangletap import TAT7

model = TAT7()
model.load_weights("weights/tat7_mobile.pt")

# Предсказание
result = model.predict(image)
```

### 🇬🇧 Usage
```python
from treeangletap import TAT7

model = TAT7()
model.load_weights("weights/tat7_mobile.pt")

# Predict
result = model.predict(image)
```

### 🇷🇺 Запустить в Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZAMW7NaNk4NqOQxPjSRJr_5i9OI6UpZR)

### 🇬🇧 Run in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZAMW7NaNk4NqOQxPjSRJr_5i9OI6UpZR)

---

## 💼 Где это нужно / Use Cases

| 🇷🇺 Область | 🇬🇧 Field | 🇷🇺 Задача | 🇬🇧 Task |
|------------|----------|------------|---------|
| IT-мониторинг | IT monitoring | Детекция сбоев в логах | Detecting failures in logs |
| Медицина | Medicine | Диагностика редких заболеваний | Diagnosing rare diseases |
| Банки | Banks | Антифрод | Fraud detection |
| Робототехника | Robotics | Обучение новым навыкам | Learning new skills |

---

## ☕ Поддержать проект / Support the Project

| 🇷🇺 Способ | 🇬🇧 Method | 🇷🇺 Реквизиты | 🇬🇧 Details |
|-----------|-----------|--------------|------------|
| ЮMoney | YooMoney | `4100119011323328` | `4100119011323328` |
| СБП | SBP | `+7-951-902-44-58` (Марат С.) | `+7-951-902-44-58` (Marat S.) |
| Коммерческая лицензия | Commercial license | maratsultanov2@gmail.com | maratsultanov2@gmail.com |

[![Поддержать](https://img.shields.io/badge/ЮMoney-Поддержать-00AAFF.svg)](https://yoomoney.ru/to/4100119011323328)

---

## 📜 Лицензии / Licenses

| 🇷🇺 Компонент | 🇬🇧 Component | 🇷🇺 Лицензия | 🇬🇧 License |
|--------------|--------------|-------------|------------|
| Исходный код | Code | MIT | MIT |
| Веса моделей | Model weights | CC BY-NC-ND 4.0 | CC BY-NC-ND 4.0 |

---

## 🔗 Ссылки / Links

- **GitHub**: [github.com/maratsultanov2/TreeAngleTap](https://github.com/maratsultanov2/TreeAngleTap)
- **Colab**: [открыть ноутбук / open notebook](https://colab.research.google.com/drive/1ZAMW7NaNk4NqOQxPjSRJr_5i9OI6UpZR)
- **Telegram**: [@Marat_Sultanow](https://t.me/Marat_Sultanow)

---

## 👤 Автор / Author

**Marat Sultanow**, 2026

*🇷🇺 Если вы используете TAT-7 в своих проектах, пожалуйста, укажите авторство.*
*🇬🇧 If you use TAT-7 in your projects, please credit the author.*