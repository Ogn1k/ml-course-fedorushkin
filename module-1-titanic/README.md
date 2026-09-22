# 🚢 Module 1 — Titanic: EDA + бинарная классификация

**Автор:** Федорушкин Станислав, ПКТб-23-1  
**Дата:** 22.09.2026  
**Датасет:** [Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## О проекте

Проект покрывает полный учебный ML-цикл: EDA, обработку пропусков, feature engineering, кодирование и масштабирование, обучение трёх моделей, оценку метрик, интерпретацию и сохранение артефактов.

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.783 | 0.681 | 0.729 | **0.849** |
| Random Forest | 0.793 | 0.722 | 0.754 | **0.738** | 0.848 |
| Decision Tree | 0.765 | 0.708 | 0.667 | 0.687 | 0.803 |

## Быстрый старт

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r module-1-titanic/requirements.txt
jupyter notebook module-1-titanic/notebook.ipynb
```

Ноутбук следует запускать из папки `module-1-titanic`, чтобы относительные пути `data/`, `models/` и `examples/` указывали на файлы модуля.

## Структура

```text
module-1-titanic/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── appendix_full_code.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── titanic_info.md
├── models/
│   ├── lr_model.pkl
│   ├── dt_model.pkl
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── le_sex.pkl
│   ├── feature_cols.json
│   ├── metrics.json
│   └── metadata.json
└── examples/
    ├── eda_plots.png
    ├── survival_breakdown.png
    ├── confusion_matrices.png
    └── roc_curves.png
```

## Загрузка модели

```python
from io import BytesIO

import joblib
import requests

base_url = "https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-1-titanic"
response = requests.get(f"{base_url}/models/lr_model.pkl", timeout=30)
response.raise_for_status()
model = joblib.load(BytesIO(response.content))
```
