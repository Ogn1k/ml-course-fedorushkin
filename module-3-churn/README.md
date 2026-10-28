# Модуль 3 — Ансамблевые методы: Telco Customer Churn

**Федорушкин Станислав, ПКТб-23-1** · Выполнено 27.10.2026.

Лабораторная №3: четыре ансамбля, стратифицированное разбиение 80/20,
пятифолдовая CV, 30 испытаний Optuna, SHAP и сохранение полных pipeline.
Выполненный [notebook.ipynb](notebook.ipynb) содержит 14 разделов отчёта.

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC | Fit, с |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7736 | 0.5591 | 0.6952 | 0.6198 | 0.8389 | 0.8430 | 0.7346 |
| XGBoost | 0.7715 | 0.5546 | 0.7059 | 0.6212 | 0.8318 | 0.8340 | 0.3553 |
| LightGBM | 0.7573 | 0.5305 | 0.7433 | 0.6192 | 0.8366 | 0.8405 | 0.2543 |
| CatBoost | 0.7544 | 0.5263 | 0.7487 | 0.6181 | 0.8378 | 0.8378 | 0.9457 |
| LightGBM (Optuna) | 0.7544 | 0.5245 | 0.8021 | 0.6342 | 0.8479 | 0.8491 | 0.3134 |

Итоговый артефакт: **LightGBM (Optuna)**, выбран по CV между baseline и tuned LightGBM.
Семейство LightGBM задано разделом Optuna методички заранее; лучшая baseline
среди четырёх моделей по CV — **Random Forest**.
Holdout: 1409 клиентов; train: 5634; seed=42. Время fit включает предобработку,
4 CPU-потока; поиск Optuna занял 43.5 с.
Все результаты рассчитаны; прирост качества не предполагается заранее.

Предобработка обучается внутри каждого CV-фолда: медиана TotalCharges,
LabelEncoder и четыре производных признака. Число входных признаков модели — 23.
Категории CatBoost закодированы так же, как у остальных моделей, по заданию.
Scaler сохраняется отдельно для комплекта артефактов, деревьям он не нужен.
Взвешенные score не проверены на калибровку; порог 0.5 фиксирован заранее.

## Запуск

Проверено на Python 3.14.0. Из корня репозитория:

```powershell
python -m venv .venv-module3
.\.venv-module3\Scripts\python.exe -m pip install -r module-3-churn/requirements.txt notebook==7.5.0
.\.venv-module3\Scripts\python.exe -m ipykernel install --prefix .venv-module3 --name module3 --display-name "Python (Module 3)"
$env:JUPYTER_PATH = "$PWD\.venv-module3\share\jupyter"
cd module-3-churn
..\.venv-module3\Scripts\python.exe -m notebook notebook.ipynb
```

В Notebook выберите kernel **Python (Module 3)** и Restart & Run All.
Без интерфейса: `..\.venv-module3\Scripts\python.exe run_notebook.py`.
Без Jupyter: `..\.venv-module3\Scripts\python.exe appendix_full_code.py`.
Проверка сохранённых результатов: `..\.venv-module3\Scripts\python.exe validate.py`.
Полное обучение перезаписывает модели/графики; исходный CSV не меняется.

## Предсказание

```python
import pandas as pd
import joblib
from churn_pipeline import predict_churn

model = joblib.load('models/best_model.pkl')
clients = pd.read_csv('data/telco_churn.csv').iloc[:3]
print(predict_churn(clients, model))
```

Вход — исходные 19 признаков, DataFrame/словарь/список словарей.
customerID и Churn допускаются, но исключаются. Пропущенное обязательное поле
вызывает ValueError; неизвестная категория кодируется -1. Модель сама выполняет
импутацию и кодирование. Нужен доступный `churn_pipeline.py` и версии из requirements.
joblib загружайте только из доверенного источника.

## GitHub raw и сдача

**Локальная сериализация и GitHub raw проверены: классы и score трёх клиентов совпали.**
Модель уже доступна в main; для публикации оставшихся изменений отчёта и документации
после их проверки используйте из корня репозитория:

```powershell
git add README.md .gitignore module-3-churn
git commit -m "Module 3: train churn ensembles, tune LightGBM and explain with SHAP"
git push origin main
```

Раздел 12 содержит загрузку с
`https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-3-churn`.
Повторить проверку без переобучения:

```powershell
cd module-3-churn
..\.venv-module3\Scripts\python.exe verify_remote.py
```

Для сохранения HTTP-проверки в выводах самого отчёта установите
`$env:CHURN_VERIFY_REMOTE = '1'` и выполните раздел 12 в уже запущенном notebook.
Предполагается ветка main; HTTP-ошибки и несовпадение прогнозов не скрываются.
Преподавателю нужны ссылка на репозиторий, ссылка на notebook и
[таблица метрик PNG](examples/metrics.png).

## Файлы

- `notebook.ipynb` — выполненный отчёт; `appendix_full_code.py` — весь вычислительный код.
- `churn_pipeline.py` — предобработка и инференс; `validate.py` — независимая проверка.
- `data/` — исходный CSV, описание всех полей, provenance и SHA-256.
- `models/` — четыре baseline, tuned LightGBM, best_model, scaler, encoders, JSON.
- `examples/` — EDA, ROC, матрицы ошибок, SHAP, метрики и история Optuna.
- `defense_notes.md` — памятка для защиты, `run_notebook.py` — запуск всего notebook.

Данные: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
скачаны из [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d).
Бизнес-интерпретации являются гипотезами; причинный эффект удержания не измерен.