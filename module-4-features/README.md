# Модуль 4 — Инженерия признаков: Online Shoppers

**Федорушкин Станислав, ПКТб-23-1** · 29.10.2026 · 2026/2027.

[Выполненный notebook.ipynb](notebook.ipynb) — отчёт из 14 разделов.
Официальные данные [UCI](https://doi.org/10.24432/C5F88Q), 11 производных признаков,
4 ансамбля, 5-фолдовая CV, 30 испытаний Optuna, абляция признаков и SHAP.
Итоговая модель: **Random Forest (Optuna)**, выбрана по train CV ROC-AUC.

## Результаты на отложенном test

| Model                  |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |   PR_AUC |   CV_ROC_AUC |   Fit_seconds |
|:-----------------------|-----------:|------------:|---------:|-------:|----------:|---------:|-------------:|--------------:|
| Random Forest          |     0.8783 |      0.5813 |   0.7958 | 0.6718 |    0.9312 |   0.7164 |       0.9308 |        1.0212 |
| XGBoost                |     0.8824 |      0.5996 |   0.7487 | 0.6659 |    0.9292 |   0.7444 |       0.9265 |        0.7654 |
| LightGBM               |     0.8697 |      0.5584 |   0.8010 | 0.6581 |    0.9310 |   0.7432 |       0.9277 |        0.4055 |
| CatBoost               |     0.8751 |      0.5714 |   0.8063 | 0.6688 |    0.9329 |   0.7490 |       0.9296 |        1.5397 |
| Random Forest (Optuna) |     0.8751 |      0.5696 |   0.8246 | 0.6738 |    0.9350 |   0.7468 |       0.9346 |        2.7653 |

Время Fit_seconds не включает CV и Optuna. PR_AUC — average precision.
Порог 0.5. Test не участвует в подборе; CV-оценка после выбора может быть оптимистична.

## Влияние признаков

| Features           |   CV_ROC_AUC |   CV_std |
|:-------------------|-------------:|---------:|
| Original           |      0.92723 |  0.00661 |
| Engineered         |      0.92771 |  0.00503 |
| Without PageValues |      0.77559 |  0.01066 |

Инженерия не гарантирует прироста. Данные агрегированы по сессии;
проверка без PageValues не подтверждает пригодность для раннего онлайн-прогноза.
SHAP объясняет модель, не причинность; числовые признаки на графиках стандартизированы.

## Воспроизведение (PowerShell)

```powershell
cd F:\GitHub\ml-course-fedorushkin\module-4-features
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_notebook.py
.\.venv\Scripts\python.exe validate.py
```

Версия использованного Python: 3.14.0. Запуск целиком повторяет обучение и Optuna.
Готовые выходы и графики уже сохранены в ноутбуке; интернет нужен только для первой загрузки CSV.

## Прогноз без повторного обучения

```python
import joblib, pandas as pd
model = joblib.load('models/best_model.pkl')
sessions = pd.read_csv('examples/inference_input.csv')
print(model.predict_proba(sessions)[:, 1])
```

Запускать из папки модуля, где доступен feature_engineering.py.
Pipeline принимает 17 исходных полей. scaler.pkl отдельно применять не нужно.

## Файлы

- data/: исходный CSV, описание, контрольная сумма и сведения о разбиении.
- models/: 4 baseline, tuned_model.pkl, best_model.pkl, scaler и JSON-метаданные.
- examples/: EDA, ROC, важности, SHAP, Optuna и примеры прогноза.
- appendix_full_code.py: полный код вычислительных ячеек.
- defense_notes.md: краткий план защиты.
- validate.py: проверка артефактов и локальной загрузки.
- verify_remote.py: проверка загрузки через raw GitHub после публикации.