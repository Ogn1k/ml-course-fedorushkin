# Модуль 2 — House Prices: предобработка и регрессия

**Федорушкин Станислав, ПКТб-23-1** · Дата обучения: 27.10.2026.

Лабораторная №2: EDA, смысловые пропуски, выбросы, log1p, создание признаков,
OLS/Ridge/Lasso/ElasticNet, пятифолдовая CV, подбор гиперпараметров и инференс.
Выполненный [notebook.ipynb](notebook.ipynb) содержит все 14 разделов задания.

## Результаты

| Модель | CV RMSE(log) | Holdout RMSE(log) | R²(log) | R²(USD) | MAE, USD | RMSE, USD |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.1377 | 0.1234 | 0.9184 | 0.9343 | 15,152 | 22,456 |
| Ridge | 0.1128 | 0.1304 | 0.9089 | 0.9368 | 14,764 | 22,021 |
| Lasso | 0.1120 | 0.1287 | 0.9112 | 0.9391 | 14,378 | 21,620 |
| ElasticNet | 0.1118 | 0.1292 | 0.9106 | 0.9390 | 14,426 | 21,630 |

Выбрана **ElasticNet по CV**, до оценки holdout. Различия CV между регуляризованными
моделями малы относительно разброса по фолдам. На holdout OLS имеет меньший RMSE(log),
но модель после просмотра holdout не перевыбиралась.
Обучение: 1166 домов; holdout: 292; random_state=42.
Удалены только из обучения ID [524, 1299] по правилу GrLivArea > 4000 и SalePrice < 300000.
Полные пайплайны и подбор параметров исключают fit импутации/кодирования/масштабирования на holdout.
Время финального fit ElasticNet: 0.249 с;
поиск по сетке: 17.35 с.

## Запуск в PowerShell

Проверено на Python 3.14.0; точные версии — в requirements.txt.

```powershell
cd F:\GitHub\ml-course-fedorushkin
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r module-2-house-prices/requirements.txt
cd module-2-house-prices
..\.venv\Scripts\python.exe -m jupyter notebook notebook.ipynb
```

В Jupyter выберите Kernel → Restart & Run All. Запускайте из каталога модуля.
Без Jupyter: `python appendix_full_code.py`. Проверка артефактов: `python validate.py`.

## Предсказание без обучения

```python
import joblib
from housing_pipeline import predict_price

model = joblib.load('models/best_model.pkl')
house = {'OverallQual': 6, 'GrLivArea': 1500,
         'YearBuilt': 2000, 'Neighborhood': 'CollgCr'}
print(predict_price(house, model))  # USD
```

Функция принимает словарь, список словарей или DataFrame исходных признаков.
Четыре указанных поля обязательны; прочие отсутствующие значения импутируются.
Не нужно вручную вызывать scaler: он уже внутри pipeline.
Для неполного описания прогноз условен, поскольку недостающие характеристики заменены типичными.

## Загрузка из GitHub

```python
from housing_pipeline import load_remote_model, predict_price
BASE_URL = 'https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-2-house-prices'
model = load_remote_model(BASE_URL)
print(predict_price(house, model))
```

Необходим файл `housing_pipeline.py` из этого каталога и версии из requirements.txt.
Загружайте joblib только из доверенного собственного источника: формат выполняет Python-код.
Локальная загрузка проверена. **GitHub raw пока не проверен: файлы ещё не опубликованы.**
После публикации для проверки трёх совпадающих прогнозов:

```powershell
$env:HOUSE_PRICES_VERIFY_REMOTE = '1'
python appendix_full_code.py
```

## Файлы

- `notebook.ipynb` — выполненный отчёт, 14 разделов.
- `housing_pipeline.py` — преобразования и функции предсказания/загрузки.
- `appendix_full_code.py` — полный вычислительный код с разделителями `# %%`.
- `validate.py` — независимая проверка артефактов, разбиения и инференса.
- `data/` — train/test, словарь признаков, описание, ссылки и SHA-256.
- `models/` — четыре полных пайплайна, best_model, scaler, признаки, метрики и метаданные.
- `examples/` — шесть EDA-панелей, остатки, прогнозы, коэффициенты, CSV метрик.
- `submission.csv` — 1459 прогнозов в исходном порядке ID, без отправки на Kaggle.
- `defense_notes.md` — краткая памятка для защиты.

Данные: [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data),
скачаны из [зеркала CSV](https://github.com/Rswcf/houseprices_kaggle).
Происхождение каждого файла — `data/provenance.json`.
Модель применима к похожим продажам Эймса 2006–2010; современный рынок не проверялся.
