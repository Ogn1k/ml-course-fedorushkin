# %%
# Базовые библиотеки и воспроизводимые настройки.
from pathlib import Path
from shutil import copy2
import json
import platform
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
RANDOM_STATE = 42

# В репозитории данные должны лежать в data/. Если исходные CSV пока рядом
# с ноутбуком, один раз копируем их в требуемую структуру.
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)
for filename in ("train.csv", "test.csv"):
    source = Path(filename)
    destination = DATA_DIR / filename
    if not destination.exists() and source.exists():
        copy2(source, destination)

# Загружаем обе части и сразу проверяем формы.
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
print(train_raw.shape, test_raw.shape)
display(train_raw.head())

# Создаём требуемое описание данных в папке data/.
class_counts = train_raw["Survived"].value_counts().sort_index()
info_text = f'''# Titanic dataset

Источник: https://www.kaggle.com/c/titanic

- train: {train_raw.shape[0]} строк, {train_raw.shape[1]} столбцов
- test: {test_raw.shape[0]} строк, {test_raw.shape[1]} столбцов
- класс 0: {class_counts[0]} ({class_counts[0] / len(train_raw):.1%})
- класс 1: {class_counts[1]} ({class_counts[1] / len(train_raw):.1%})

Пропуски train:
{train_raw.isna().sum().to_string()}
'''
(DATA_DIR / "titanic_info.md").write_text(info_text, encoding="utf-8")
print("Создан файл:", DATA_DIR / "titanic_info.md")

# %%
# Первичный осмотр структуры, типов и описательных статистик.
print("Размер train:", train_raw.shape)
display(train_raw.head())
train_raw.info()
display(train_raw.describe(include="all").T)

# Таблица пропусков в абсолютных значениях и процентах.
missing_table = pd.DataFrame({
    "missing": train_raw.isna().sum(),
    "percent": train_raw.isna().mean().mul(100).round(2),
}).query("missing > 0").sort_values("percent", ascending=False)
display(missing_table)

# Статистики формы распределения и частоты категорий.
numeric_cols = train_raw.select_dtypes(include=np.number).columns
shape_stats = pd.DataFrame({
    "skew": train_raw[numeric_cols].skew(),
    "kurtosis": train_raw[numeric_cols].kurtosis(),
})
display(shape_stats)
for column in ["Survived", "Pclass", "Sex", "Embarked"]:
    print(f"\n{column}:\n{train_raw[column].value_counts(dropna=False)}")

# %%
# Семь визуальных срезов: классы, возраст, boxplot, пропуски,
# корреляции и выживаемость по трём категориальным признакам.
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
sns.countplot(data=train_raw, x="Survived", ax=axes[0, 0])
axes[0, 0].set(title="Распределение целевого класса", xlabel="Выжил (0/1)", ylabel="Число пассажиров")

sns.histplot(data=train_raw, x="Age", hue="Survived", kde=True, bins=30, ax=axes[0, 1])
axes[0, 1].set(title="Возраст и выживание", xlabel="Возраст, лет", ylabel="Частота")

sns.boxplot(data=train_raw, x="Pclass", y="Age", hue="Survived", ax=axes[0, 2])
axes[0, 2].set(title="Возраст по классу и исходу", xlabel="Класс билета", ylabel="Возраст, лет")

missing_plot = train_raw.isna().mean().mul(100).sort_values(ascending=False)
sns.barplot(x=missing_plot.values, y=missing_plot.index, ax=axes[1, 0], color="steelblue")
axes[1, 0].set(title="Доля пропусков", xlabel="Пропуски, %", ylabel="Столбец")

corr = train_raw.select_dtypes(include=np.number).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1, 1])
axes[1, 1].set_title("Корреляции числовых признаков")

survival_by_sex = train_raw.groupby("Sex", as_index=False)["Survived"].mean()
sns.barplot(data=survival_by_sex, x="Sex", y="Survived", ax=axes[1, 2])
axes[1, 2].set(title="Выживаемость по полу", xlabel="Пол", ylabel="Доля выживших")
plt.tight_layout()
fig.savefig(EXAMPLES_DIR / "eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()

# Ещё два обязательных сравнения: класс билета и порт посадки.
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for axis, column, title in zip(axes, ["Pclass", "Embarked"], ["классу билета", "порту посадки"]):
    rates = train_raw.groupby(column, as_index=False)["Survived"].mean()
    sns.barplot(data=rates, x=column, y="Survived", ax=axis)
    axis.set(title=f"Выживаемость по {title}", xlabel=column, ylabel="Доля выживших")
plt.tight_layout()
fig.savefig(EXAMPLES_DIR / "survival_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
from sklearn.preprocessing import LabelEncoder

# Работаем с копией, сохраняя исходный DataFrame для проверки и графиков.
train_df = train_raw.copy()
test_df = test_raw.copy()

# Медианы возраста вычисляем только на train и применяем к обоим наборам.
age_medians = train_df.groupby(["Pclass", "Sex"])["Age"].median()
def fill_age(frame):
    frame = frame.copy()
    missing = frame["Age"].isna()
    frame.loc[missing, "Age"] = frame.loc[missing].apply(
        lambda row: age_medians.loc[(row["Pclass"], row["Sex"])], axis=1
    )
    return frame

train_df = fill_age(train_df)
test_df = fill_age(test_df)

# Остальные заполнения также используют статистики train.
embarked_mode = train_df["Embarked"].mode()[0]
fare_median = train_df["Fare"].median()
train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)
test_df["Fare"] = test_df["Fare"].fillna(fare_median)
train_df = train_df.drop(columns=["Cabin"])
test_df = test_df.drop(columns=["Cabin"])

# Создаём размер семьи, признак одиночного путешествия и возрастную группу.
for frame in (train_df, test_df):
    frame["Family_Size"] = frame["SibSp"] + frame["Parch"] + 1
    frame["Is_Alone"] = (frame["Family_Size"] == 1).astype(int)
    frame["Age_Group"] = pd.cut(
        frame["Age"], bins=[0, 12, 18, 35, 60, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"], include_lowest=True
    )

# Бинарный Sex кодируем LabelEncoder, Embarked — one-hot.
le_sex = LabelEncoder()
train_df["Sex_Encoded"] = le_sex.fit_transform(train_df["Sex"])
test_df["Sex_Encoded"] = le_sex.transform(test_df["Sex"])
train_df = pd.get_dummies(train_df, columns=["Embarked"], prefix="Embarked", drop_first=True, dtype=int)
test_df = pd.get_dummies(test_df, columns=["Embarked"], prefix="Embarked", drop_first=True, dtype=int)

print("Кодировка Sex:", dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))
print("Осталось пропусков в модельных признаках:", train_df.isna().sum().sum())
display(train_df[["Family_Size", "Is_Alone", "Age_Group", "Sex_Encoded", "Embarked_Q", "Embarked_S"]].head())

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Признаки доступны и для train, и для нового пассажира.
feature_cols = [
    "Pclass", "Sex_Encoded", "Age", "SibSp", "Parch", "Fare",
    "Family_Size", "Is_Alone", "Embarked_Q", "Embarked_S",
]
X = train_df[feature_cols].astype(float)
y = train_df["Survived"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Масштабируем без утечки статистик validation в train.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Инициализируем модели с фиксированным random_state.
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=3,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)

# Замеряем время обучения каждой модели.
training_times = {}
for name, model, features in [
    ("Logistic Regression", lr_model, X_train_scaled),
    ("Decision Tree", dt_model, X_train),
    ("Random Forest", rf_model, X_train),
]:
    started = time.perf_counter()
    model.fit(features, y_train)
    training_times[name] = time.perf_counter() - started
    print(f"{name}: {training_times[name]:.4f} с")

# %%
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

# Получаем классы и вероятности на одной и той же validation-выборке.
model_specs = {
    "Logistic Regression": (lr_model, X_valid_scaled),
    "Decision Tree": (dt_model, X_valid),
    "Random Forest": (rf_model, X_valid),
}
predictions = {}
metrics_rows = []
for name, (model, features) in model_specs.items():
    y_pred = model.predict(features)
    y_prob = model.predict_proba(features)[:, 1]
    predictions[name] = {"class": y_pred, "probability": y_prob}
    metrics_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_valid, y_pred),
        "Precision": precision_score(y_valid, y_pred),
        "Recall": recall_score(y_valid, y_pred),
        "F1": f1_score(y_valid, y_pred),
        "ROC-AUC": roc_auc_score(y_valid, y_prob),
        "Train time, s": training_times[name],
    })

results_table = pd.DataFrame(metrics_rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
display(results_table.style.format("{:.4f}"))

# Матрицы ошибок для всех сравниваемых моделей.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, (name, values) in zip(axes, predictions.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_valid, values["class"])).plot(
        ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set_title(name)
plt.tight_layout()
fig.savefig(EXAMPLES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# ROC-кривые и диагональ случайного классификатора.
roc_fig = plt.figure(figsize=(8, 6))
for name, values in predictions.items():
    fpr, tpr, _ = roc_curve(y_valid, values["probability"])
    auc_value = roc_auc_score(y_valid, values["probability"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Случайная модель")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые на validation-выборке")
plt.legend()
roc_fig.savefig(EXAMPLES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Коэффициенты линейной модели и важности двух деревьев.
lr_importance = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values()
dt_importance = pd.Series(dt_model.feature_importances_, index=feature_cols).sort_values()
rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
lr_importance.plot.barh(ax=axes[0], title="Коэффициенты Logistic Regression")
dt_importance.plot.barh(ax=axes[1], title="Feature importance Decision Tree")
rf_importance.plot.barh(ax=axes[2], title="Feature importance Random Forest")
for axis in axes:
    axis.set_xlabel("Вклад / важность")
plt.tight_layout()
plt.show()

display(pd.DataFrame({
    "LR coefficient": lr_importance,
    "DT importance": dt_importance,
    "RF importance": rf_importance,
}).sort_values("RF importance", ascending=False))

# %%
def prepare_passenger(passenger):
    '''Преобразовать словарь с сырыми полями в строку модельных признаков.'''
    family_size = passenger["SibSp"] + passenger["Parch"] + 1
    sex_encoded = int(le_sex.transform([passenger["Sex"]])[0])
    row = {
        "Pclass": passenger["Pclass"], "Sex_Encoded": sex_encoded,
        "Age": passenger["Age"], "SibSp": passenger["SibSp"],
        "Parch": passenger["Parch"], "Fare": passenger["Fare"],
        "Family_Size": family_size, "Is_Alone": int(family_size == 1),
        "Embarked_Q": int(passenger["Embarked"] == "Q"),
        "Embarked_S": int(passenger["Embarked"] == "S"),
    }
    return pd.DataFrame([row], columns=feature_cols).astype(float)

def predict_passenger(passenger, model_name=None):
    '''Вернуть прогноз (0/1), вероятность выживания и имя модели.'''
    selected_name = model_name or results_table["ROC-AUC"].idxmax()
    model_lookup = {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
    }
    row = prepare_passenger(passenger)
    model_input = scaler.transform(row) if selected_name == "Logistic Regression" else row
    probability = float(model_lookup[selected_name].predict_proba(model_input)[0, 1])
    return {"prediction": int(probability >= 0.5), "probability": probability, "model": selected_name}

# Пять пассажиров с разными сочетаниями пола, класса, возраста и семьи.
demo_passengers = [
    {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100, "Embarked": "C"},
    {"Pclass": 3, "Sex": "male", "Age": 35, "SibSp": 0, "Parch": 0, "Fare": 8, "Embarked": "S"},
    {"Pclass": 2, "Sex": "female", "Age": 8, "SibSp": 1, "Parch": 2, "Fare": 30, "Embarked": "S"},
    {"Pclass": 1, "Sex": "male", "Age": 54, "SibSp": 1, "Parch": 0, "Fare": 80, "Embarked": "C"},
    {"Pclass": 3, "Sex": "female", "Age": 22, "SibSp": 0, "Parch": 0, "Fare": 7.5, "Embarked": "Q"},
]
demo_results = []
for number, passenger in enumerate(demo_passengers, start=1):
    result = predict_passenger(passenger)
    demo_results.append(result)
    print(f"Пассажир {number}: класс={result['prediction']}, p={result['probability']:.3f}, модель={result['model']}")

# %%
# Создаём стандартные папки артефактов модуля.
MODELS_DIR = Path("models")
EXAMPLES_DIR = Path("examples")
MODELS_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True)

# Бинарные объекты sklearn сохраняем joblib.
joblib.dump(lr_model, MODELS_DIR / "lr_model.pkl")
joblib.dump(dt_model, MODELS_DIR / "dt_model.pkl")
joblib.dump(rf_model, MODELS_DIR / "rf_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(le_sex, MODELS_DIR / "le_sex.pkl")

# JSON хранит переносимые сведения без Python-объектов.
(MODELS_DIR / "feature_cols.json").write_text(
    json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8"
)
metrics_json = {
    model: {metric: float(value) for metric, value in row.items()}
    for model, row in results_table.to_dict(orient="index").items()
}
(MODELS_DIR / "metrics.json").write_text(
    json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8"
)
metadata = {
    "created": "2026-09-22", "python": platform.python_version(),
    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    "random_state": RANDOM_STATE, "target": "Survived",
    "sex_classes": le_sex.classes_.tolist(),
}
(MODELS_DIR / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Сохранено:", sorted(path.name for path in MODELS_DIR.iterdir()))

# %%
from io import BytesIO
import requests

# Вставьте raw URL после создания публичного репозитория.
GITHUB_RAW_BASE = ""  # пример: https://raw.githubusercontent.com/USER/REPO/main/module-1-titanic

if GITHUB_RAW_BASE:
    # Загружаем бинарные объекты и JSON по HTTPS с проверкой статуса.
    def load_remote_joblib(relative_path):
        response = requests.get(f"{GITHUB_RAW_BASE}/{relative_path}", timeout=30)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    loaded_models = {
        "Logistic Regression": load_remote_joblib("models/lr_model.pkl"),
        "Decision Tree": load_remote_joblib("models/dt_model.pkl"),
        "Random Forest": load_remote_joblib("models/rf_model.pkl"),
    }
    loaded_scaler = load_remote_joblib("models/scaler.pkl")
    loaded_features = requests.get(
        f"{GITHUB_RAW_BASE}/models/feature_cols.json", timeout=30
    ).json()
    source_label = "GitHub raw"
else:
    # Локальный round-trip проверяет сериализацию до публикации.
    loaded_models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "lr_model.pkl"),
        "Decision Tree": joblib.load(MODELS_DIR / "dt_model.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "rf_model.pkl"),
    }
    loaded_scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    loaded_features = json.loads((MODELS_DIR / "feature_cols.json").read_text(encoding="utf-8"))
    source_label = "локальные сохранённые файлы"

# Сравниваем вероятности загруженной и исходной модели на трёх примерах.
best_model_name = results_table["ROC-AUC"].idxmax()
for number, passenger in enumerate(demo_passengers[:3], start=1):
    row = prepare_passenger(passenger)[loaded_features]
    loaded_input = loaded_scaler.transform(row) if best_model_name == "Logistic Regression" else row
    loaded_probability = float(loaded_models[best_model_name].predict_proba(loaded_input)[0, 1])
    original_probability = demo_results[number - 1]["probability"]
    assert np.isclose(loaded_probability, original_probability), "Предсказания не совпали"
    print(f"Пример {number}: {loaded_probability:.6f} — совпадает")
print("Источник загрузки:", source_label)

# %%
from IPython.display import Markdown, display

# Подставляем фактические значения лучшей модели в итоговый текст.
best_name = results_table["ROC-AUC"].idxmax()
best = results_table.loc[best_name]
display(Markdown(f'''
### Что получилось

- Достигнут ROC-AUC **{best['ROC-AUC']:.3f}** на отложенной выборке.
- Лучшая по ROC-AUC модель: **{best_name}**; Accuracy = **{best['Accuracy']:.3f}**, F1 = **{best['F1']:.3f}**.
- Время её обучения: **{best['Train time, s']:.4f} с** на текущем компьютере.
- Главные наблюдения EDA: более высокая выживаемость женщин и пассажиров высоких классов; `Cabin` имеет критически много пропусков.

### Трудности

- Около 20% значений `Age` пришлось восстанавливать групповыми медианами.
- `Cabin` содержит около 77% пропусков и исключён из базовой модели.
- Классы умеренно несбалансированы: 61,6% против 38,4%.
- Глубина дерева ограничена, чтобы уменьшить переобучение.
- Удалённая проверка GitHub станет доступна только после публикации репозитория и заполнения `GITHUB_RAW_BASE`.

### Возможные улучшения

1. Извлечь обращение (`Title`: Mr, Mrs, Miss и т. п.) из `Name`.
2. Добавить взаимодействие `Pclass × Sex`.
3. Проверить SMOTE только внутри обучающих фолдов.
4. Настроить Random Forest и градиентный бустинг.
5. Подобрать гиперпараметры через `GridSearchCV` со стратифицированной кросс-валидацией.
6. Применить `np.log1p(Fare)` и сравнить метрики.
7. Настроить порог классификации под цену ошибок, а не всегда использовать 0,5.
'''))

# %%
# Записываем зависимости, необходимые для воспроизведения проекта.
requirements = '''numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
joblib>=1.3
requests>=2.31
nbformat>=5.9
'''
Path("requirements.txt").write_text(requirements, encoding="utf-8")

# Собираем полный код из текущего ноутбука с разделителями # %%.
notebook_candidates = [
    Path("Модуль 1. Практическое задание 1.ipynb"),
    Path("notebook.ipynb"),
]
notebook_path = next(path for path in notebook_candidates if path.exists())
current_notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
script_parts = []
for cell in current_notebook["cells"]:
    if cell["cell_type"] == "code":
        script_parts.append("# %%\n" + "".join(cell.get("source", [])))
full_script = "\n\n".join(script_parts)
Path("appendix_full_code.py").write_text(full_script, encoding="utf-8")
print("Созданы requirements.txt и appendix_full_code.py")
print("Число экспортированных кодовых ячеек:", len(script_parts))