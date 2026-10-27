"""Полный код лабораторной №3; запускать из каталога модуля."""


# %%
# Импорты и фиксированные параметры воспроизводимости.
import os, sys, json, time, hashlib, platform, warnings
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, requests, optuna, shap
from IPython.display import display, Markdown
from scipy.special import expit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from churn_pipeline import ChurnPreprocessor, predict_churn, load_remote_model

# Относительные пути: запускать из каталога module-3-churn.
SEED, THREADS = 42, 4
for folder in ['models', 'examples', 'data']:
    Path(folder).mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 100, 'font.family': 'DejaVu Sans'})
df = pd.read_csv('data/telco_churn.csv')
print(df.shape)
print(df.head())
# Проверяем именно обязательный набор без дубликатов клиентов.
assert df.shape == (7043, 21) and df.customerID.is_unique
assert df.Churn.value_counts().to_dict() == {'No': 5174, 'Yes': 1869}
display(df.Churn.value_counts().rename('count').to_frame().assign(share=df.Churn.value_counts(normalize=True)))
df.info()
display(df.describe(include='all').T)

# %%
# Цель кодируется явно; target и ID не подаются в модели.
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'No': 0, 'Yes': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
print('Train / test:', X_train.shape, X_test.shape, 'Churn:', y_train.mean(), y_test.mean())
# EDA использует только train, никаких новых решений по тестовым графикам.
eda = X_train.copy()
eda['Churn'] = y_train
eda['TotalCharges'] = pd.to_numeric(eda['TotalCharges'], errors='coerce')
print('Пропуски после to_numeric:')
display(eda.isna().sum().to_frame('missing'))

# Шесть обязательных EDA-панелей: классы, стаж, платежи, корреляции, контракты, интернет.
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
sns.countplot(data=eda, x='Churn', ax=axes[0, 0])
axes[0, 0].set_title('Распределение Churn (train)')
sns.histplot(data=eda, x='tenure', hue='Churn', bins=24, ax=axes[0, 1])
axes[0, 1].set_title('Стаж и отток')
sns.boxplot(data=eda, x='Churn', y='MonthlyCharges', ax=axes[0, 2])
axes[0, 2].set_title('Ежемесячный платёж')
sns.heatmap(eda[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr(), annot=True, fmt='.2f', ax=axes[1, 0], cmap='coolwarm', center=0)
axes[1, 0].set_title('Корреляции (pairwise complete)')
for axis, column in zip(axes[1, 1:], ['Contract', 'InternetService']):
    # Доли без скрытого усреднения и без случайных bootstrap-интервалов.
    rates = eda.groupby(column)['Churn'].mean()
    axis.bar(rates.index, rates.values)
    axis.set(title=f'Отток по {column}', ylabel='Доля Churn=1')
    axis.tick_params(axis='x', rotation=15)
fig.tight_layout()
fig.savefig('examples/eda_plots.png', bbox_inches='tight')
plt.show()
# Отдельно визуализируем пропуски, включая нулевые значения остальных столбцов.
fig, ax = plt.subplots(figsize=(10, 3))
eda.isna().sum().plot.bar(ax=ax)
ax.set(title='Пропуски в train после преобразования', ylabel='Число значений')
fig.tight_layout()
fig.savefig('examples/missing_values.png')
plt.show()

# Демонстрация преобразований; для CV далее создаются новые экземпляры внутри pipeline.
preview = ChurnPreprocessor().fit(X_train)
display(preview.transform(X_train).head())
assert preview.transform(X_train).shape[1] == 23
display(eda.groupby('Contract')['Churn'].agg(['mean', 'size']))
display(eda.groupby(eda.tenure <= 6)['Churn'].agg(['mean', 'size']))

# %%
# Фолды одинаковы у всех baseline; каждый fold получает собственную предобработку.
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
ratio = float((y_train == 0).sum() / (y_train == 1).sum())
estimators = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=THREADS),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=.1, subsample=.9, colsample_bytree=.9, scale_pos_weight=ratio, random_state=SEED, eval_metric='logloss', n_jobs=THREADS),
    'LightGBM': LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=.05, class_weight='balanced', random_state=SEED, verbosity=-1, n_jobs=THREADS),
    'CatBoost': CatBoostClassifier(iterations=200, depth=6, learning_rate=.1, auto_class_weights='Balanced', random_seed=SEED, verbose=False, thread_count=THREADS, allow_writing_files=False),
}
models, cv_stats, fit_times = {}, {}, {}
for name, estimator in estimators.items():
    # Pipeline исключает утечку медианы и кодировщиков между CV-фолдами.
    pipeline = Pipeline([('preprocess', ChurnPreprocessor()), ('model', estimator)])
    started = time.perf_counter()
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    cv_stats[name] = {'mean': float(scores.mean()), 'std': float(scores.std()), 'folds': scores.tolist(), 'seconds': time.perf_counter() - started}
    started = time.perf_counter()
    models[name] = pipeline.fit(X_train, y_train)
    fit_times[name] = time.perf_counter() - started
    print(name, 'CV:', round(scores.mean(), 5), '+/-', round(scores.std(), 5), 'fit, s:', round(fit_times[name], 3))
# Здесь лучшая baseline определяется по CV, до отображения любых тестовых метрик.
baseline_cv_winner = max(cv_stats, key=lambda n: cv_stats[n]['mean'])
print('Лучшая baseline по CV:', baseline_cv_winner)

# %%
# Единая функция метрик; нулевой precision/recall обрабатывается явно.
def evaluate(model, name):
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= .5).astype(int)
    return {'Model': name, 'Accuracy': accuracy_score(y_test, prediction),
        'Precision': precision_score(y_test, prediction, zero_division=0),
        'Recall': recall_score(y_test, prediction, zero_division=0),
        'F1': f1_score(y_test, prediction, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, probability),
        'PR-AUC': average_precision_score(y_test, probability)}
results = pd.DataFrame([evaluate(model, name) for name, model in models.items()]).set_index('Model')
results['CV ROC-AUC'] = pd.Series({n: v['mean'] for n, v in cv_stats.items()})
results['Fit seconds'] = pd.Series(fit_times)
display(results.round(5))

# ROC и confusion matrix сохраняются также отдельными PNG для сдачи.
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    RocCurveDisplay.from_predictions(y_test, model.predict_proba(X_test)[:, 1], name=name, ax=ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC-кривые baseline, holdout')
fig.tight_layout()
fig.savefig('examples/roc_curves.png')
plt.show()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, models[baseline_cv_winner].predict(X_test)), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set(title=f'{baseline_cv_winner}: выбор по CV', xlabel='Предсказание (1 = уход)', ylabel='Истинный класс')
fig.tight_layout()
fig.savefig('examples/confusion_matrix_baseline.png')
plt.show()
# Динамический комментарий опирается на измеренные результаты, а не на ожидаемые цифры.
display(Markdown(f"Лучшая baseline по CV — **{baseline_cv_winner}**. Её holdout ROC-AUC = **{results.loc[baseline_cv_winner, 'ROC-AUC']:.4f}**. "
    f"Самый быстрый финальный fit: **{min(fit_times, key=fit_times.get)}**. При пороге 0.5 оцениваем компромисс Precision/Recall по таблице."))

# %%
# Имена преобразованных признаков одинаковы для обеих моделей.
feature_cols = models['LightGBM'].named_steps['preprocess'].get_feature_names_out().tolist()
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for axis, name in zip(axes, ['Random Forest', 'LightGBM']):
    importance = pd.Series(models[name].named_steps['model'].feature_importances_, index=feature_cols).sort_values()
    importance.tail(15).plot.barh(ax=axis)
    axis.set(title=f'{name}: ТОП-15', xlabel='Важность внутри модели')
    importance.sort_values(ascending=False).to_csv(f'examples/{name.lower().replace(" ", "_")}_importance.csv')
fig.tight_layout()
fig.savefig('examples/feature_importance.png')
plt.show()
# Бизнес-таблицы используют только train; показываем также размеры групп.
for column in ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']:
    display(eda.groupby(column)['Churn'].agg(['mean', 'size']).rename(columns={'mean': 'churn_rate', 'size': 'n'}))

# %%
# Убираем подробный лог каждой попытки из отчёта; история сохраняется в CSV.
optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    # Число листьев ограничиваем глубиной, исключая бессмысленные комбинации.
    depth = trial.suggest_int('max_depth', 3, 9)
    params = {'max_depth': depth, 'num_leaves': trial.suggest_int('num_leaves', 7, min(63, 2**depth)),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', .01, .2, log=True),
        'subsample': trial.suggest_float('subsample', .6, 1.),
        'colsample_bytree': trial.suggest_float('colsample_bytree', .6, 1.),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'reg_alpha': trial.suggest_float('reg_alpha', .001, 10., log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', .001, 10., log=True)}
    estimator = LGBMClassifier(**params, subsample_freq=1, class_weight='balanced', random_state=SEED, verbosity=-1, n_jobs=THREADS)
    pipeline = Pipeline([('preprocess', ChurnPreprocessor()), ('model', estimator)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    trial.set_user_attr('fold_scores', scores.tolist())
    return float(scores.mean())

# Последовательный поиск воспроизводит порядок предложений TPE.
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
search_start = time.perf_counter()
study.optimize(objective, n_trials=30, n_jobs=1)
search_seconds = time.perf_counter() - search_start
study.trials_dataframe().to_csv('examples/optuna_trials.csv', index=False)
print('Best CV ROC-AUC:', study.best_value, 'Search seconds:', search_seconds)
print(json.dumps(study.best_params, indent=2))

# Переобучаем победителя поиска на train, без использования holdout.
tuned_model = Pipeline([('preprocess', ChurnPreprocessor()), ('model', LGBMClassifier(
    **study.best_params, subsample_freq=1, class_weight='balanced', random_state=SEED, verbosity=-1, n_jobs=THREADS))])
started = time.perf_counter()
tuned_model.fit(X_train, y_train)
fit_times['LightGBM (Optuna)'] = time.perf_counter() - started
cv_stats['LightGBM (Optuna)'] = {'mean': study.best_value, 'std': float(np.std(study.best_trial.user_attrs['fold_scores'])), 'folds': study.best_trial.user_attrs['fold_scores']}
# Правило выбора зафиксировано заранее: максимум CV среди двух LightGBM.
best_name = 'LightGBM (Optuna)' if study.best_value > cv_stats['LightGBM']['mean'] else 'LightGBM'
best_model = tuned_model if best_name == 'LightGBM (Optuna)' else models['LightGBM']
models['LightGBM (Optuna)'] = tuned_model
results.loc['LightGBM (Optuna)'] = {**evaluate(tuned_model, 'LightGBM (Optuna)'), 'CV ROC-AUC': study.best_value, 'Fit seconds': fit_times['LightGBM (Optuna)']}
display(results.round(5))
gain = results.loc['LightGBM (Optuna)', 'ROC-AUC'] - results.loc['LightGBM', 'ROC-AUC']
display(Markdown(f"Выбран **{best_name}**. Изменение holdout ROC-AUC относительно baseline LightGBM: **{gain:+.4f}** "
    "(абсолютная разница; улучшение не гарантировано и статистическая значимость не проверена)."))

# Полный набор метрик и confusion matrix финальной модели.
results.to_csv('examples/metrics.csv')
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set(title=best_name, xlabel='Предсказание', ylabel='Истинный класс')
fig.tight_layout()
fig.savefig('examples/confusion_matrix.png')
plt.show()
# Таблица метрик в PNG — готовый скриншот для преподавателя.
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('off')
table = ax.table(cellText=results.round(4).values, rowLabels=results.index, colLabels=results.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
fig.savefig('examples/metrics.png', bbox_inches='tight', dpi=160)
plt.show()

# %%
# Для TreeExplainer берём estimator из полного pipeline и ровно его входные признаки.
encoded_test = best_model.named_steps['preprocess'].transform(X_test)
tree = best_model.named_steps['model']
explainer = shap.TreeExplainer(tree, model_output='raw')
explanation = explainer(encoded_test)
values = explanation.values
base_values = np.asarray(explanation.base_values)
# Совместимость с API, возвращающим измерение класса.
if values.ndim == 3:
    values = values[:, :, 1]
    base_values = base_values[:, 1]
base_values = np.broadcast_to(base_values, (len(encoded_test),))
raw_score = tree.predict(encoded_test, raw_score=True)
assert np.allclose(base_values + values.sum(axis=1), raw_score, atol=1e-5)
assert np.allclose(expit(raw_score), best_model.predict_proba(X_test)[:, 1], atol=1e-7)

# Summary и bar: глобальная структура вкладов, без причинной интерпретации.
shap.summary_plot(values, encoded_test, show=False)
plt.savefig('examples/shap_summary.png', bbox_inches='tight')
plt.show()
shap.summary_plot(values, encoded_test, plot_type='bar', show=False)
plt.savefig('examples/shap_bar.png', bbox_inches='tight')
plt.show()
# Локальное объяснение объекта с наибольшим риском; индекс сопоставлен с исходным CSV.
worst_position = int(np.argmax(best_model.predict_proba(X_test)[:, 1]))
local = shap.Explanation(values=values[worst_position], base_values=base_values[worst_position], data=encoded_test.iloc[worst_position].values, feature_names=feature_cols)
shap.plots.waterfall(local, max_display=12, show=False)
plt.savefig('examples/shap_waterfall.png', bbox_inches='tight')
plt.show()
# Dependence для tenure понятен в исходных месяцах; дополнительная окраска — платёж.
shap.dependence_plot('tenure', values, encoded_test, interaction_index='MonthlyCharges', show=False)
plt.savefig('examples/shap_dependence.png', bbox_inches='tight')
plt.show()
shap_ranking = pd.Series(np.abs(values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
display(shap_ranking.head(10).to_frame('mean_abs_SHAP_log_odds'))
local_table = pd.DataFrame({'feature': feature_cols, 'encoded_value': encoded_test.iloc[worst_position].values, 'SHAP_log_odds': values[worst_position]})
display(local_table.iloc[np.argsort(-np.abs(values[worst_position]))[:8]])
print('customerID:', df.loc[X_test.index[worst_position], 'customerID'])
print('Score:', best_model.predict_proba(X_test)[worst_position, 1], 'Base log-odds:', base_values[worst_position])
display(Markdown('Наибольшие средние абсолютные SHAP-вклады: **' + ', '.join(shap_ranking.head(5).index) + '**. '
    'В локальной таблице положительный вклад повышает raw score ухода, отрицательный снижает.'))
# Сохраняем числа для независимой проверки SHAP и объяснения на защите.
np.savez_compressed('examples/shap_values.npz', values=values, base_values=base_values, raw_score=raw_score, test_indices=X_test.index.to_numpy())
shap_ranking.to_csv('examples/shap_importance.csv')
local_table.to_csv('examples/shap_local_client.csv', index=False)

# %%
# Имена файлов соответствуют методичке, артефакты сжаты для GitHub.
file_names = {'Random Forest': 'rf_model.pkl', 'XGBoost': 'xgb_model.pkl',
    'LightGBM': 'lgbm_model.pkl', 'CatBoost': 'cat_model.pkl', 'LightGBM (Optuna)': 'tuned_lgbm_model.pkl'}
for name, file_name in file_names.items():
    joblib.dump(models[name], Path('models') / file_name, compress=3)
joblib.dump(best_model, 'models/best_model.pkl', compress=3)
preprocessor = best_model.named_steps['preprocess']
joblib.dump(preprocessor.encoders_, 'models/label_encoders.pkl')
# Scaler обучается исключительно на train, но в деревьях не используется.
scaler = StandardScaler().fit(preprocessor.transform(X_train))
joblib.dump(scaler, 'models/scaler.pkl')
Path('models/feature_cols.json').write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding='utf-8')

# Записываем происхождение данных, версии и точные индексы разбиения.
packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib', 'requests',
    'xgboost', 'lightgbm', 'catboost', 'optuna', 'shap', 'nbformat', 'nbclient', 'ipykernel']
versions = {name: version(name) for name in packages}
Path('requirements.txt').write_text(''.join(f'{name}=={v}\n' for name, v in versions.items()), encoding='utf-8')
data_hash = hashlib.sha256(Path('data/telco_churn.csv').read_bytes()).hexdigest()
metadata = {'date': '2026-10-27', 'student': 'Федорушкин Станислав', 'seed': SEED, 'python': platform.python_version(),
    'versions': versions, 'best_model': best_name, 'selection': 'LightGBM baseline vs tuned: train 5-fold CV ROC-AUC',
    'baseline_cv_winner': baseline_cv_winner, 'threshold': .5, 'score_calibrated': False,
    'train_rows': len(X_train), 'test_rows': len(X_test), 'raw_features': list(X_train.columns),
    'engineered_features': feature_cols, 'data_sha256': data_hash, 'scaler_used_in_model': False,
    'train_indices': X_train.index.tolist(), 'test_indices': X_test.index.tolist(), 'remote_verified': False}
metrics = {'test': results.to_dict(orient='index'), 'cv': cv_stats, 'best_model': best_name,
    'best_test_roc_auc': float(results.loc[best_name, 'ROC-AUC']), 'best_cv_roc_auc': cv_stats[best_name]['mean'],
    'optuna_best_params': study.best_params, 'optuna_trials': len(study.trials), 'search_seconds': search_seconds}
for filename, content in [('metadata.json', metadata), ('metrics.json', metrics)]:
    Path('models', filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
provenance = {'download_date': '2026-10-27', 'url': 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
    'assignment_source': 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn', 'sha256': data_hash,
    'rows': len(df), 'columns': len(df.columns), 'note': 'Original CSV, unchanged bytes; preprocessing is inside pipelines.'}
Path('data/provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
print('Сохранено: модели, предобработка, признаки, метрики, метаданные, версии.')

# %%
# Проверяем сериализацию на трёх исходных клиентах.
clients = X_test.iloc[:3]
loaded = joblib.load('models/best_model.pkl')
original_scores = best_model.predict_proba(clients)[:, 1]
loaded_scores = loaded.predict_proba(clients)[:, 1]
assert np.array_equal(best_model.predict(clients), loaded.predict(clients))
assert np.allclose(original_scores, loaded_scores, atol=1e-12)
display(predict_churn(clients, loaded))
print('Локальные классы и score совпадают на трёх клиентах.')

# Для проверки опубликованного репозитория включите переменную окружения и выполните ячейку.
BASE_URL = 'https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-3-churn'
if os.environ.get('CHURN_VERIFY_REMOTE') == '1':
    remote = load_remote_model(BASE_URL)
    response = requests.get(f'{BASE_URL}/models/feature_cols.json', timeout=60)
    response.raise_for_status()
    assert response.json() == feature_cols
    assert np.array_equal(remote.predict(clients), best_model.predict(clients))
    assert np.allclose(remote.predict_proba(clients), best_model.predict_proba(clients))
    metadata['remote_verified'] = True
    Path('models/metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('GitHub raw: классы и score совпадают на трёх клиентах.')
else:
    print('GitHub raw НЕ проверен: сначала опубликуйте модуль, затем включите CHURN_VERIFY_REMOTE=1.')

# %%
# Итоговые числа берутся из вычисленных таблиц, без ручного копирования метрик.
row = results.loc[best_name]
display(Markdown(f"Итоговая модель: **{best_name}**, CV ROC-AUC **{cv_stats[best_name]['mean']:.4f}**, "
    f"holdout ROC-AUC **{row['ROC-AUC']:.4f}**, F1 **{row['F1']:.4f}**, Recall **{row['Recall']:.4f}**. "
    f"Топ SHAP: {', '.join(shap_ranking.head(5).index)}. Локальная загрузка проверена, GitHub raw требует публикации."))
