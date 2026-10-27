# Зависимости: requirements.txt
# %%
# Импортируем библиотеки и ограничиваем BLAS для воспроизводимого времени.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import json
import time
import platform
import warnings
import hashlib
import importlib.metadata as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display, Markdown
from scipy import stats
from threadpoolctl import threadpool_limits
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from housing_pipeline import make_pipeline, predict_price, load_remote_model

# Новые категории ожидаемы внутри CV: ignore кодирует их нулями.
# Скрываем только повторяющееся уведомление энкодера, остальные предупреждения видны.
warnings.filterwarnings('ignore', message='Found unknown categories in columns.*',
                        category=UserWarning, module=r'sklearn\.preprocessing\._encoders')
# Запускаем notebook из каталога модуля; все файлы имеют относительные пути.
assert Path('housing_pipeline.py').exists(), 'Запускайте из module-2-house-prices'
for folder in ['models', 'examples']:
    Path(folder).mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', font='DejaVu Sans')
train_df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=['NA', ''])
test_df = pd.read_csv('data/test.csv', keep_default_na=False, na_values=['NA', ''])
print(train_df.shape, test_df.shape)
assert train_df.shape == (1460, 81) and test_df.shape == (1459, 80)
assert train_df.Id.is_unique and test_df.Id.is_unique
assert set(train_df.Id).isdisjoint(test_df.Id)
display(train_df.head())
train_df.info()
display(train_df.describe())
# Статистики считаются по данным: текстовые категории None не теряются при чтении.
display(train_df.SalePrice.describe().to_frame())
print('skew:', train_df.SalePrice.skew(), 'kurtosis:', train_df.SalePrice.kurtosis())
print('Числовых признаков:', train_df.drop(columns=['Id','SalePrice']).select_dtypes('number').shape[1])

# %%
# Отделяем Id и цель; SalePrice_log никогда не входит в признаки.
X = train_df.drop(columns=['Id', 'SalePrice'])
y = np.log1p(train_df.SalePrice)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=42)
eda = train_df.loc[X_train.index].copy()
missing = eda.isna().sum().sort_values(ascending=False)
display(missing[missing > 0].to_frame('Пропусков'))
q1, q3 = eda.GrLivArea.quantile([.25, .75])
iqr_mask = (eda.GrLivArea < q1-1.5*(q3-q1)) | (eda.GrLivArea > q3+1.5*(q3-q1))
outlier_mask = (X_train.GrLivArea > 4000) & (np.expm1(y_train) < 300000)
print('IQR кандидатов:', iqr_mask.sum(), 'Удаляем по доменному правилу:', outlier_mask.sum())
display(eda.loc[outlier_mask, ['Id', 'GrLivArea', 'SalePrice']])

# Шесть графиков с отдельными заголовками и подписями осей.
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
sns.histplot(eda.SalePrice, kde=True, ax=axes[0,0])
axes[0,0].set(title='Цена продажи', xlabel='SalePrice, USD', ylabel='Число домов')
sns.histplot(np.log1p(eda.SalePrice), kde=True, ax=axes[0,1])
axes[0,1].set(title='Логарифм цены', xlabel='log1p(SalePrice)', ylabel='Число домов')
top = eda.select_dtypes('number').corr().SalePrice.abs().nlargest(11).index
sns.heatmap(eda[top].corr(), cmap='coolwarm', center=0, ax=axes[0,2])
axes[0,2].set(title='Корреляции 10 признаков и цели', xlabel='Признак', ylabel='Признак')
sns.boxplot(data=eda, x='OverallQual', y='GrLivArea', ax=axes[1,0])
axes[1,0].set(title='Площадь и качество', xlabel='OverallQual, баллы', ylabel='GrLivArea, кв. футы')
sns.scatterplot(data=eda, x='GrLivArea', y='SalePrice', hue=outlier_mask.map({True:'Удаляем',False:'Оставляем'}), ax=axes[1,1])
axes[1,1].set(title='Выбросы площади и цены', xlabel='GrLivArea, кв. футы', ylabel='Цена, USD')
(missing[missing > 0] / len(eda) * 100).sort_values().plot.barh(ax=axes[1,2])
axes[1,2].set(title='Пропуски', xlabel='Доля, %', ylabel='Признак')
fig.tight_layout()
fig.savefig('examples/eda_plots.png', dpi=140)
plt.show()
print('Асимметрия до/после log1p:', eda.SalePrice.skew(), np.log1p(eda.SalePrice).skew())
print('Корреляция GarageCars/GarageArea:', eda.GarageCars.corr(eda.GarageArea))
# Убираем аномалии только из обучения; все 292 holdout-дома сохраняются.
X_train, y_train = X_train.loc[~outlier_mask], y_train.loc[~outlier_mask]
print('Размер обучения и holdout:', len(X_train), len(X_valid))

# %%
# Показываем, что признаки создаются без изучения holdout и без целевой переменной.
preview_pipeline = make_pipeline(Ridge())
prepared = preview_pipeline[:-1].fit_transform(X_train, y_train)
feature_names = preview_pipeline.named_steps['preprocess'].get_feature_names_out().tolist()
assert np.isfinite(prepared).all()
assert 'SalePrice' not in feature_names and 'Id' not in feature_names
print('После преобразований:', prepared.shape, 'NaN:', np.isnan(prepared).sum())
display(preview_pipeline.named_steps['features'].transform(X_train).head())

# %%
# Сетки заданы до просмотра результатов holdout.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
specs = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'model__alpha': [.1, 1, 10, 30, 100]}),
    'Lasso': (Lasso(max_iter=30000, tol=1e-5), {'model__alpha': [.0001, .0003, .0005, .001, .003]}),
    'ElasticNet': (ElasticNet(max_iter=30000, tol=1e-5),
                   {'model__alpha': [.0003, .001, .003], 'model__l1_ratio': [.2, .5, .8]})
}
models, training_rows, searches = {}, [], {}
with threadpool_limits(limits=1):
    for name, (regressor, grid) in specs.items():
        # GridSearch обучает весь pipeline внутри каждого CV-разбиения.
        start = time.perf_counter()
        search = GridSearchCV(make_pipeline(regressor), grid, cv=cv,
                              scoring='neg_root_mean_squared_error', n_jobs=1, error_score='raise')
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        models[name], searches[name] = search.best_estimator_, search
        idx = search.best_index_
        training_rows.append({'model': name, 'cv_rmse_log': -search.best_score_,
                              'cv_std': search.cv_results_['std_test_score'][idx],
                              'search_seconds': elapsed, 'fit_seconds': search.refit_time_,
                              'parameters': search.best_params_})
training = pd.DataFrame(training_rows).set_index('model')
best_name = training.cv_rmse_log.idxmin()
best_model = models[best_name]
display(training)
print('Выбрана до проверки holdout:', best_name)

# %%
# Вычисляем метрики в двух шкалах, сохраняя предсказания каждой модели.
rows, predictions = [], {}
actual = np.expm1(y_valid)
for name, model in models.items():
    z = model.predict(X_valid)
    prices = np.expm1(z)
    predictions[name] = prices
    rows.append({'model':name, 'rmse_log':np.sqrt(mean_squared_error(y_valid,z)),
                 'mae_log':mean_absolute_error(y_valid,z), 'r2_log':r2_score(y_valid,z),
                 'rmse_usd':np.sqrt(mean_squared_error(actual,prices)),
                 'mae_usd':mean_absolute_error(actual,prices), 'r2_usd':r2_score(actual,prices)})
metrics = pd.DataFrame(rows).set_index('model')
display(metrics)
pred = predictions[best_name]
residuals = actual - pred
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(actual, pred, alpha=.6)
lims = [min(actual.min(),pred.min()),max(actual.max(),pred.max())]
ax.plot(lims,lims,'r--')
ax.set(xlabel='Фактическая цена, USD', ylabel='Прогноз, USD', title=f'Holdout: {best_name}')
fig.tight_layout()
fig.savefig('examples/predictions_scatter.png', dpi=140)
plt.show()
# Диагностика остатков: доллары и QQ-график логарифмических ошибок.
fig, axes = plt.subplots(1,2,figsize=(13,5))
axes[0].scatter(pred,residuals,alpha=.6)
axes[0].axhline(0,color='red',linestyle='--')
axes[0].set(xlabel='Прогноз, USD',ylabel='Факт − прогноз, USD',title='Остатки на holdout')
stats.probplot(y_valid-best_model.predict(X_valid), dist='norm', plot=axes[1])
axes[1].set(title='QQ: остатки log1p',xlabel='Теоретические квантили',ylabel='Наблюдаемые квантили')
fig.tight_layout()
fig.savefig('examples/residuals.png',dpi=140)
plt.show()
worst = pd.DataFrame({'Id':train_df.loc[X_valid.index,'Id'], 'actual':actual,
                     'prediction':pred, 'residual':residuals})
display(worst.assign(abs_error=abs(residuals)).nlargest(5,'abs_error'))
print('Среднее остатка USD:',residuals.mean(), 'Std USD:',residuals.std())
display(Markdown(f'По CV выбрана **{best_name}**. На holdout RMSE(log) = **{metrics.loc[best_name,"rmse_log"]:.4f}**, '
                 f'R²(log) = **{metrics.loc[best_name,"r2_log"]:.4f}**, R²(USD) = **{metrics.loc[best_name,"r2_usd"]:.4f}**.'))

# %%
# Имена кодированных признаков получаем из fitted ColumnTransformer.
coef_tables = {}
for name, model in models.items():
    names = model.named_steps['preprocess'].get_feature_names_out()
    coef_tables[name] = pd.Series(model.named_steps['model'].coef_,index=names,name=name)
coefs = coef_tables[best_name]
display(coefs.nlargest(5).to_frame('Положительные коэффициенты'))
display(coefs.nsmallest(5).to_frame('Отрицательные коэффициенты'))
lasso_coef = coef_tables['Lasso']
print('Lasso обнулил:', int((lasso_coef == 0).sum()), 'из', len(lasso_coef))
display(lasso_coef[lasso_coef != 0].abs().nlargest(15).to_frame('|Lasso coefficient|'))
top_coef = coefs.loc[coefs.abs().nlargest(20).index].sort_values()
fig, ax = plt.subplots(figsize=(10,8))
top_coef.plot.barh(ax=ax,color=['#c44747' if v<0 else '#277e59' for v in top_coef])
ax.axvline(0,color='black',linewidth=.8)
ax.set(title=f'20 крупнейших коэффициентов: {best_name}',xlabel='Коэффициент log1p цены',ylabel='Признак')
fig.tight_layout()
fig.savefig('examples/coefficients.png',dpi=140)
plt.show()

# %%
# Пять объектов holdout по квантилям фактической цены, без повторного обучения.
ordered = y_valid.sort_values().index
demo_idx = ordered[np.linspace(0,len(ordered)-1,5,dtype=int)]
demo = X_valid.loc[demo_idx]
demo_table = demo[['OverallQual','GrLivArea','YearBuilt','GarageCars']].copy()
demo_table['Факт USD'] = np.expm1(y_valid.loc[demo_idx])
demo_table['Прогноз USD'] = predict_price(demo,best_model)
demo_table['Ошибка USD'] = demo_table['Прогноз USD'] - demo_table['Факт USD']
display(demo_table)
# Словарь одного дома возвращает float; неизвестная категория не ломает схему.
partial = {'OverallQual':6,'GrLivArea':1500,'YearBuilt':2000,'Neighborhood':'CollgCr'}
print('Неполный запрос USD:',predict_price(partial,best_model))
new_area = dict(partial, Neighborhood='NewNeighborhood')
print('Новый район USD:',predict_price(new_area,best_model))

# %%
# Имена артефактов соответствуют методичке; файл best_model — полный pipeline.
filenames = {'Linear Regression':'lr_model.pkl','Ridge':'ridge_model.pkl',
             'Lasso':'lasso_model.pkl','ElasticNet':'enet_model.pkl'}
for name, model in models.items():
    joblib.dump(model,Path('models')/filenames[name],compress=3)
joblib.dump(best_model,'models/best_model.pkl',compress=3)
prep = best_model.named_steps['preprocess']
joblib.dump(prep.named_transformers_['num'].named_steps['scale'],'models/scaler.pkl')
names = prep.get_feature_names_out().tolist()
Path('models/feature_cols.json').write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding='utf-8')
Path('models/metrics.json').write_text(metrics.to_json(orient='index',indent=2),encoding='utf-8')
Path('models/training.json').write_text(training.to_json(orient='index',indent=2),encoding='utf-8')
# Сохраняем разделение, версии и контрольные суммы для повторения эксперимента.
metadata = {'author':'Федорушкин Станислав','group':'ПКТб-23-1','date':'2026-10-27',
            'best_model':best_name,'selection_metric':'5-fold CV RMSE(log1p)',
            'random_state':42,'n_train':len(X_train),'n_holdout':len(X_valid),
            'train_ids':train_df.loc[X_train.index,'Id'].tolist(),
            'holdout_ids':train_df.loc[X_valid.index,'Id'].tolist(),
            'removed_ids':eda.loc[outlier_mask,'Id'].tolist(),
            'python':platform.python_version(),
            'versions':{p:md.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib']},
            'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data').glob('*.csv')},
            'refit_on_holdout':False}
Path('models/metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
metrics.to_csv('examples/metrics.csv')
submission = pd.DataFrame({'Id':test_df.Id,'SalePrice':predict_price(test_df,best_model)})
assert len(submission)==1459 and submission.SalePrice.gt(0).all()
submission.to_csv('submission.csv',index=False)
print('Артефакты сохранены; submission:',submission.shape)

# %%
# Проверяем joblib round-trip на трёх исходных домах без переобучения.
BASE_URL = 'https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-2-house-prices'
loaded = joblib.load('models/best_model.pkl')
reference = predict_price(demo.iloc[:3],best_model)
np.testing.assert_allclose(predict_price(demo.iloc[:3],loaded),reference,rtol=1e-12)
print('Локальная загрузка: три прогноза совпали')
# Включается только для реальной проверки уже опубликованного файла.
if os.environ.get('HOUSE_PRICES_VERIFY_REMOTE') == '1':
    remote = load_remote_model(BASE_URL)
    np.testing.assert_allclose(predict_price(demo.iloc[:3],remote),reference,rtol=1e-12)
    print('GitHub raw: загрузка выполнена, три прогноза совпали')
else:
    print('GitHub raw не проверен: требуется публикация файлов и HOUSE_PRICES_VERIFY_REMOTE=1')

# %%
# Подставляем измеренные результаты, не используя примерные цифры методички.
m = metrics.loc[best_name]
display(Markdown(f"Лучшая модель по CV: **{best_name}**. "
                 f"Holdout: RMSE(log) **{m.rmse_log:.4f}**, MAE **${m.mae_usd:,.0f}**, "
                 f"RMSE **${m.rmse_usd:,.0f}**, R²(USD) **{m.r2_usd:.4f}**, R²(log) **{m.r2_log:.4f}**. "
                 f"Финальное обучение: **{training.loc[best_name,'fit_seconds']:.3f} с**, "
                 f"поиск: **{training.loc[best_name,'search_seconds']:.2f} с**. "
                 f"Ориентир R²(USD) ≥ 0.90: **{'достигнут' if m.r2_usd >= .90 else 'не достигнут'}**."))
