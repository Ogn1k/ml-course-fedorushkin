# Полный код вычислительных ячеек ноутбука

# Импортируем библиотеки и фиксируем параметры воспроизводимости.
import io, json, time, hashlib, platform, importlib.metadata, zipfile
from pathlib import Path
import requests, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, Image
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, RocCurveDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna, shap
SEED = 42
BASE = Path('.')
for folder in ['data', 'models', 'examples']:
    (BASE / folder).mkdir(exist_ok=True)

# Скачиваем только официальный архив; при повторном запуске используем локальный CSV.
DATA_URL = 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'
csv_path = BASE / 'data/online_shoppers_intention.csv'
if not csv_path.exists():
    response = requests.get(DATA_URL, timeout=90)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_path.write_bytes(archive.read('online_shoppers_intention.csv'))
df = pd.read_csv(csv_path)
assert df.shape == (12330, 18)
print(df.shape)
print(df.head().to_string())
df.info()
display(df.describe(include='all').T)
display(df['Revenue'].value_counts().to_frame('count').assign(share=df['Revenue'].value_counts(normalize=True)))
# Сохраняем контрольную сумму для проверки источника данных.
provenance = dict(url=DATA_URL, doi='10.24432/C5F88Q', license='CC BY 4.0',
                  sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
Path('data/provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')

# Удаляем дубликаты до разделения; не используем целевую для генерации признаков.
print('Пропуски:', int(df.isna().sum().sum()), 'Дубликаты:', int(df.duplicated().sum()))
clean = df.drop_duplicates().copy()
X = clean.drop(columns='Revenue')
y = clean['Revenue'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
pd.DataFrame({'split': ['train', 'test'], 'rows': [len(y_train),len(y_test)],
              'positive_share': [y_train.mean(), y_test.mean()]}).to_csv('data/splits.csv', index=False)

# Строим шесть графиков EDA и отдельный график пропусков.
eda = X_train.assign(Revenue=y_train)
fig, axes = plt.subplots(2, 3, figsize=(18,10))
sns.countplot(data=eda, x='Revenue', ax=axes[0,0])
sns.histplot(data=eda.assign(LogDuration=np.log1p(eda['ProductRelated_Duration'])), x='LogDuration', hue='Revenue', bins=40, ax=axes[0,1])
sns.boxplot(data=eda, x='Revenue', y='PageValues', ax=axes[0,2])
sns.heatmap(eda.drop(columns=['OperatingSystems','Browser','Region','TrafficType']).select_dtypes(include='number').corr(), cmap='coolwarm', ax=axes[1,0])
eda.groupby('VisitorType')['Revenue'].mean().plot.bar(ax=axes[1,1], ylabel='Purchase share')
eda.groupby('Month')['Revenue'].mean().reindex(['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec']).plot.bar(ax=axes[1,2], ylabel='Purchase share')
for ax, title in zip(axes.flat, ['Target balance','Product duration: log(1 + seconds)','Page value by target',
                              'Numeric correlations','Conversion by visitor','Conversion by month']):
    ax.set_title(title)
fig.tight_layout()
fig.savefig('examples/eda_plots.png', dpi=140)
plt.close(fig)
display(Image(filename='examples/eda_plots.png'))
fig, ax = plt.subplots(figsize=(10,3))
X_train.isna().sum().plot.bar(ax=ax, title='Missing values in training data', ylabel='Missing count')
fig.tight_layout()
fig.savefig('examples/missing_values.png', dpi=120)
plt.close(fig)
display(Image(filename='examples/missing_values.png'))

# Импортируем собственный трансформер из файла рядом с ноутбуком.
from feature_engineering import SessionFeatures
CATEGORICAL = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
def make_pipeline(model, engineered=True, drop_page_values=False):
    feature_step = SessionFeatures(engineered, drop_page_values)
    sample = feature_step.fit_transform(X_train.iloc[:2])
    numeric = [c for c in sample.columns if c not in CATEGORICAL]
    # Все обучаемые преобразования находятся внутри Pipeline, в том числе в CV.
    prep = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)
    ], verbose_feature_names_out=False)
    return Pipeline([('features', feature_step), ('preprocessor', prep), ('model', model)])
display(SessionFeatures().fit_transform(X_train).head())
assert np.isfinite(SessionFeatures().fit_transform(X_train).select_dtypes(include='number')).all().all()

# Фиксируем CV и ограничиваем число потоков для воспроизводимого локального запуска.
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
ratio = float((y_train == 0).sum() / (y_train == 1).sum())
def estimator(name, params=None):
    p = params or {}
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=p.get('n_estimators',200), max_depth=p.get('max_depth',10),
            min_samples_leaf=p.get('min_samples_leaf',1), max_features=p.get('max_features','sqrt'),
            class_weight='balanced', random_state=SEED, n_jobs=4)
    if name == 'XGBoost':
        return XGBClassifier(**({'n_estimators':200,'max_depth':6,'learning_rate':0.1,
            'subsample':0.9,'colsample_bytree':0.9} | p), scale_pos_weight=ratio,
            random_state=SEED, eval_metric='logloss', n_jobs=4)
    if name == 'LightGBM':
        return LGBMClassifier(**({'n_estimators':200,'num_leaves':31,'max_depth':7,'learning_rate':0.05} | p),
            class_weight='balanced', random_state=SEED, verbosity=-1, n_jobs=4, subsample_freq=1)
    return CatBoostClassifier(**({'iterations':200,'depth':6,'learning_rate':0.1} | p),
        auto_class_weights='Balanced', random_seed=SEED, verbose=False, thread_count=4, allow_writing_files=False)

# Сначала оцениваем модели на обучающей CV; затем обучаем на полном train.
names = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
models, cv_scores, fit_times = {}, {}, {}
for name in names:
    pipe = make_pipeline(estimator(name))
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    cv_scores[name] = dict(mean=float(scores.mean()), std=float(scores.std()), folds=scores.tolist())
    start = time.perf_counter()
    models[name] = pipe.fit(X_train, y_train)
    fit_times[name] = time.perf_counter()-start
    print(name, 'CV:', cv_scores[name], 'Fit seconds:', round(fit_times[name],3), flush=True)
winner = max(names, key=lambda n: cv_scores[n]['mean'])
display(Markdown(f'Семейство для тюнинга по CV: **{winner}**. Время выше относится к финальному fit, без CV.'))

# Единая функция вычисляет пороговые и вероятностные метрики.
def evaluate(pipe, name):
    prob = pipe.predict_proba(X_test)[:,1]
    pred = (prob >= 0.5).astype(int)
    return dict(Model=name, Accuracy=accuracy_score(y_test,pred), Precision=precision_score(y_test,pred,zero_division=0),
        Recall=recall_score(y_test,pred), F1=f1_score(y_test,pred), ROC_AUC=roc_auc_score(y_test,prob),
        PR_AUC=average_precision_score(y_test,prob))
results = [evaluate(models[n],n) | {'CV_ROC_AUC':cv_scores[n]['mean'], 'Fit_seconds':fit_times[n]} for n in names]
results_df = pd.DataFrame(results)
display(results_df)
print('Константный baseline Accuracy:', float((y_test==0).mean()), 'ROC-AUC: 0.5; AP:', y_test.mean())
results_df.to_csv('examples/metrics_baseline.csv', index=False)

# Строим общую ROC и матрицу ошибок выбранного по CV baseline.
fig, axes = plt.subplots(1,2,figsize=(13,5))
for name, pipe in models.items():
    RocCurveDisplay.from_predictions(y_test,pipe.predict_proba(X_test)[:,1],name=name,ax=axes[0])
sns.heatmap(confusion_matrix(y_test,models[winner].predict(X_test)),annot=True,fmt='d',ax=axes[1],cmap='Blues')
axes[1].set(xlabel='Predicted',ylabel='True',title=winner+' (selected by CV)')
fig.tight_layout()
fig.savefig('examples/roc_curves.png',dpi=140)
plt.close(fig)
display(Image(filename='examples/roc_curves.png'))

# Сохраняем имена после One-Hot и строим важности для двух семейств.
fig, axes = plt.subplots(1,2,figsize=(15,7))
for name, ax in zip(['Random Forest','LightGBM'],axes):
    pipe = models[name]
    cols = pipe['preprocessor'].get_feature_names_out()
    imp = pd.Series(pipe['model'].feature_importances_,index=cols).sort_values(ascending=False)
    imp.to_csv('examples/'+name.lower().replace(' ','_')+'_importance.csv',header=['importance'])
    imp.head(15).sort_values().plot.barh(ax=ax,title=name)
fig.tight_layout()
fig.savefig('examples/feature_importance.png',dpi=140)
plt.close(fig)
display(Image(filename='examples/feature_importance.png'))

# Абляция оценивает изменения только по обучающей CV.
ablation = []
for label, engineered, drop in [('Original',False,False),('Engineered',True,False),('Without PageValues',True,True)]:
    pipe = make_pipeline(estimator('LightGBM'),engineered,drop)
    scores = cross_val_score(pipe,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1)
    ablation.append(dict(Features=label,CV_ROC_AUC=float(scores.mean()),CV_std=float(scores.std())))
ablation_df = pd.DataFrame(ablation)
display(ablation_df)
ablation_df.to_csv('examples/feature_ablation.csv',index=False)
# Фактические доли помогают избежать неподтверждённых утверждений о новых посетителях.
display(eda.groupby('VisitorType')['Revenue'].agg(['mean','count']))
display(eda.groupby('Month')['Revenue'].agg(['mean','count']))
gain = ablation[1]['CV_ROC_AUC']-ablation[0]['CV_ROC_AUC']
display(Markdown(f'Изменение CV ROC-AUC после инженерии: **{gain:+.5f}**. '
    'Добавление признаков не гарантирует улучшения. Различия по месяцам и типам посетителя '
    'помогают сформулировать гипотезы для A/B-теста, но не доказывают эффект рекламы. '
    'Сильное падение без PageValues означало бы зависимость качества от доступности этого поля.'))

# Пространство поиска соответствует выбранному семейству модели.
def objective(trial):
    if winner == 'Random Forest':
        params = dict(n_estimators=trial.suggest_int('n_estimators',100,400,step=50),
            max_depth=trial.suggest_int('max_depth',5,18),min_samples_leaf=trial.suggest_int('min_samples_leaf',1,12),
            max_features=trial.suggest_categorical('max_features',['sqrt',0.5,0.8]))
    elif winner == 'CatBoost':
        params = dict(iterations=trial.suggest_int('iterations',100,500,step=50),
            depth=trial.suggest_int('depth',3,8),learning_rate=trial.suggest_float('learning_rate',0.01,0.2,log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg',0.1,10,log=True))
    else:
        params = dict(n_estimators=trial.suggest_int('n_estimators',100,500,step=50),
            max_depth=trial.suggest_int('max_depth',3,10), learning_rate=trial.suggest_float('learning_rate',0.01,0.2,log=True),
            subsample=trial.suggest_float('subsample',0.6,1),colsample_bytree=trial.suggest_float('colsample_bytree',0.6,1),
            reg_alpha=trial.suggest_float('reg_alpha',0.001,10,log=True),reg_lambda=trial.suggest_float('reg_lambda',0.001,10,log=True))
        if winner == 'LightGBM':
            params['num_leaves'] = trial.suggest_int('num_leaves',10,60)
            params['min_child_samples'] = trial.suggest_int('min_child_samples',10,50)
    pipe = make_pipeline(estimator(winner,params))
    return float(cross_val_score(pipe,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1).mean())

# TPE с фиксированным seed; сохраняем журнал всех испытаний.
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective,n_trials=30)
study.trials_dataframe().to_csv('examples/optuna_trials.csv',index=False)
print('Best parameters:',study.best_params,'CV ROC-AUC:',study.best_value)
tuned = make_pipeline(estimator(winner,study.best_params))
start = time.perf_counter()
tuned.fit(X_train,y_train)
tuned_time = time.perf_counter()-start
final_name = winner+' (Optuna)'
tuned_result = evaluate(tuned,final_name) | {'CV_ROC_AUC':study.best_value,'Fit_seconds':tuned_time}
# Сохраняем baseline, если тюнинг не улучшил CV; test на решение не влияет.
use_tuned = study.best_value > cv_scores[winner]['mean']
best = tuned if use_tuned else models[winner]
best_name = final_name if use_tuned else winner
results.append(tuned_result)
display(pd.DataFrame(results))
print('Test delta ROC-AUC:',tuned_result['ROC_AUC']-next(r['ROC_AUC'] for r in results if r['Model']==winner))
print('Сохранённая модель по CV:',best_name)

# Преобразуем test обученными шагами Pipeline; выбираем до 500 объектов.
transformed = best['preprocessor'].transform(best['features'].transform(X_test))
feature_cols = best['preprocessor'].get_feature_names_out().tolist()
z = pd.DataFrame(transformed,columns=feature_cols,index=X_test.index)
sample = z.sample(min(500,len(z)),random_state=SEED)
explainer = shap.TreeExplainer(best['model'])
explanation = explainer(sample)
if explanation.values.ndim == 3:
    explanation = explanation[:,:,1]
values = explanation.values
np.savez_compressed('examples/shap_values.npz',values=values,base_values=explanation.base_values,index=sample.index)

# Сохраняем глобальные и локальные объяснения.
shap.summary_plot(values,sample,show=False)
plt.tight_layout()
plt.savefig('examples/shap_summary.png',dpi=140,bbox_inches='tight')
plt.close('all')
shap.summary_plot(values,sample,plot_type='bar',show=False)
plt.savefig('examples/shap_bar.png',dpi=140,bbox_inches='tight')
plt.close('all')
local_idx = int(np.argmax(best['model'].predict_proba(sample)[:,1]))
shap.plots.waterfall(explanation[local_idx],show=False)
plt.savefig('examples/shap_waterfall.png',dpi=140,bbox_inches='tight')
plt.close('all')
shap_importance = pd.Series(np.abs(values).mean(axis=0),index=feature_cols).sort_values(ascending=False)
shap_importance.to_csv('examples/shap_importance.csv',header=['mean_abs_shap'])
top_feature = shap_importance.index[0]
shap.dependence_plot(top_feature,values,sample,interaction_index=None,show=False)
plt.savefig('examples/shap_dependence.png',dpi=140,bbox_inches='tight')
plt.close('all')
for file in ['shap_summary','shap_bar','shap_waterfall','shap_dependence']:
    display(Image(filename='examples/'+file+'.png'))
# Проверяем аддитивность: RF объясняет вероятность, бустинги — log-odds.
raw = values.sum(axis=1)+np.asarray(explanation.base_values)
shap_units = 'probability' if isinstance(best['model'], RandomForestClassifier) else 'log-odds'
prob_from_shap = raw if shap_units == 'probability' else 1/(1+np.exp(-raw))
print('SHAP units:', shap_units)
np.testing.assert_allclose(prob_from_shap,best['model'].predict_proba(sample)[:,1],atol=1e-5)
local = pd.Series(values[local_idx],index=feature_cols).sort_values(key=abs,ascending=False).head(5)
display(Markdown('Ведущий признак по mean(|SHAP|): **'+top_feature+'**. Локальные вклады ('+shap_units+'):'))
display(local.to_frame('SHAP'))
print('Вероятность выбранной сессии:',best['model'].predict_proba(sample.iloc[[local_idx]])[0,1])

# Сохраняем все baseline и итоговую модель с одним интерфейсом сырых данных.
files = dict(zip(names,['rf_model','xgb_model','lgbm_model','cat_model']))
for name, pipe in models.items():
    joblib.dump(pipe,'models/'+files[name]+'.pkl',compress=3)
joblib.dump(tuned,'models/tuned_model.pkl',compress=3)
joblib.dump(best,'models/best_model.pkl',compress=3)
joblib.dump(best['preprocessor'].named_transformers_['num']['scaler'],'models/scaler.pkl')
Path('models/feature_cols.json').write_text(json.dumps(feature_cols,ensure_ascii=False,indent=2),encoding='utf-8')
Path('models/metrics.json').write_text(json.dumps({'results':results,'selected_model':best_name,
    'selection':'highest train CV ROC-AUC; test excluded','ablation':ablation},ensure_ascii=False,indent=2),encoding='utf-8')
# Версии и схема входа позволяют воспроизвести загрузку.
packages = ['pandas','numpy','scikit-learn','xgboost','lightgbm','catboost','optuna','shap',
            'matplotlib','seaborn','joblib','requests','nbformat','nbclient','ipykernel','tabulate']
versions = {p:importlib.metadata.version(p) for p in packages}
metadata = dict(author='Федорушкин Станислав',group='ПКТб-23-1',date='2026-10-29',
    seed=SEED,python=platform.python_version(),versions=versions,selected_model=best_name,
    selected_cv_roc_auc=max(study.best_value,cv_scores[winner]['mean']),best_trial_params=study.best_params,
    input_columns=X.columns.tolist(),input_dtypes=X.dtypes.astype(str).to_dict(),
    train_rows=len(X_train),test_rows=len(X_test),threshold=0.5,data=provenance)
Path('models/metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
Path('requirements.txt').write_text(''.join(p+'=='+v+'\n' for p,v in versions.items()),encoding='utf-8')
X_test.iloc[:3].to_csv('examples/inference_input.csv',index=False)
np.save('examples/inference_probabilities.npy',best.predict_proba(X_test.iloc[:3]))
loaded = joblib.load('models/best_model.pkl')
np.testing.assert_allclose(loaded.predict_proba(X_test.iloc[:3]),best.predict_proba(X_test.iloc[:3]))
print('Локальный round-trip: вероятности 3 сессий совпадают.')

# Показываем исполняемый код удалённой проверки без подмены локальным fallback.
print(Path('verify_remote.py').read_text(encoding='utf-8'))

# Генерируем итоговые выводы по реальным измерениям.
chosen = next(r for r in results if r['Model']==best_name)
display(Markdown(f"Итог: **{best_name}**, test ROC-AUC **{chosen['ROC_AUC']:.4f}**, "
    f"F1 **{chosen['F1']:.4f}**, Precision **{chosen['Precision']:.4f}**, Recall **{chosen['Recall']:.4f}**. "
    f"Изменение CV от инженерии признаков: **{gain:+.5f}**. "
    'Выбор сделан по train CV; независимый test использован для отчёта.'))

# Выводим версии и полный код приложения для автономного чтения отчёта.
print(Path('requirements.txt').read_text(encoding='utf-8'))
display(Markdown('```python\n'+Path('appendix_full_code.py').read_text(encoding='utf-8')+'\n```'))