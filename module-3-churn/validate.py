"""Независимая проверка сохранённых артефактов без повторного обучения."""
import hashlib
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from churn_pipeline import predict_churn

ROOT = Path(__file__).resolve().parent


def main():
    metadata = json.loads((ROOT / 'models/metadata.json').read_text(encoding='utf-8'))
    metrics = json.loads((ROOT / 'models/metrics.json').read_text(encoding='utf-8'))
    df = pd.read_csv(ROOT / 'data/telco_churn.csv')
    assert hashlib.sha256((ROOT / 'data/telco_churn.csv').read_bytes()).hexdigest() == metadata['data_sha256']
    # Независимо восстанавливаем train/test, включая порядок строк.
    X = df.drop(columns=['Churn', 'customerID'])
    y = df.Churn.map({'No': 0, 'Yes': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    assert X_train.index.tolist() == metadata['train_indices']
    assert X_test.index.tolist() == metadata['test_indices']
    assert set(X_train.index).isdisjoint(X_test.index)
    # Пересчитываем каждую сохранённую метрику по загруженным pipeline.
    files = {'Random Forest': 'rf_model.pkl', 'XGBoost': 'xgb_model.pkl', 'LightGBM': 'lgbm_model.pkl',
             'CatBoost': 'cat_model.pkl', 'LightGBM (Optuna)': 'tuned_lgbm_model.pkl'}
    for name, file in files.items():
        model = joblib.load(ROOT / 'models' / file)
        pre = model.named_steps['preprocess']
        assert np.isclose(pre.total_median_, pd.to_numeric(X_train.TotalCharges, errors='coerce').median())
        assert len(pre.get_feature_names_out()) == 23
        assert not {'customerID', 'Churn'} & set(pre.get_feature_names_out())
        probability = model.predict_proba(X_test)[:, 1]
        predicted = (probability >= .5).astype(int)
        values = {'Accuracy': accuracy_score(y_test, predicted), 'Precision': precision_score(y_test, predicted),
                  'Recall': recall_score(y_test, predicted), 'F1': f1_score(y_test, predicted),
                  'ROC-AUC': roc_auc_score(y_test, probability), 'PR-AUC': average_precision_score(y_test, probability)}
        for key, value in values.items():
            assert np.isclose(value, metrics['test'][name][key], atol=1e-12), (name, key)
    best = joblib.load(ROOT / 'models/best_model.pkl')
    assert metrics['optuna_trials'] == 30
    expected = max(['LightGBM', 'LightGBM (Optuna)'], key=lambda n: metrics['cv'][n]['mean'])
    assert metrics['best_model'] == expected
    # Проверяем новые категории, текстовые пропуски и неизменность входных данных.
    clients = X_test.iloc[:3].copy()
    clients.loc[clients.index[0], 'Contract'] = 'Never seen contract'
    clients.loc[clients.index[1], 'TotalCharges'] = ' '
    before = clients.copy(deep=True)
    output = predict_churn(clients, best)
    pd.testing.assert_frame_equal(clients, before)
    assert output.churn_score.between(0, 1).all()
    assert best.named_steps['preprocess'].transform(clients).iloc[0]['Contract'] == -1
    assert len(predict_churn(X_test.iloc[0].to_dict(), best)) == 1
    # Порядок исходных колонок не должен менять результат pipeline.
    assert np.allclose(best.predict_proba(X_test), best.predict_proba(X_test[X_test.columns[::-1]]))
    try:
        predict_churn(clients.drop(columns='tenure'), best)
        raise AssertionError('Schema validation did not run')
    except ValueError as error:
        assert 'Missing required columns' in str(error)
    # SHAP действительно объясняет сырые выходы сохранённой модели.
    explanation = np.load(ROOT / 'examples/shap_values.npz')
    assert np.array_equal(explanation['test_indices'], X_test.index)
    assert np.allclose(explanation['base_values'] + explanation['values'].sum(axis=1), explanation['raw_score'])
    assert np.allclose(expit(explanation['raw_score']), best.predict_proba(X_test)[:, 1])
    # Проверка структуры отчёта и отсутствия невыполненных/упавших ячеек.
    notebook = nbformat.read(ROOT / 'notebook.ipynb', as_version=4)
    nbformat.validate(notebook)
    sections = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == 'markdown' and c.source.startswith('## ')]
    assert len(sections) == 14, sections
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            assert all(o.output_type != 'error' for o in cell.outputs)
    assert len(list((ROOT / 'examples').glob('*.png'))) >= 10
    print('PASS: dataset, split, five models, metrics, preprocessing, schema, SHAP, notebook (14 sections).')
    print('GitHub raw verified:', metadata['remote_verified'])


if __name__ == '__main__':
    main()
