"""Проверить GitHub raw на трёх клиентах без повторного обучения."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from churn_pipeline import load_remote_model

if __name__ == '__main__':
    root = Path(__file__).parent
    base = 'https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-3-churn'
    metadata = json.loads((root / 'models/metadata.json').read_text(encoding='utf-8'))
    clients = pd.read_csv(root / 'data/telco_churn.csv').iloc[metadata['test_indices'][:3]]
    local = joblib.load(root / 'models/best_model.pkl')
    remote = load_remote_model(base)
    response = requests.get(f'{base}/models/feature_cols.json', timeout=60)
    response.raise_for_status()
    assert response.json() == metadata['engineered_features']
    assert np.array_equal(local.predict(clients), remote.predict(clients))
    assert np.allclose(local.predict_proba(clients), remote.predict_proba(clients))
    metadata['remote_verified'] = True
    (root / 'models/metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: GitHub raw, feature names, predictions and scores match on three clients.')
