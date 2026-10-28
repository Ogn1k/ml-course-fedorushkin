"""Проверка структуры отчёта, артефактов и прогноза в свежем процессе."""
from pathlib import Path
import os
import json
import joblib
import nbformat
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
os.chdir(base)
book = nbformat.read('notebook.ipynb',as_version=4)
nbformat.validate(book)
headers = [c.source.splitlines()[0] for c in book.cells if c.cell_type=='markdown' and c.source.startswith('## ')]
assert len(headers)==14, headers
assert all(h.startswith(f'## {i}.') for i,h in enumerate(headers,1))
code = [c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not [(o.get('ename'),o.get('evalue')) for c in code for o in c.outputs if o.output_type=='error']
for name in ['rf_model','xgb_model','lgbm_model','cat_model','tuned_model','best_model','scaler']:
    assert Path('models/'+name+'.pkl').is_file()
assert len(pd.read_csv('examples/optuna_trials.csv'))==30
for name in ['eda_plots','missing_values','feature_importance','roc_curves','shap_summary','shap_waterfall','shap_dependence','shap_bar']:
    assert Path('examples/'+name+'.png').stat().st_size>1000
model = joblib.load('models/best_model.pkl')
x = pd.read_csv('examples/inference_input.csv')
expected = np.load('examples/inference_probabilities.npy')
np.testing.assert_allclose(model.predict_proba(x),expected,atol=1e-10)
assert json.loads(Path('models/feature_cols.json').read_text(encoding='utf-8')) == model['preprocessor'].get_feature_names_out().tolist()
assert 'Revenue' not in model['features'].transform(x).columns
# Незнакомая категория не должна ломать опубликованный pipeline.
unseen = x.copy()
unseen['VisitorType'] = 'Unknown_Visitor'
assert np.isfinite(model.predict_proba(unseen)).all()
assert Path('data/dataset_info.md').is_file()
print('OK: 14 разделов; все ячейки выполнены; 30 trials; модели, графики и прогноз проверены.')
