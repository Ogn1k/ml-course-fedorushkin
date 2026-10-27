"""Проверяет выполненный отчёт и сохранённый инференс в новом процессе."""
from pathlib import Path
import json
import hashlib
import joblib
import nbformat
import numpy as np
import pandas as pd
from housing_pipeline import predict_price


def main():
    # Ноутбук должен содержать 14 разделов и выполненные ячейки без ошибок.
    nb = nbformat.read('notebook.ipynb', as_version=4)
    nbformat.validate(nb)
    sections = [c.source for c in nb.cells if c.cell_type == 'markdown' and c.source.startswith('## ')]
    assert len(sections) == 14, len(sections)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            assert not any(o.output_type == 'error' for o in cell.outputs)
    meta = json.loads(Path('models/metadata.json').read_text(encoding='utf-8'))
    # Проверяем, что holdout не попадал в обучение и данные не изменились.
    assert set(meta['train_ids']).isdisjoint(meta['holdout_ids'])
    assert len(meta['holdout_ids']) == 292
    for name, checksum in meta['sha256'].items():
        assert hashlib.sha256((Path('data') / name).read_bytes()).hexdigest() == checksum
    read_opts = dict(keep_default_na=False, na_values=['NA', ''])
    train = pd.read_csv('data/train.csv', **read_opts)
    test = pd.read_csv('data/test.csv', **read_opts)
    model = joblib.load('models/best_model.pkl')
    # Каждый пайплайн можно загрузить независимо от notebook.
    for name in ['lr', 'ridge', 'lasso', 'enet']:
        loaded = joblib.load(f'models/{name}_model.pkl')
        assert np.isfinite(predict_price(test.head(5), loaded)).all()
    raw = train.loc[train.Id.isin(meta['holdout_ids'])].head(5)
    batch = predict_price(raw, model)
    single = np.array([predict_price(row.to_dict(), model) for _, row in raw.iterrows()])
    np.testing.assert_allclose(batch, single, rtol=1e-10)
    # Неполный запрос и новая категория должны проходить без повторного fit.
    sample = {'OverallQual': 6, 'GrLivArea': 1500, 'YearBuilt': 2000, 'Neighborhood': 'UnknownArea'}
    assert predict_price(sample, model) > 0
    for invalid in [{}, dict(sample, OverallQual=11), dict(sample, GrLivArea=-1)]:
        try:
            predict_price(invalid, model)
        except ValueError:
            pass
        else:
            raise AssertionError('Некорректный запрос должен отклоняться')
    # transform на другой выборке не изменяет изученные медианы районов.
    features = model.named_steps['features']
    before = features.frontage_.copy()
    features.transform(test)
    pd.testing.assert_series_equal(features.frontage_, before)
    expected = train.loc[train.Id.isin(meta['train_ids'])].groupby('Neighborhood').LotFrontage.median()
    pd.testing.assert_series_equal(features.frontage_, expected)
    names = model.named_steps['preprocess'].get_feature_names_out().tolist()
    assert names == json.loads(Path('models/feature_cols.json').read_text(encoding='utf-8'))
    assert 'Id' not in names and 'SalePrice' not in names
    # Файл для Kaggle должен сохранять все исходные ID и сами предсказания.
    submission = pd.read_csv('submission.csv')
    assert submission.Id.tolist() == test.Id.tolist()
    np.testing.assert_allclose(submission.SalePrice, predict_price(test, model), rtol=1e-10)
    print('PASS: notebook, 4 models, split, imputation, inference, schema, submission')


if __name__ == '__main__':
    main()
