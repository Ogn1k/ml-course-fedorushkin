"""Проверка опубликованных артефактов: python verify_remote.py."""
import io
import sys
import tempfile
from pathlib import Path
import importlib.util
import joblib
import numpy as np
import pandas as pd
import requests

BASE_URL = 'https://raw.githubusercontent.com/Ogn1k/ml-course-fedorushkin/main/module-4-features'
def download(path):
    # HTTP-ошибки, включая отсутствие опубликованного файла, не скрываем.
    response = requests.get(f'{BASE_URL}/{path}', timeout=90)
    response.raise_for_status()
    return response.content

def main():
    # Трансформер берём из той же доверенной ревизии, что и pipeline.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'feature_engineering.py'
        path.write_bytes(download('feature_engineering.py'))
        spec = importlib.util.spec_from_file_location('feature_engineering', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['feature_engineering'] = module
        spec.loader.exec_module(module)
        model = joblib.load(io.BytesIO(download('models/best_model.pkl')))
        import json
        features = json.loads(download('models/feature_cols.json'))
        inputs = pd.read_csv(io.BytesIO(download('examples/inference_input.csv')))
        expected = np.load(io.BytesIO(download('examples/inference_probabilities.npy')))
        np.testing.assert_allclose(model.predict_proba(inputs), expected, atol=1e-10)
        assert features == model['preprocessor'].get_feature_names_out().tolist()
        print('Raw GitHub: вероятности 3 сессий совпадают; признаков:', len(features))

if __name__ == '__main__':
    main()
