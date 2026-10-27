"""Предобработка Telco Churn и загрузка сохранённого полного pipeline."""
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class ChurnPreprocessor(TransformerMixin, BaseEstimator):
    """Fit медианы и LabelEncoder только на train; новые категории получают -1."""

    def _engineer(self, X):
        # ID и целевая переменная никогда не становятся признаками.
        frame = X.drop(columns=['customerID', 'Churn'], errors='ignore').copy()
        frame['TotalCharges'] = pd.to_numeric(frame['TotalCharges'], errors='coerce')
        # Используем медиану из fit; статистики новых клиентов не учитываем.
        frame['TotalCharges'] = frame['TotalCharges'].fillna(self.total_median_)
        frame['TenureYears'] = frame['tenure'] / 12
        frame['AvgMonthlyCharge'] = frame['TotalCharges'] / (frame['tenure'] + 1)
        frame['IsNewClient'] = (frame['tenure'] <= 6).astype(int)
        # Считаем 9 разных услуг: интернет считается один раз.
        services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        frame['HasMultipleServices'] = frame[services].eq('Yes').sum(axis=1)
        frame['HasMultipleServices'] += frame['InternetService'].ne('No').astype(int)
        return frame

    def fit(self, X, y=None):
        # Сохраняем схему исходных данных и единственную обучаемую импутацию.
        self.raw_columns_ = [c for c in X if c not in ['customerID', 'Churn']]
        self.total_median_ = float(pd.to_numeric(X['TotalCharges'], errors='coerce').median())
        frame = self._engineer(X)
        # LabelEncoder применяем отдельно к каждому категориальному столбцу.
        self.encoders_ = {c: LabelEncoder().fit(frame[c].fillna('__missing__').astype(str))
                          for c in frame.select_dtypes(include=['object', 'string'])}
        self.feature_names_out_ = list(frame.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, 'encoders_')
        # Проверка схемы предотвращает незаметное предсказание на неполных данных.
        missing = set(self.raw_columns_) - set(X.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        frame = self._engineer(X[self.raw_columns_])
        for column, encoder in self.encoders_.items():
            # Неизвестные категории обрабатываем без переобучения кодировщика.
            mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
            frame[column] = frame[column].fillna('__missing__').astype(str).map(mapping).fillna(-1)
        return frame[self.feature_names_out_].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def predict_churn(clients, model, threshold=0.5):
    """Принимает исходные 19 признаков: DataFrame, словарь или список словарей."""
    frame = pd.DataFrame([clients] if isinstance(clients, dict) else clients)
    # Полный pipeline сам выполняет импутацию и кодирование.
    probability = model.predict_proba(frame)[:, 1]
    return pd.DataFrame({'churn_score': probability, 'prediction': (probability >= threshold).astype(int)})


def load_remote_model(base_url):
    """Загружать joblib следует только из доверенного собственного репозитория."""
    response = requests.get(f'{base_url.rstrip("/")}/models/best_model.pkl', timeout=60)
    response.raise_for_status()
    # Модуль churn_pipeline должен быть доступен для десериализации класса.
    return joblib.load(BytesIO(response.content))
