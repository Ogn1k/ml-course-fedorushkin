"""Предобработка House Prices и инференс; классы доступны при joblib.load."""
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HouseFeatures(TransformerMixin, BaseEstimator):
    """Изучает импутацию только на fit; transform не использует SalePrice."""

    def fit(self, X, y=None):
        # Запоминаем исходную схему и медианы районов только обучающей части.
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.frontage_ = X.groupby('Neighborhood')['LotFrontage'].median()
        self.frontage_default_ = X['LotFrontage'].median()
        return self

    def transform(self, X):
        # Пропущенные входные поля восстанавливаем как NaN, а не как ложные нули.
        d = X.reindex(columns=self.feature_names_in_).copy()
        d['LotFrontage'] = d['LotFrontage'].fillna(
            d['Neighborhood'].map(self.frontage_)).fillna(self.frontage_default_)
        # Указанные NA в описании означают отсутствие сооружения/элемента.
        absent = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in absent:
            d[col] = d[col].fillna('None')
        # Числовые пропуски обнуляем только при подтверждённом отсутствии объекта.
        groups = {'GarageType': ['GarageArea', 'GarageCars'],
                  'BsmtQual': ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                               'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']}
        for indicator, cols in groups.items():
            for col in cols:
                d.loc[d[indicator].eq('None'), col] = d.loc[d[indicator].eq('None'), col].fillna(0)
        # None в MasVnrType является категорией в CSV; потерянные значения — Unknown.
        d['MasVnrType'] = d['MasVnrType'].fillna('Unknown')
        d.loc[d['MasVnrType'].eq('None'), 'MasVnrArea'] = 0
        # Качество кодируем в известном порядке; неизвестные значения оставляем импутеру.
        quality = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
            d[col] = d[col].map(quality)
        # Номинальные числовые коды не задают линейного порядка классов.
        for col in ['MSSubClass', 'MoSold']:
            d[col] = d[col].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
        # Агрегации сохраняют NaN, если часть неизвестна: импутация будет внутри CV.
        d['TotalSF'] = d[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']].sum(axis=1, min_count=3)
        d['TotalBath'] = d['FullBath'] + .5*d['HalfBath'] + d['BsmtFullBath'] + .5*d['BsmtHalfBath']
        porches = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
        d['TotalPorch'] = d[porches].sum(axis=1, min_count=len(porches))
        d['HouseAge'] = (d['YrSold'] - d['YearBuilt']).clip(lower=0)
        d['RemodAge'] = (d['YrSold'] - d['YearRemodAdd']).clip(lower=0)
        # Год гаража не превращаем в фиктивный древний год при отсутствии гаража.
        d['GarageAge'] = (d['YrSold'] - d['GarageYrBlt']).clip(lower=0)
        d.loc[d['GarageType'].eq('None'), 'GarageAge'] = 0
        d = d.drop(columns='GarageYrBlt')
        for name, col in {'HasPool':'PoolArea', 'HasGarage':'GarageArea',
                          'Has2ndFloor':'2ndFlrSF', 'HasFireplace':'Fireplaces',
                          'HasBsmt':'TotalBsmtSF'}.items():
            d[name] = np.where(d[col].isna(), np.nan, (d[col] > 0).astype(float))
        d['OverallQual_TotalSF'] = d['OverallQual'] * d['TotalSF']
        return d


def make_pipeline(model):
    # Каждый fit в CV заново обучает импутацию, категории и масштабирование.
    num = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                    ('scale', StandardScaler())])
    cat = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])
    prep = ColumnTransformer([('num', num, make_column_selector(dtype_include=np.number)),
                              ('cat', cat, make_column_selector(dtype_exclude=np.number))],
                             verbose_feature_names_out=False)
    return Pipeline([('features', HouseFeatures()), ('preprocess', prep), ('model', model)])


def predict_price(house_data, model):
    """Принимает исходные признаки: dict → float, DataFrame/list → ndarray ($)."""
    single = isinstance(house_data, dict)
    # В модель не передаём Id или целевую; схема восстанавливается трансформером.
    frame = pd.DataFrame([house_data]) if single else pd.DataFrame(house_data)
    if frame.empty:
        raise ValueError('Нужен хотя бы один дом')
    required = {'OverallQual', 'GrLivArea', 'YearBuilt', 'Neighborhood'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Укажите OverallQual, GrLivArea, YearBuilt и Neighborhood')
    # Отрицательные площади и нечисловые входные значения не допускаем.
    for col in ['OverallQual', 'GrLivArea', 'YearBuilt']:
        values = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f'Некорректное значение {col}')
        frame[col] = values
    if not frame['OverallQual'].between(1, 10).all():
        raise ValueError('OverallQual должен быть от 1 до 10')
    prices = np.expm1(model.predict(frame))
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Модель выдала недопустимую цену')
    return float(prices[0]) if single else prices


def load_remote_model(base_url):
    """Загружает доверенную собственную модель; HTTP-ошибки не маскируются."""
    # Для joblib должен быть доступен этот модуль housing_pipeline.py.
    response = requests.get(f'{base_url}/models/best_model.pkl', timeout=30)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))
