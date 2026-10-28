"""Детерминированные признаки сессии; целевая переменная не используется."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SessionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True, drop_page_values=False):
        self.enabled = enabled
        self.drop_page_values = drop_page_values

    def fit(self, X, y=None):
        # Преобразование не оценивает статистики по выборке.
        return self

    def transform(self, X):
        d = X.copy()
        if self.enabled:
            # Суммы и отношения описывают глубину и интенсивность сессии.
            d['TotalPages'] = d[['Administrative','Informational','ProductRelated']].sum(axis=1)
            d['TotalDuration'] = d[['Administrative_Duration','Informational_Duration','ProductRelated_Duration']].sum(axis=1)
            d['AvgPageValue'] = d['PageValues'] / (d['TotalPages'] + 1)
            d['BounceExitRatio'] = d['BounceRates'] / (d['ExitRates'] + 0.001)
            # Бинарные индикаторы и ручное отображение учитывают June (не Jun).
            d['IsNewVisitor'] = (d['VisitorType'] == 'New_Visitor').astype(int)
            d['IsWeekend'] = d['Weekend'].astype(int)
            months = dict(zip(['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], range(1,13)))
            d['MonthNum'] = d['Month'].map(months)
            d['IsHighTraffic'] = (d['TrafficType'] > 10).astype(int)
            # Дополнительные признаки не требуют полной даты сессии.
            d['DurationPerPage'] = d['TotalDuration'] / (d['TotalPages'] + 1)
            d['MonthSin'] = np.sin(2*np.pi*d['MonthNum']/12)
            d['MonthCos'] = np.cos(2*np.pi*d['MonthNum']/12)
        if self.drop_page_values:
            # Удаляем и прямое поле, и его производное для проверки чувствительности.
            d = d.drop(columns=['PageValues','AvgPageValue'], errors='ignore')
        return d
