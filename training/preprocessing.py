"""
preprocessing.py  —  Shared preprocessing transformers
Giữ trong module riêng để joblib pickle/unpickle đúng cách.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Clip mỗi feature về [lower_pct, upper_pct] percentile tính trên training data."""
    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanpercentile(arr, self.lower_pct, axis=0)
        self.upper_bounds_ = np.nanpercentile(arr, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float).copy(), self.lower_bounds_, self.upper_bounds_)


class PreprocessorPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline cột-aware: Winsorize [p1,p99] + RobustScaler (median/IQR).
    Lưu scale_columns_ để transform() không bao giờ bị shape mismatch.

    Training:
        pp = PreprocessorPipeline()
        X_train = pp.fit_transform_df(X_train, scale_columns)
        X_test  = pp.transform_df(X_test)
        joblib.dump(pp, 'scaler.pkl')

    Inference:
        pp = joblib.load('scaler.pkl')
        X = pp.transform_df(X_new)
    """
    def __init__(self):
        self.scale_columns_ = []
        self._inner = None

    def fit(self, X_arr, scale_columns, y=None):
        self.scale_columns_ = list(scale_columns)
        self._inner = Pipeline([
            ("winsorize", WinsorizerTransformer(1.0, 99.0)),
            ("scale",     RobustScaler()),
        ])
        self._inner.fit(X_arr)
        return self

    def transform(self, X_arr):
        if self._inner is None:
            raise RuntimeError("PreprocessorPipeline chua duoc fit.")
        return self._inner.transform(X_arr)

    def fit_transform_df(self, df, scale_columns):
        df = df.copy()
        self.fit(df[scale_columns].values, scale_columns)
        df[scale_columns] = self.transform(df[scale_columns].values)
        return df

    def transform_df(self, df, fill_value=0.0):
        df = df.copy()
        for c in self.scale_columns_:
            if c not in df.columns:
                df[c] = fill_value
        df[self.scale_columns_] = self.transform(df[self.scale_columns_].values)
        return df