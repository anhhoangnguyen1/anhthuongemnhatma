"""
Shared preprocessing transformers used in both training and inference.
Keeping this in a standalone module ensures joblib can pickle/unpickle
these classes regardless of how train_xgboost_dss.py is executed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Clip each feature to [lower_pct, upper_pct] percentile fitted on training data."""

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_: np.ndarray = np.nanpercentile(arr, self.lower_pct, axis=0)
        self.upper_bounds_: np.ndarray = np.nanpercentile(arr, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


class PreprocessorPipeline(BaseEstimator, TransformerMixin):
    """
    Column-aware preprocessing pipeline: winsorize + RobustScale.

    Stores scale_columns_ so transform() always uses the exact same
    columns as training — no shape mismatch regardless of input columns.

    Usage (training):
        pp = PreprocessorPipeline()
        X_train = pp.fit_transform_df(X_train, scale_columns)
        X_test  = pp.transform_df(X_test)
        joblib.dump(pp, scaler_path)

    Usage (inference):
        pp = joblib.load(scaler_path)
        X_scaled = pp.transform_df(X)   # handles missing cols with fill_value=0
    """

    def __init__(self) -> None:
        self.scale_columns_: list[str] = []
        self._inner: Pipeline | None = None

    def fit(self, X_arr: np.ndarray, scale_columns: list[str], y=None):
        self.scale_columns_ = list(scale_columns)
        self._inner = Pipeline([
            ("winsorize", WinsorizerTransformer(lower_pct=1.0, upper_pct=99.0)),
            ("scale", RobustScaler()),
        ])
        self._inner.fit(X_arr)
        return self

    def transform(self, X_arr: np.ndarray) -> np.ndarray:
        """Transform a numpy array (same column order as fit)."""
        if self._inner is None:
            raise RuntimeError("PreprocessorPipeline not fitted yet.")
        return self._inner.transform(X_arr)

    def fit_transform_df(self, df: pd.DataFrame, scale_columns: list[str]) -> pd.DataFrame:
        """Fit on df[scale_columns] and return df with those columns scaled in-place."""
        df = df.copy()
        self.fit(df[scale_columns].values, scale_columns)
        df[scale_columns] = self.transform(df[scale_columns].values)
        return df

    def transform_df(self, df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
        """
        Transform df using saved scale_columns_.
        Missing columns are filled with fill_value; extra columns are kept as-is.
        """
        df = df.copy()
        valid_cols = [c for c in self.scale_columns_ if c in df.columns]
        missing = [c for c in self.scale_columns_ if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = fill_value
        if valid_cols or missing:
            df[self.scale_columns_] = self.transform(df[self.scale_columns_].values)
        return df
