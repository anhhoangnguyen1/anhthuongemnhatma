"""
Shared preprocessing transformers used in both training and inference.
Keeping this in a standalone module ensures joblib can pickle/unpickle
WinsorizerTransformer regardless of how train_xgboost_dss.py is executed.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
