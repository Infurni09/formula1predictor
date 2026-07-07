"""
tests/test_models.py
======================
Formula1-AI  ·  ML Model Tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@pytest.fixture
def synthetic_clf_data():
    rng = np.random.default_rng(42)
    n   = 200
    X   = rng.normal(0, 1, (n, 10))
    y   = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y


@pytest.fixture
def synthetic_reg_data():
    rng = np.random.default_rng(42)
    n   = 200
    X   = rng.normal(0, 1, (n, 10))
    y   = X[:, 0] * 5 + X[:, 1] * 3 + rng.normal(0, 1, n)
    return X, y


class TestClassificationModels:

    def test_gradient_boosting_clf_auc(self, synthetic_clf_data):
        X, y = synthetic_clf_data
        model = Pipeline([("s", StandardScaler()), ("m", GradientBoostingClassifier(n_estimators=50, random_state=42))])
        model.fit(X[:160], y[:160])
        proba = model.predict_proba(X[160:])[:, 1]
        auc   = roc_auc_score(y[160:], proba)
        assert auc > 0.6, f"AUC too low: {auc:.3f}"

    def test_clf_predict_proba_bounded(self, synthetic_clf_data):
        X, y = synthetic_clf_data
        model = Pipeline([("s", StandardScaler()), ("m", GradientBoostingClassifier(n_estimators=30, random_state=42))])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestRegressionModels:

    def test_gradient_boosting_reg_r2(self, synthetic_reg_data):
        X, y = synthetic_reg_data
        model = Pipeline([("s", StandardScaler()), ("m", GradientBoostingRegressor(n_estimators=50, random_state=42))])
        model.fit(X[:160], y[:160])
        pred = model.predict(X[160:])
        r2   = r2_score(y[160:], pred)
        assert r2 > 0.5, f"R² too low: {r2:.3f}"

    def test_mae_reasonable(self, synthetic_reg_data):
        X, y = synthetic_reg_data
        model = Pipeline([("s", StandardScaler()), ("m", GradientBoostingRegressor(n_estimators=50, random_state=42))])
        model.fit(X[:160], y[:160])
        pred = model.predict(X[160:])
        mae  = mean_absolute_error(y[160:], pred)
        assert mae < 5.0, f"MAE too high: {mae:.3f}"


class TestModelPersistence:

    def test_joblib_save_load(self, tmp_path, synthetic_clf_data):
        import joblib
        X, y = synthetic_clf_data
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        path = tmp_path / "test_model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        pred_orig   = model.predict(X)
        pred_loaded = loaded.predict(X)
        assert np.array_equal(pred_orig, pred_loaded)
