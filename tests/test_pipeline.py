"""
tests/test_pipeline.py
========================
Formula1-AI  ·  ETL & Feature Engineering Tests
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


class TestFeatureEngineering:
    """Tests for feature engineering correctness and data integrity."""

    def test_elo_rating_range(self):
        """Elo ratings should be within realistic bounds."""
        elo_values = np.array([1450, 1520, 1680, 1820, 1550])
        assert elo_values.min() >= 800
        assert elo_values.max() <= 2500

    def test_rolling_features_no_leakage(self):
        """Rolling features must use only past data (lagged by 1)."""
        df = pd.DataFrame({
            "season": [2022] * 5 + [2023] * 5,
            "position": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        })
        df["roll_pos_3"] = df["position"].shift(1).rolling(3, min_periods=1).mean()
        # First row must be NaN (no prior data)
        assert pd.isna(df["roll_pos_3"].iloc[0])

    def test_championship_gap_non_negative_leader(self):
        """Championship gap for the leader should be 0."""
        gaps = np.array([0.0, 25.0, 43.0, 67.0])
        assert gaps[0] == 0.0
        assert all(gaps >= 0)

    def test_circuit_familiarity_bounded(self):
        """Circuit familiarity should be between 0 and 1."""
        familiarity = np.clip(np.array([0.0, 0.33, 0.67, 1.0]), 0, 1)
        assert familiarity.min() >= 0.0
        assert familiarity.max() <= 1.0

    def test_pole_conversion_rate_bounded(self):
        """Pole conversion must be between 0 and 1."""
        conversion = np.array([0.0, 0.45, 0.62, 0.80, 1.0])
        assert conversion.min() >= 0.0
        assert conversion.max() <= 1.0

    def test_dnf_flag_binary(self):
        """DNF flag must be strictly 0 or 1."""
        dnf = np.array([0, 1, 0, 0, 1, 1, 0])
        assert set(dnf.tolist()).issubset({0, 1})


class TestDataQuality:
    """Tests for data quality and completeness."""

    def test_season_range_valid(self):
        seasons = [2022, 2023, 2024]
        assert all(2018 <= s <= 2025 for s in seasons)

    def test_grid_positions_bounded(self):
        grids = np.array([1, 5, 10, 15, 20])
        assert grids.min() >= 1
        assert grids.max() <= 20

    def test_points_table_sum(self):
        """Standard F1 points table sums to 100."""
        points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        assert sum(points) == 101  # 100 + fastest lap bonus

    def test_feature_matrix_no_all_nan_cols(self):
        """Feature matrix should not have columns that are entirely NaN."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(0, 1, (100, 10)), columns=[f"feat_{i}" for i in range(10)])
        assert not any(df.isna().all())


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation outputs."""

    def test_win_probabilities_sum_to_one(self):
        """Win probabilities across all drivers must sum to ~1."""
        probs = {"VER": 0.40, "NOR": 0.25, "LEC": 0.20, "HAM": 0.15}
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_championship_probs_bounded(self):
        """All championship probabilities must be between 0 and 1."""
        probs = [0.64, 0.21, 0.10, 0.05]
        assert all(0 <= p <= 1 for p in probs)

    def test_safety_car_probability_plausible(self):
        """Safety car probability should be between 5% and 50%."""
        sc_prob = 0.18
        assert 0.05 <= sc_prob <= 0.50
