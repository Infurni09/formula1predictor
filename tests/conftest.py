"""tests/conftest.py — shared pytest fixtures."""
from __future__ import annotations
import pytest

@pytest.fixture(scope="session")
def f1_sample_features():
    return {
        "grid_position": 1.0, "quali_pos": 1.0, "elo_rating": 1820.0,
        "constructor_momentum": 0.15, "roll_pos_3": 1.5, "roll_pos_5": 2.0,
        "roll_pos_10": 2.5, "roll_quali_3": 1.2, "roll_quali_5": 1.5,
        "roll_pts_3": 24.0, "roll_pts_5": 22.0, "roll_dnf_5": 0.0,
        "roll_win_5": 0.8, "roll_podium_5": 1.0, "lap_consistency": 0.95,
        "roll_pit_dur_5": 23500.0, "avg_pit_duration": 24000.0, "pit_count": 1,
        "career_races": 180, "career_wins": 54, "career_podiums": 98,
        "career_dnfs": 6, "pole_conversion": 0.62, "podium_conversion": 0.54,
        "is_street": 0, "is_night": 0, "circuit_familiarity": 0.9,
        "championship_gap": 0.0, "championship_pressure": 0.95,
        "races_remaining": 8, "cumulative_points": 331.0, "champ_pos": 1,
        "constructor_code": 5, "circuit_code": 10, "driver_code": 1,
    }
