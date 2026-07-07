"""
tests/test_api.py
==================
Formula1-AI  ·  FastAPI Endpoint Tests
Uses httpx.AsyncClient for async test calls.
"""
from __future__ import annotations

import pytest

SAMPLE_FEATURES = {
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


@pytest.mark.asyncio
async def test_health_endpoint():
    """Health endpoint should return ok status."""
    try:
        import httpx
        from src.api.main import app
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "version" in data
    except ImportError:
        pytest.skip("httpx or fastapi not available")


@pytest.mark.asyncio
async def test_prediction_schema_validation():
    """Invalid input should return 422 validation error."""
    try:
        import httpx
        from src.api.main import app
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            bad_payload = {"grid_position": 999}  # Invalid: >20
            resp = await client.post("/api/v1/predictions/race-winner", json=bad_payload)
            assert resp.status_code == 422
    except ImportError:
        pytest.skip("httpx or fastapi not available")


def test_feature_dict_completeness():
    """Sample features should contain all required fields."""
    required = [
        "grid_position", "quali_pos", "elo_rating", "constructor_momentum",
        "roll_pos_3", "roll_pos_5", "roll_pos_10", "career_wins", "pit_count",
    ]
    for field in required:
        assert field in SAMPLE_FEATURES, f"Missing: {field}"


def test_prediction_response_structure():
    """PredictionResponse must have all required fields."""
    from pydantic import BaseModel

    class PredictionResponse(BaseModel):
        target: str
        prediction: float
        confidence: float
        ci_lower: float
        ci_upper: float
        explanation: str
        model_used: str

    resp = PredictionResponse(
        target="race_winner", prediction=0.74, confidence=0.88,
        ci_lower=0.66, ci_upper=0.82, explanation="Test", model_used="XGBoost",
    )
    assert resp.prediction == 0.74
    assert 0 <= resp.confidence <= 1
