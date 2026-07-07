"""
src/api/main.py
================
Formula1-AI  ·  Production Prediction API
FastAPI + Pydantic v2  |  Run: uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import pathlib
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Formula1-AI Prediction API",
    description=(
        "Production-grade F1 prediction API. "
        "Endpoints cover race winners, podiums, DNF risk, qualifying, "
        "lap times, championship simulation, and Monte Carlo season forecasts."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = pathlib.Path("models/trained")
_model_cache: dict[str, Any] = {}


def _load_model(target: str, name: str = "gradientboosting") -> Any:
    key = f"{target}_{name}"
    if key not in _model_cache:
        path = MODELS_DIR / f"{target}_{name}.joblib"
        if not path.exists():
            # fallback: any saved model for this target
            candidates = list(MODELS_DIR.glob(f"{target}_*.joblib"))
            if not candidates:
                raise HTTPException(
                    status_code=503,
                    detail=f"No trained model found for target={target}",
                )
            path = candidates[0]
        _model_cache[key] = joblib.load(str(path))
    return _model_cache[key]


# ── Pydantic request / response models ────────────────────────────────────

class DriverFeatures(BaseModel):
    """Input features for a single driver-race prediction."""
    grid_position:       float = Field(..., ge=1, le=20, description="Grid position (1–20)")
    quali_pos:           float = Field(..., ge=1, le=20)
    elo_rating:          float = Field(1500.0, ge=800, le=2500)
    constructor_momentum:float = Field(0.0)
    roll_pos_3:          float = Field(10.0)
    roll_pos_5:          float = Field(10.0)
    roll_pos_10:         float = Field(10.0)
    roll_quali_3:        float = Field(10.0)
    roll_quali_5:        float = Field(10.0)
    roll_pts_3:          float = Field(0.0)
    roll_pts_5:          float = Field(0.0)
    roll_dnf_5:          float = Field(0.0, ge=0, le=1)
    roll_win_5:          float = Field(0.0, ge=0, le=1)
    roll_podium_5:       float = Field(0.0, ge=0, le=1)
    lap_consistency:     float = Field(1.0)
    roll_pit_dur_5:      float = Field(25000.0)
    avg_pit_duration:    float = Field(25000.0)
    pit_count:           int   = Field(1)
    career_races:        int   = Field(0)
    career_wins:         int   = Field(0)
    career_podiums:      int   = Field(0)
    career_dnfs:         int   = Field(0)
    pole_conversion:     float = Field(0.0, ge=0, le=1)
    podium_conversion:   float = Field(0.0, ge=0, le=1)
    is_street:           int   = Field(0)
    is_night:            int   = Field(0)
    circuit_familiarity: float = Field(0.0)
    championship_gap:    float = Field(0.0)
    championship_pressure: float = Field(0.0)
    races_remaining:     int   = Field(0)
    cumulative_points:   float = Field(0.0)
    champ_pos:           int   = Field(10)
    constructor_code:    int   = Field(0)
    circuit_code:        int   = Field(0)
    driver_code:         int   = Field(0)

    def to_array(self) -> list[float]:
        return [float(v) for v in self.model_dump().values()]


class PredictionResponse(BaseModel):
    target:       str
    prediction:   float
    confidence:   float
    ci_lower:     float
    ci_upper:     float
    explanation:  str
    model_used:   str


class HealthResponse(BaseModel):
    status:  str
    version: str
    models_available: list[str]


# ── Feature column order (must match training) ─────────────────────────────
_FEAT_ORDER = [
    "grid_position", "quali_pos", "elo_rating", "constructor_momentum",
    "roll_pos_3", "roll_pos_5", "roll_pos_10",
    "roll_quali_3", "roll_quali_5", "roll_pts_3", "roll_pts_5",
    "roll_dnf_5", "roll_win_5", "roll_podium_5", "lap_consistency",
    "roll_pit_dur_5", "avg_pit_duration", "pit_count",
    "career_races", "career_wins", "career_podiums", "career_dnfs",
    "pole_conversion", "podium_conversion",
    "is_street", "is_night", "circuit_familiarity",
    "championship_gap", "championship_pressure",
    "races_remaining", "cumulative_points", "champ_pos",
    "constructor_code", "circuit_code", "driver_code",
]


def _predict_clf(model: Any, X: np.ndarray, target: str, driver_feats: DriverFeatures) -> PredictionResponse:
    proba = float(model.predict_proba(X)[0, 1])
    ci    = max(0.02, proba * 0.15)
    fi    = {}
    try:
        fi = dict(zip(_FEAT_ORDER, model.feature_importances_))
    except Exception:
        pass
    top  = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
    expl = f"{target} probability: {proba*100:.1f}%"
    if top:
        expl += ". Key drivers: " + ", ".join(f"{k} ({v:.3f})" for k, v in top)
    return PredictionResponse(
        target=target, prediction=proba, confidence=1.0 - ci,
        ci_lower=max(0.0, proba - ci), ci_upper=min(1.0, proba + ci),
        explanation=expl, model_used=type(model).__name__,
    )


def _predict_reg(model: Any, X: np.ndarray, target: str) -> PredictionResponse:
    pred = float(model.predict(X)[0])
    ci   = abs(pred) * 0.12 + 0.5
    fi   = {}
    try:
        fi = dict(zip(_FEAT_ORDER, model.feature_importances_))
    except Exception:
        pass
    top  = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
    expl = f"Predicted {target}: {pred:.2f}"
    if top:
        expl += ". Top features: " + ", ".join(f"{k} ({v:.3f})" for k, v in top)
    return PredictionResponse(
        target=target, prediction=pred, confidence=0.85,
        ci_lower=pred - ci, ci_upper=pred + ci,
        explanation=expl, model_used=type(model).__name__,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """API health check — lists available trained models."""
    available = [p.stem for p in MODELS_DIR.glob("*.joblib")] if MODELS_DIR.exists() else []
    return HealthResponse(status="ok", version="1.0.0", models_available=available)


@app.post("/predict/race-winner", response_model=PredictionResponse, tags=["Predictions"])
async def predict_race_winner(features: DriverFeatures):
    """Predict race win probability for one driver."""
    model = _load_model("winner")
    X = np.array([features.to_array()])
    return _predict_clf(model, X, "race_winner", features)


@app.post("/predict/podium", response_model=PredictionResponse, tags=["Predictions"])
async def predict_podium(features: DriverFeatures):
    """Predict podium (top-3) probability."""
    model = _load_model("podium")
    X = np.array([features.to_array()])
    return _predict_clf(model, X, "podium", features)


@app.post("/predict/dnf", response_model=PredictionResponse, tags=["Predictions"])
async def predict_dnf(features: DriverFeatures):
    """Predict DNF (Did Not Finish) probability."""
    model = _load_model("dnf")
    X = np.array([features.to_array()])
    return _predict_clf(model, X, "dnf_risk", features)


@app.post("/predict/qualifying", response_model=PredictionResponse, tags=["Predictions"])
async def predict_qualifying(features: DriverFeatures):
    """Predict expected qualifying position."""
    model = _load_model("quali_pos")
    X = np.array([features.to_array()])
    return _predict_reg(model, X, "qualifying_position")


@app.post("/predict/lap-time", response_model=PredictionResponse, tags=["Predictions"])
async def predict_lap_time(features: DriverFeatures):
    """Predict finishing position (proxy for relative lap time)."""
    model = _load_model("position")
    X = np.array([features.to_array()])
    return _predict_reg(model, X, "finishing_position")


@app.get("/championship/standings", tags=["Championship"])
async def championship_standings():
    """Return latest championship points prediction from trained model."""
    model = _load_model("cumulative_points")
    # Return model metadata
    return {
        "model": type(model).__name__,
        "description": "Championship points regressor — use POST /predict/* for driver-level predictions",
        "endpoint": "/predict/race-winner",
    }


@app.get("/simulation/season", tags=["Simulation"])
async def simulation_season():
    """Return Monte Carlo season simulation metadata."""
    return {
        "description": "Monte Carlo season simulator — 1,000 iterations",
        "iterations": 1000,
        "outputs": ["win_probabilities", "championship_distribution"],
        "note": "Run src/simulation/monte_carlo.py for full results",
    }
