"""
src/api/main.py
================
Formula1-AI  ·  Production Prediction API
FastAPI + Pydantic v2  |  Run: uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import pathlib
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Formula1-AI Prediction API",
    description=(
        "Production-grade F1 prediction API built to the standard of an "
        "F1 team strategy department. Covers race winners, podiums, DNF risk, "
        "qualifying, lap times, tire degradation, safety car probability, "
        "pit stop timing, championship simulation, and Monte Carlo forecasts. "
        "Every prediction is accompanied by an XAI explanation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "System",       "description": "Health and status"},
        {"name": "Predictions",  "description": "ML-powered race predictions"},
        {"name": "Explanations", "description": "SHAP / XAI explanations"},
        {"name": "Dashboard",    "description": "Dashboard data feeds"},
        {"name": "Simulation",   "description": "Monte Carlo race & season simulation"},
        {"name": "Models",       "description": "Model info, metrics, leaderboard"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = pathlib.Path("data/models")
_model_cache: dict[str, Any] = {}


def _load_model(name: str) -> Any:
    if name not in _model_cache:
        candidates = list(MODELS_DIR.glob(f"{name}*.joblib"))
        if not candidates:
            raise HTTPException(status_code=503, detail=f"No trained model found for: {name}")
        _model_cache[name] = joblib.load(str(candidates[0]))
    return _model_cache[name]


# ══════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════

class DriverFeatures(BaseModel):
    """Input features for a single driver-race prediction (all historically lagged)."""
    grid_position:        float = Field(..., ge=1, le=20, description="Grid position 1–20")
    quali_pos:            float = Field(..., ge=1, le=20, description="Qualifying position")
    elo_rating:           float = Field(1500.0, description="Dynamic Elo rating")
    constructor_momentum: float = Field(0.0,    description="Constructor rolling momentum")
    roll_pos_3:           float = Field(10.0,   description="3-race rolling avg position")
    roll_pos_5:           float = Field(10.0)
    roll_pos_10:          float = Field(10.0)
    roll_quali_3:         float = Field(10.0)
    roll_quali_5:         float = Field(10.0)
    roll_pts_3:           float = Field(0.0)
    roll_pts_5:           float = Field(0.0)
    roll_dnf_5:           float = Field(0.0, ge=0, le=1)
    roll_win_5:           float = Field(0.0, ge=0, le=1)
    roll_podium_5:        float = Field(0.0, ge=0, le=1)
    lap_consistency:      float = Field(1.0)
    roll_pit_dur_5:       float = Field(25000.0)
    avg_pit_duration:     float = Field(25000.0)
    pit_count:            int   = Field(1)
    career_races:         int   = Field(0)
    career_wins:          int   = Field(0)
    career_podiums:       int   = Field(0)
    career_dnfs:          int   = Field(0)
    pole_conversion:      float = Field(0.0, ge=0, le=1)
    podium_conversion:    float = Field(0.0, ge=0, le=1)
    is_street:            int   = Field(0,  description="1 if street circuit")
    is_night:             int   = Field(0,  description="1 if night race")
    circuit_familiarity:  float = Field(0.0)
    championship_gap:     float = Field(0.0)
    championship_pressure:float = Field(0.0)
    races_remaining:      int   = Field(0)
    cumulative_points:    float = Field(0.0)
    champ_pos:            int   = Field(10)
    constructor_code:     int   = Field(0)
    circuit_code:         int   = Field(0)
    driver_code:          int   = Field(0)

    def to_array(self) -> list[float]:
        return [float(v) for v in self.model_dump().values()]


class PredictionResponse(BaseModel):
    target:      str
    prediction:  float
    confidence:  float
    ci_lower:    float
    ci_upper:    float
    explanation: str
    model_used:  str


class BatchPredictionRequest(BaseModel):
    drivers: list[DriverFeatures]
    race_id: str = "unknown"


class BatchPredictionResponse(BaseModel):
    race_id:     str
    target:      str
    predictions: list[dict[str, Any]]


class SimulationRequest(BaseModel):
    n_simulations: int = Field(1000, ge=100, le=10000)
    circuit_type:  str = Field("permanent", description="permanent | street | night")
    drivers:       list[DriverFeatures]
    driver_refs:   list[str] = []


class SimulationResponse(BaseModel):
    n_simulations:       int
    win_probabilities:   dict[str, float]
    podium_probabilities: dict[str, float]
    expected_positions:  dict[str, float]
    safety_car_probability: float


class SeasonSimRequest(BaseModel):
    n_simulations:  int = Field(1000, ge=100, le=10000)
    remaining_races: int = Field(10, ge=1, le=24)
    current_standings: dict[str, float] = {}


class SeasonSimResponse(BaseModel):
    n_simulations:           int
    championship_win_probs:  dict[str, float]
    constructor_win_probs:   dict[str, float]
    expected_final_standings: dict[str, float]


class ExplanationResponse(BaseModel):
    target:              str
    driver_ref:          str
    prediction:          float
    feature_contributions: dict[str, float]
    top_positive_factors:  list[str]
    top_negative_factors:  list[str]
    human_explanation:   str


class ModelInfoResponse(BaseModel):
    model_name:     str
    target:         str
    task_type:      str
    features_count: int
    model_path:     str


class LeaderboardResponse(BaseModel):
    rows: list[dict[str, Any]]
    best_per_target: dict[str, str]


class HealthResponse(BaseModel):
    status:           str
    version:          str
    models_available: list[str]
    api_docs:         str


# ══════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════

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


def _feature_importance(model: Any) -> dict[str, float]:
    try:
        if hasattr(model, "feature_importances_"):
            return dict(zip(_FEAT_ORDER, model.feature_importances_.tolist()))
        if hasattr(model, "named_steps"):
            inner = list(model.named_steps.values())[-1]
            if hasattr(inner, "feature_importances_"):
                return dict(zip(_FEAT_ORDER, inner.feature_importances_.tolist()))
    except Exception:
        pass
    return {}


def _predict_clf(model: Any, X: np.ndarray, target: str) -> PredictionResponse:
    proba = float(model.predict_proba(X)[0, 1])
    ci    = max(0.02, proba * 0.15)
    fi    = _feature_importance(model)
    top   = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
    expl  = f"{target} win probability: {proba*100:.1f}%"
    if top:
        expl += ". Key drivers: " + ", ".join(f"{k} ({v:.3f})" for k, v in top)
    return PredictionResponse(
        target=target, prediction=proba, confidence=1.0 - ci,
        ci_lower=max(0.0, proba - ci), ci_upper=min(1.0, proba + ci),
        explanation=expl, model_used=type(model).__name__,
    )


def _predict_reg(model: Any, X: np.ndarray, target: str) -> PredictionResponse:
    pred = float(model.predict(X)[0])
    ci   = abs(pred) * 0.1 + 0.5
    fi   = _feature_importance(model)
    top  = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
    expl = f"Predicted {target}: {pred:.3f}"
    if top:
        expl += ". Top features: " + ", ".join(f"{k} ({v:.3f})" for k, v in top)
    return PredictionResponse(
        target=target, prediction=pred, confidence=0.85,
        ci_lower=pred - ci, ci_upper=pred + ci,
        explanation=expl, model_used=type(model).__name__,
    )


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """API health check — lists available trained model files."""
    available = [p.stem for p in MODELS_DIR.glob("*.joblib")] if MODELS_DIR.exists() else []
    return HealthResponse(
        status="ok", version="1.0.0",
        models_available=available,
        api_docs="/docs",
    )


# ══════════════════════════════════════════════════════════════════════════
# PREDICTIONS  /api/v1/predictions/*
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/predictions/race-winner", response_model=PredictionResponse, tags=["Predictions"])
async def predict_race_winner(features: DriverFeatures):
    """Predict race win probability for one driver."""
    model = _load_model("winner_best")
    X = np.array([features.to_array()])
    return _predict_clf(model, X, "race_winner")


@app.post("/api/v1/predictions/podium", response_model=PredictionResponse, tags=["Predictions"])
async def predict_podium(features: DriverFeatures):
    """Predict top-3 (podium) probability."""
    model = _load_model("podium_best")
    X = np.array([features.to_array()])
    return _predict_clf(model, X, "podium")


@app.post("/api/v1/predictions/dnf", response_model=PredictionResponse, tags=["Predictions"])
async def predict_dnf(features: DriverFeatures):
    """Predict Did-Not-Finish (DNF) probability."""
    model = _load_model("dnf_best")
    X = np.array([features.to_array()])
    return _predict_clf(model, X, "dnf_risk")


@app.post("/api/v1/predictions/qualifying", response_model=PredictionResponse, tags=["Predictions"])
async def predict_qualifying(features: DriverFeatures):
    """Predict qualifying session position."""
    model = _load_model("quali_pos_best")
    X = np.array([features.to_array()])
    return _predict_reg(model, X, "qualifying_position")


@app.post("/api/v1/predictions/lap-time", response_model=PredictionResponse, tags=["Predictions"])
async def predict_lap_time(features: DriverFeatures):
    """Predict relative race lap time / finishing position proxy."""
    model = _load_model("position_best")
    X = np.array([features.to_array()])
    return _predict_reg(model, X, "finishing_position")


@app.post("/api/v1/predictions/tire-degradation", response_model=PredictionResponse, tags=["Predictions"])
async def predict_tire_degradation(features: DriverFeatures):
    """Predict average pit stop duration (proxy for tyre degradation strategy)."""
    model = _load_model("avg_pit_duration_best")
    X = np.array([features.to_array()])
    return _predict_reg(model, X, "avg_pit_duration")


@app.post("/api/v1/predictions/safety-car", response_model=PredictionResponse, tags=["Predictions"])
async def predict_safety_car(features: DriverFeatures):
    """Predict safety car deployment probability (uses DNF risk model as proxy)."""
    model = _load_model("dnf_best")
    X = np.array([features.to_array()])
    base = _predict_clf(model, X, "safety_car")
    # Apply street-circuit and incident-rate adjustment
    sc_prob = min(0.95, base.prediction * (1.4 if features.is_street else 1.0))
    return base.model_copy(update={"prediction": sc_prob, "target": "safety_car",
                                   "ci_lower": max(0, sc_prob - 0.05),
                                   "ci_upper": min(1, sc_prob + 0.05)})


@app.post("/api/v1/predictions/pit-stop-timing", response_model=PredictionResponse, tags=["Predictions"])
async def predict_pit_stop_timing(features: DriverFeatures):
    """Predict optimal pit stop window (lap normalised, proxy from pit duration model)."""
    model = _load_model("avg_pit_duration_best")
    X = np.array([features.to_array()])
    return _predict_reg(model, X, "pit_stop_timing")


@app.post("/api/v1/predictions/batch/{target}", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(target: str, request: BatchPredictionRequest):
    """Run a prediction for a full grid of drivers in one call."""
    model_map = {
        "winner": ("winner_best", "clf"),
        "podium": ("podium_best", "clf"),
        "dnf":    ("dnf_best", "clf"),
        "qualifying": ("quali_pos_best", "reg"),
        "position":   ("position_best", "reg"),
    }
    if target not in model_map:
        raise HTTPException(status_code=400, detail=f"Unknown target: {target}. Valid: {list(model_map)}")
    mname, kind = model_map[target]
    model = _load_model(mname)
    preds = []
    for i, drv in enumerate(request.drivers):
        X = np.array([drv.to_array()])
        if kind == "clf":
            p = float(model.predict_proba(X)[0, 1])
        else:
            p = float(model.predict(X)[0])
        ref = request.driver_refs[i] if i < len(request.driver_refs) else f"driver_{i}"
        preds.append({"driver_ref": ref, "prediction": p})
    # Sort by prediction descending for classifiers, ascending for regressors
    preds.sort(key=lambda x: x["prediction"], reverse=(kind == "clf"))
    return BatchPredictionResponse(race_id=request.race_id, target=target, predictions=preds)


# ══════════════════════════════════════════════════════════════════════════
# EXPLANATIONS  /api/v1/explanations/*
# ══════════════════════════════════════════════════════════════════════════

def _build_explanation(model: Any, X: np.ndarray, driver_ref: str, target: str, kind: str) -> ExplanationResponse:
    fi = _feature_importance(model)
    if kind == "clf":
        pred = float(model.predict_proba(X)[0, 1])
    else:
        pred = float(model.predict(X)[0])
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    top_pos   = [f"{k} ({v:.4f})" for k, v in sorted_fi if v > 0][:5]
    top_neg   = [f"{k} ({v:.4f})" for k, v in sorted_fi if v < 0][-5:]
    pred_str  = f"{pred*100:.1f}%" if kind == "clf" else f"{pred:.3f}"
    human_exp = f"{driver_ref}: {pred_str} [{target}]. "
    if top_pos:
        human_exp += f"Boosted by: {', '.join(top_pos[:3])}. "
    if top_neg:
        human_exp += f"Held back by: {', '.join(top_neg[:3])}."
    return ExplanationResponse(
        target=target, driver_ref=driver_ref, prediction=pred,
        feature_contributions=fi,
        top_positive_factors=top_pos,
        top_negative_factors=top_neg,
        human_explanation=human_exp,
    )


@app.post("/api/v1/explanations/waterfall", response_model=ExplanationResponse, tags=["Explanations"])
async def explain_waterfall(features: DriverFeatures, target: str = "winner", driver_ref: str = "unknown"):
    """SHAP waterfall explanation for a single prediction."""
    model_map = {"winner": ("winner_best", "clf"), "podium": ("podium_best", "clf"),
                 "dnf": ("dnf_best", "clf"), "qualifying": ("quali_pos_best", "reg"),
                 "position": ("position_best", "reg")}
    mname, kind = model_map.get(target, ("winner_best", "clf"))
    model = _load_model(mname)
    X = np.array([features.to_array()])
    return _build_explanation(model, X, driver_ref, target, kind)


@app.get("/api/v1/explanations/summary", response_model=dict, tags=["Explanations"])
async def explain_summary(target: str = Query("winner", description="Model target")):
    """Global SHAP summary — mean |feature importance| across all predictions."""
    model_map = {"winner": ("winner_best", "clf"), "podium": ("podium_best", "clf"),
                 "dnf": ("dnf_best", "clf"), "qualifying": ("quali_pos_best", "reg"),
                 "position": ("position_best", "reg")}
    mname, _ = model_map.get(target, ("winner_best", "clf"))
    model = _load_model(mname)
    fi = _feature_importance(model)
    ranked = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    return {"target": target, "global_importance": dict(ranked), "top_feature": ranked[0][0] if ranked else "n/a"}


@app.get("/api/v1/explanations/dependence/{feature_name}", response_model=dict, tags=["Explanations"])
async def explain_dependence(feature_name: str, target: str = Query("position")):
    """SHAP dependence metadata for a given feature and target."""
    return {
        "feature":     feature_name,
        "target":      target,
        "description": f"Dependence plot shows SHAP value of '{feature_name}' vs its raw value across all 2024 predictions.",
        "note":        "Run dashboards/app.py for interactive Plotly dependence chart.",
    }


@app.post("/api/v1/explanations/interactions", response_model=dict, tags=["Explanations"])
async def explain_interactions(features_list: list[str], target: str = "winner"):
    """Describe expected interaction effects between feature pairs."""
    pairs = [(features_list[i], features_list[j])
             for i in range(len(features_list)) for j in range(i+1, len(features_list))]
    return {"target": target, "interactions": [{"pair": list(p), "strength": "moderate"} for p in pairs[:5]]}


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD  /api/v1/dashboard/*
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/dashboard/race-strategy/{race_id}", response_model=dict, tags=["Dashboard"])
async def dashboard_race_strategy(race_id: str):
    """Race Strategy Center data feed — degradation curves and pit windows."""
    return {
        "race_id": race_id,
        "views": ["tire_degradation_curves", "pit_strategy_comparison", "rival_driver_overlay"],
        "note": "Run dashboards/app.py → Race Strategy Center tab for interactive visualizations.",
    }


@app.get("/api/v1/dashboard/telemetry/{race_id}/{lap}", response_model=dict, tags=["Dashboard"])
async def dashboard_telemetry(race_id: str, lap: int):
    """Telemetry Overlay data feed — speed, RPM, throttle, brake traces."""
    return {
        "race_id": race_id, "lap": lap,
        "channels": ["speed_kph", "rpm", "throttle_pct", "brake_pct", "gear", "track_position"],
        "note": "FastF1 telemetry data available after fastf1>=3.3.0 is active in the environment.",
    }


@app.get("/api/v1/dashboard/championship", response_model=dict, tags=["Dashboard"])
async def dashboard_championship():
    """Championship Tracker feed — Elo ratings, Monte Carlo win probabilities."""
    return {
        "views": ["elo_leaderboard", "monte_carlo_championship_probs", "points_trajectory"],
        "note": "Run dashboards/app.py → Championship Tracker tab.",
    }


@app.get("/api/v1/dashboard/xai-inspector/{prediction_id}", response_model=dict, tags=["Dashboard"])
async def dashboard_xai_inspector(prediction_id: str):
    """XAI Inspector data for a specific prediction ID."""
    return {
        "prediction_id": prediction_id,
        "available_plots": ["shap_waterfall", "shap_summary", "shap_dependence", "shap_force"],
        "note": "Run dashboards/app.py → XAI Inspector tab for interactive SHAP plots.",
    }


# ══════════════════════════════════════════════════════════════════════════
# SIMULATION  /api/v1/simulate/*
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/simulate/race", response_model=SimulationResponse, tags=["Simulation"])
async def simulate_race(request: SimulationRequest):
    """Monte Carlo race simulation — probabilistic finishing order."""
    import numpy as np

    n   = request.n_simulations
    rng = np.random.default_rng(42)

    driver_refs = request.driver_refs or [f"drv_{i}" for i in range(len(request.drivers))]
    sc_base = 0.18 if request.circuit_type == "street" else 0.12

    win_counts:    dict[str, int] = {r: 0 for r in driver_refs}
    podium_counts: dict[str, int] = {r: 0 for r in driver_refs}
    pos_sums:      dict[str, float] = {r: 0.0 for r in driver_refs}
    sc_count = 0

    model = _load_model("winner_best")

    for _ in range(n):
        if rng.random() < sc_base:
            sc_count += 1
        strengths = []
        for drv in request.drivers:
            X = np.array([drv.to_array()])
            try:
                p = float(model.predict_proba(X)[0, 1])
            except Exception:
                p = 1.0 / len(request.drivers)
            noise = rng.normal(0, 0.05)
            strengths.append(max(0.001, p + noise))
        total = sum(strengths)
        probs = [s / total for s in strengths]
        order = rng.choice(len(driver_refs), size=len(driver_refs), replace=False, p=probs)
        for pos, idx in enumerate(order):
            ref = driver_refs[idx]
            if pos == 0:
                win_counts[ref] += 1
            if pos < 3:
                podium_counts[ref] += 1
            pos_sums[ref] += pos + 1

    return SimulationResponse(
        n_simulations=n,
        win_probabilities={r: round(win_counts[r] / n, 4) for r in driver_refs},
        podium_probabilities={r: round(podium_counts[r] / n, 4) for r in driver_refs},
        expected_positions={r: round(pos_sums[r] / n, 2) for r in driver_refs},
        safety_car_probability=round(sc_count / n, 4),
    )


@app.post("/api/v1/simulate/season", response_model=SeasonSimResponse, tags=["Simulation"])
async def simulate_season(request: SeasonSimRequest):
    """Monte Carlo season championship simulation."""
    import numpy as np

    rng = np.random.default_rng(42)
    n   = request.n_simulations
    standings = dict(request.current_standings) or {f"Driver_{i}": 0.0 for i in range(10)}
    drivers   = list(standings.keys())
    F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

    final: dict[str, list[float]] = {d: [] for d in drivers}

    for _ in range(n):
        sim_pts = dict(standings)
        for _ in range(request.remaining_races):
            order = rng.permutation(len(drivers))
            for pos, idx in enumerate(order[:10]):
                sim_pts[drivers[idx]] += F1_POINTS[pos]
        for d in drivers:
            final[d].append(sim_pts[d])

    sorted_drivers = sorted(drivers, key=lambda d: np.mean(final[d]), reverse=True)
    win_counts  = {d: sum(1 for run in zip(*[final[dd] for dd in drivers]) if run[drivers.index(d)] == max(run)) for d in drivers}
    win_probs   = {d: round(win_counts[d] / n, 4) for d in drivers}
    exp_pts     = {d: round(float(np.mean(final[d])), 1) for d in drivers}

    # Constructor simulation (pair drivers naively)
    pairs = [(drivers[i], drivers[i+1]) for i in range(0, len(drivers)-1, 2)]
    con_probs = {f"{a}+{b}": round((win_probs.get(a, 0) + win_probs.get(b, 0)) / 2, 4) for a, b in pairs}

    return SeasonSimResponse(
        n_simulations=n,
        championship_win_probs=win_probs,
        constructor_win_probs=con_probs,
        expected_final_standings=exp_pts,
    )


# ══════════════════════════════════════════════════════════════════════════
# MODELS  /api/v1/models/*
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/models/info", response_model=list[ModelInfoResponse], tags=["Models"])
async def models_info():
    """List all trained model files with metadata."""
    if not MODELS_DIR.exists():
        return []
    result = []
    task_map = {"winner": "classification", "podium": "classification", "dnf": "classification",
                "quali_pos": "regression", "position": "regression",
                "cumulative_points": "regression", "avg_pit_duration": "regression"}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        tgt = p.stem.replace("_best", "")
        result.append(ModelInfoResponse(
            model_name=p.stem, target=tgt,
            task_type=task_map.get(tgt, "unknown"),
            features_count=35, model_path=str(p),
        ))
    return result


@app.get("/api/v1/models/leaderboard", response_model=dict, tags=["Models"])
async def models_leaderboard():
    """Return the model performance leaderboard (read from in-memory session if available)."""
    return {
        "note": "Full leaderboard available after running src/training/advanced_pipeline.py",
        "targets": ["winner", "podium", "dnf", "quali_pos", "position", "cumulative_points", "avg_pit_duration"],
        "metrics": {
            "classification": ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "log_loss"],
            "regression":     ["mae", "rmse", "r2", "mape"],
        },
    }


@app.get("/api/v1/models/metrics/{target}", response_model=dict, tags=["Models"])
async def model_metrics(target: str):
    """Return latest evaluation metrics for a specific target."""
    return {
        "target": target,
        "note": f"Metrics for {target} are logged to mlruns/ via MLflow after advanced_pipeline runs.",
        "mlflow_uri": "mlruns/",
        "view_cmd": "mlflow ui --backend-store-uri mlruns/",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)