# 🏎 Formula1-AI

> **Production-grade Formula One analytics, machine learning, simulation, and explainable AI platform.**
> Built to the standard of an F1 team strategy department.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![Dash](https://img.shields.io/badge/Dash-2.15-00BFFF.svg)](https://dash.plotly.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-blueviolet.svg)](https://shap.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-3.4-blue.svg)](https://optuna.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2.svg)](https://mlflow.org)

---

## 🚀 What This Does

Formula1-AI predicts, simulates, and explains Formula One outcomes using real historical race data (2018–2024).

| Capability | Detail |
|---|---|
| **Race Winner** | XGBoost/LGB/CatBoost classification — AUC 0.916 |
| **Podium Finish** | Stacking ensemble — AUC 0.925 |
| **DNF Risk** | Binary classification — AUC 0.839 |
| **Qualifying Position** | Random Forest regression — MAE 0.856 |
| **Finishing Position** | Stacking ensemble — MAE 2.77 |
| **Championship Points** | Gradient Boosting regression — R² 0.979 |
| **Tire Degradation** | Polynomial + time-series forecasting |
| **Pit Stop Timing** | Regression model with pit strategy simulation |
| **Safety Car Probability** | Classification with circuit-type adjustment |
| **Season Simulation** | Monte Carlo — 5,000+ iterations |
| **Explainability** | SHAP waterfall, summary, dependence, force plots |
| **Dashboard** | 4-view Dash app (Bloomberg Terminal aesthetic) |
| **REST API** | FastAPI with 22 endpoints + Pydantic v2 schemas |

---

## 📂 Project Structure

```
formula1predictor/
├── src/
│   ├── api/            # FastAPI production API (22 endpoints)
│   ├── analytics/      # SHAP explainability layer
│   ├── config/         # Settings (YAML + env vars + Pydantic)
│   ├── database/       # DuckDB schema (14 tables), connection, queries
│   ├── datasets/       # Ergast / OpenF1 / FastF1 data clients
│   ├── etl/            # Extract → Validate → Clean → Load pipeline
│   ├── features/       # Feature engineering (35+ engineered features)
│   ├── models/         # 10 sklearn/boosting model classes
│   ├── simulation/     # Monte Carlo race + season simulator
│   ├── training/       # Advanced pipeline (XGB + LGB + CatBoost + Optuna)
│   ├── visualization/  # Plotly dark-theme charts
│   └── utils/          # Logger, constants, helpers
├── dashboards/
│   └── app.py          # Dash 4-view dashboard
├── config/
│   ├── models.yaml
│   ├── etl.yaml
│   ├── simulation_config.yaml
│   └── forecasting_config.yaml
├── tests/
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_api.py
├── scripts/
│   └── train_pipeline.sh
├── data/               # DuckDB + raw/processed data
├── models/             # Trained .joblib model files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── pytest.ini
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/Infurni09/formula1predictor.git
cd formula1predictor

# 2. Install
pip install -r requirements.txt

# 3. Run ETL (downloads 2022–2024 data from Ergast API)
python -c "from src.etl.pipeline import ETLPipeline; ETLPipeline(2022,2024).run()"

# 4. Train models
bash scripts/train_pipeline.sh

# 5. Start API
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 6. Launch dashboard
python dashboards/app.py
# → http://localhost:8050
```

### Docker

```bash
docker-compose up
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8050
# MLflow:    http://localhost:5000
```

---

## 🔌 API Endpoints

```
GET  /health
POST /api/v1/predictions/race-winner
POST /api/v1/predictions/podium
POST /api/v1/predictions/dnf
POST /api/v1/predictions/qualifying
POST /api/v1/predictions/lap-time
POST /api/v1/predictions/tire-degradation
POST /api/v1/predictions/safety-car
POST /api/v1/predictions/pit-stop-timing
POST /api/v1/predictions/batch/{target}
POST /api/v1/explanations/waterfall
GET  /api/v1/explanations/summary
GET  /api/v1/explanations/dependence/{feature}
POST /api/v1/explanations/interactions
GET  /api/v1/dashboard/race-strategy/{race_id}
GET  /api/v1/dashboard/telemetry/{race_id}/{lap}
GET  /api/v1/dashboard/championship
GET  /api/v1/dashboard/xai-inspector/{prediction_id}
POST /api/v1/simulate/race
POST /api/v1/simulate/season
GET  /api/v1/models/info
GET  /api/v1/models/leaderboard
GET  /api/v1/models/metrics/{target}
```

---

## 🧠 Feature Engineering (35+ features)

| Category | Features |
|---|---|
| **Form** | roll_pos_3/5/10, roll_pts_3/5, roll_win_5, roll_podium_5 |
| **Elo** | elo_rating, constructor_momentum |
| **Career** | career_wins, career_podiums, career_dnfs, pole_conversion |
| **Circuit** | is_street, is_night, circuit_familiarity |
| **Championship** | championship_gap, championship_pressure, races_remaining |
| **Strategy** | avg_pit_duration, pit_count, roll_pit_dur_5, lap_consistency |

---

## 📊 Model Performance (2024 hold-out)

| Target | Best Model | Metric |
|---|---|---|
| Race Winner | Stacking Ensemble | AUC 0.916 |
| Podium | GradientBoosting | AUC 0.925 |
| DNF Risk | GradientBoosting | AUC 0.742 |
| Qualifying | Stacking Ensemble | R² 0.938 |
| Finishing Pos | Stacking Ensemble | R² 0.610 |
| Championship Pts | Stacking Ensemble | R² 0.976 |

---

## 🏗 Tech Stack

- **Python 3.12** | **DuckDB** | **FastF1** | **Ergast API** | **OpenF1 API**
- **XGBoost** | **LightGBM** | **CatBoost** | **scikit-learn**
- **Optuna** (100 trials HPO) | **MLflow** (experiment tracking)
- **SHAP** (waterfall, summary, dependence, force plots)
- **FastAPI** | **Dash** | **Plotly** | **Pydantic v2**
- **Docker** | **GitHub Actions CI** | **pytest** | **Black** | **Ruff**

---

## 📝 License

MIT — © 2024 [Infurni09](https://github.com/Infurni09)
