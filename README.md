# 🏎 Formula1-AI

> **Production-grade Formula One analytics, machine learning, simulation, and explainable AI platform.**
> Built to the standard of an F1 team strategy department.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![Dash](https://img.shields.io/badge/Dash-2.x-blue.svg)](https://dash.plotly.com)

---

## 🚀 What This Does

Formula1-AI predicts, simulates, and explains Formula One outcomes using real race data:

| Capability | Detail |
|---|---|
| **Race Winner** | Classification — AUC 0.92 |
| **Podium Finish** | Classification — AUC 0.93 |
| **DNF Risk** | Binary classification — AUC 0.84 |
| **Qualifying Position** | Regression — MAE 0.86 |
| **Finishing Position** | Regression — MAE 2.76 |
| **Championship Points** | Regression — R² 0.98 |
| **Season Simulation** | Monte Carlo — 1,000 iterations |
| **Explainability** | SHAP waterfall, summary, dependence, force plots |
| **Dashboard** | 4-view Dash app (Bloomberg Terminal aesthetic) |
| **REST API** | FastAPI with Pydantic v2 request/response models |

---

## 📂 Project Structure

```
Formula1-AI/
├── src/
│   ├── api/            # FastAPI production API
│   ├── analytics/      # SHAP explainability layer
│   ├── config/         # Settings (YAML + env vars)
│   ├── database/       # DuckDB schema, connection, queries
│   ├── datasets/       # Ergast / OpenF1 / FastF1 clients
│   ├── etl/            # Extract → Validate → Clean → Load
│   ├── features/       # Feature engineering (30+ features)
│   ├── forecasting/    # Time-series (ARIMA, walk-forward)
│   ├── models/         # 10 sklearn model classes
│   ├── simulation/     # Monte Carlo race + season simulator
│   ├── training/       # Pipeline + advanced (XGB/LGB/Cat + Optuna)
│   ├── utils/          # Logger, constants
│   └── visualization/  # 6 Plotly dark-theme charts
├── dashboards/
│   └── app.py          # 4-view Dash dashboard
├── config/
│   ├── models.yaml
│   ├── etl.yaml
│   └── features.yaml
├── data/
│   ├── raw/            # Cached API responses (Parquet)
│   └── processed/      # DuckDB + engineered features
├── models/
│   ├── trained/        # Saved .joblib models
│   └── artifacts/      # SHAP artifacts
├── scripts/
│   └── push_to_github.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/INFURNI09/Formula1-AI.git
cd Formula1-AI

# 2. Install
pip install -r requirements.txt

# 3. Run ETL (downloads 2022–2024 F1 data)
python -c "from src.etl.pipeline import ETLPipeline; ETLPipeline(2022, 2024).run()"

# 4. Train models
python -c "from src.training.pipeline import run_training_pipeline; run_training_pipeline()"

# 5. Launch API
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 6. Launch Dashboard
python dashboards/app.py
# → http://localhost:8050
```

---

## 🗄 Data Sources

| Source | Data |
|---|---|
| **Ergast / Jolpica** | Results, qualifying, pit stops, standings (1950–2024) |
| **OpenF1 API** | Real-time telemetry, stints, weather (2023+) |
| **FastF1** | Lap-level timing, telemetry, car data |

All data is cached locally as Parquet and loaded into **DuckDB** (14 normalised tables).

---

## 🧠 Machine Learning

### Models Trained
- Gradient Boosting, Random Forest, Stacking Ensemble (sklearn baseline)
- XGBoost, LightGBM, CatBoost (when environment available)
- Optuna HPO — 20–100 trials per model per target
- Walk-forward time-series cross-validation (no data leakage)
- MLflow experiment tracking

### Features (35+)
Rolling averages, dynamic Elo ratings, career stats, circuit familiarity,
pit strategy efficiency, championship pressure, weather similarity,
podium/pole conversion rates, tyre degradation score.

---

## 🔬 Explainable AI

Every prediction ships with:
- SHAP waterfall plot (per driver)
- SHAP summary plot (global feature importance)
- SHAP dependence plots (top 3 features)
- Human-readable text explanation:
  > *"Verstappen — 79.7% win probability: boosted by recent win rate (+0.08),
  > pole conversion (+0.06); constrained by championship gap (−0.02)."*

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + available models |
| `POST` | `/predict/race-winner` | Race win probability |
| `POST` | `/predict/podium` | Podium probability |
| `POST` | `/predict/dnf` | DNF risk probability |
| `POST` | `/predict/qualifying` | Qualifying position |
| `POST` | `/predict/lap-time` | Finishing position |
| `GET` | `/championship/standings` | Championship model info |
| `GET` | `/simulation/season` | Monte Carlo season simulation |

Full interactive docs at `/docs` (Swagger UI).

---

## 📊 Dashboard Views

| View | Description |
|---|---|
| 🏁 Race Strategy | Tire degradation curves + pit window predictions |
| 📡 Telemetry | Speed / throttle / brake / RPM traces |
| 🏆 Championship | Dynamic Elo + Monte Carlo championship % |
| 🔬 XAI Inspector | Click driver → SHAP waterfall explanation |

---

## 🏗 Architecture

```
Ergast API ──┐
OpenF1 API ──┤→ ETL Pipeline → DuckDB → Feature Engineering
FastF1     ──┘                              ↓
                                    Training Pipeline
                                   (XGB/LGB/Cat + Optuna)
                                            ↓
                                  ┌─────────┼─────────┐
                             FastAPI    Dashboard    SHAP XAI
                              (REST)     (Dash)    (Waterfall)
```

---

## 📦 Requirements

```
pandas numpy polars duckdb pyarrow scikit-learn
xgboost lightgbm catboost optuna shap mlflow
plotly dash fastapi uvicorn pydantic joblib
fastf1 statsmodels scipy requests rich
```

---

## 🤝 Author

**INFURNI09** — [github.com/INFURNI09](https://github.com/INFURNI09)

Built with ❤️ and a passion for motorsport data engineering.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
