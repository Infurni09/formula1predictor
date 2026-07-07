"""
dashboards/app.py
==================
Formula1-AI  ·  Interactive Dash Dashboard
  Run: python dashboards/app.py
  Open: http://localhost:8050

  4 tabs:
    1. Race Strategy Center
    2. Telemetry Overlay
    3. Championship Tracker
    4. XAI Inspector
"""
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go

# ── Color system ───────────────────────────────────────────────────────────
BG      = "#1D1D20"
BG_CARD = "#28282C"
TEXT    = "#fbfbff"
SECOND  = "#909094"
ACCENT  = "#ffd400"
GRID    = "#333338"

# ── App ────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Formula1-AI Strategy Platform",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.index_string = \'\'\'
<!DOCTYPE html>
<html>
  <head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #1D1D20; font-family: Inter, Arial, sans-serif; color: #fbfbff; }
    .header { background: #111114; border-bottom: 2px solid #ffd400; padding: 14px 28px;
              display: flex; align-items: center; gap: 16px; }
    .header h1 { font-size: 1.4rem; color: #ffd400; letter-spacing: 0.06em; }
    .header p  { font-size: 0.8rem; color: #909094; }
    .tab-content { padding: 20px; }
    .card { background: #28282C; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .kpi-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    .kpi { background: #2F2F35; border-radius: 6px; padding: 14px 20px; flex: 1; min-width: 160px; }
    .kpi-label { font-size: 0.7rem; color: #909094; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #ffd400; margin-top: 4px; }
    .custom-tabs .tab { background: #2F2F35 !important; color: #909094 !important;
                        border: none !important; border-bottom: 2px solid #333 !important; }
    .custom-tabs .tab--selected { color: #ffd400 !important; border-bottom: 2px solid #ffd400 !important; }
  </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
\'\'\'

app.layout = html.Div([
    # ── Header ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("🏎 FORMULA1-AI"),
            html.P("Strategy Intelligence Platform — Production ML · Monte Carlo · SHAP XAI"),
        ]),
        html.Div([
            html.Span("● LIVE", style={"color": "#17b26a", "fontWeight": "700", "fontSize": "0.75rem"}),
            html.Span(" 2024 Season Analysis", style={"color": "#909094", "fontSize": "0.75rem", "marginLeft": "8px"}),
        ]),
    ], className="header"),

    # ── KPI Strip ────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("RACE WIN PROB", className="kpi-label"),
            html.Div("64.1%", className="kpi-value"),
        ], className="kpi"),
        html.Div([
            html.Div("PODIUM PROB", className="kpi-label"),
            html.Div("91.2%", className="kpi-value"),
        ], className="kpi"),
        html.Div([
            html.Div("DNF RISK", className="kpi-label"),
            html.Div("3.6%", className="kpi-value", style={"color": "#17b26a"}),
        ], className="kpi"),
        html.Div([
            html.Div("SAFETY CAR", className="kpi-label"),
            html.Div("18%", className="kpi-value", style={"color": "#FFB482"}),
        ], className="kpi"),
        html.Div([
            html.Div("QUALIFYING", className="kpi-label"),
            html.Div("P1 → 1.2", className="kpi-value"),
        ], className="kpi"),
        html.Div([
            html.Div("ELO RATING", className="kpi-label"),
            html.Div("1,847", className="kpi-value"),
        ], className="kpi"),
    ], className="kpi-row", style={"padding": "16px 20px 0 20px"}),

    # ── Tabs ─────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="strategy", className="custom-tabs", children=[
        dcc.Tab(label="🏎 Race Strategy", value="strategy"),
        dcc.Tab(label="📡 Telemetry",     value="telemetry"),
        dcc.Tab(label="🏆 Championship",  value="championship"),
        dcc.Tab(label="🔬 XAI Inspector", value="xai"),
    ]),
    html.Div(id="tab-content", className="tab-content"),

], style={"backgroundColor": BG, "minHeight": "100vh"})


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "strategy":
        return html.Div([
            html.P("Comparative tyre degradation curves with optimal pit windows highlighted.",
                   style={"color": SECOND, "marginBottom": "12px", "fontSize": "0.85rem"}),
            html.Div(dcc.Graph(id="strategy-graph", style={"height": "500px"}), className="card"),
        ])
    elif tab == "telemetry":
        return html.Div([
            html.P("Speed, throttle, and brake traces for head-to-head driver comparison.",
                   style={"color": SECOND, "marginBottom": "12px", "fontSize": "0.85rem"}),
            html.Div(dcc.Graph(id="telemetry-graph", style={"height": "540px"}), className="card"),
        ])
    elif tab == "championship":
        return html.Div([
            html.P("Dynamic Elo ratings + Monte Carlo championship win probability distribution.",
                   style={"color": SECOND, "marginBottom": "12px", "fontSize": "0.85rem"}),
            html.Div(dcc.Graph(id="championship-graph", style={"height": "500px"}), className="card"),
        ])
    elif tab == "xai":
        return html.Div([
            html.P("SHAP-based explainable AI — click any driver to drill into their prediction factors.",
                   style={"color": SECOND, "marginBottom": "12px", "fontSize": "0.85rem"}),
            html.Div(dcc.Graph(id="xai-graph", style={"height": "480px"}), className="card"),
        ])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)