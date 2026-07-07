"""
dashboards/app.py
==================
Formula1-AI · Full Interactive Dash Dashboard
Run: python dashboards/app.py
"""
from __future__ import annotations
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

app = dash.Dash(
    __name__,
    title="Formula1-AI | Strategy Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

BG      = "#1D1D20"
BG_CARD = "#26262B"
TEXT    = "#fbfbff"
ACCENT  = "#A1C9F4"

_TABS_STYLE    = {"backgroundColor": BG, "borderBottom": f"2px solid {ACCENT}"}
_TAB_STYLE     = {"backgroundColor": BG, "color": "#909094", "border": "none",
                   "padding": "10px 20px", "fontFamily": "monospace"}
_TAB_SEL_STYLE = {**_TAB_STYLE, "color": TEXT, "borderBottom": f"2px solid {ACCENT}"}

app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "monospace"},
    children=[
        # ── Header ─────────────────────────────────────────────────────
        html.Div(
            style={"backgroundColor": "#12121A", "padding": "14px 24px",
                   "borderBottom": f"1px solid {ACCENT}", "display": "flex",
                   "alignItems": "center", "gap": "16px"},
            children=[
                html.Span("🏎", style={"fontSize": "28px"}),
                html.H1("FORMULA1-AI", style={"color": ACCENT, "margin": 0,
                                               "fontSize": "22px", "letterSpacing": "4px"}),
                html.Span("Strategy Intelligence Platform",
                          style={"color": "#909094", "fontSize": "13px", "marginLeft": "8px"}),
            ],
        ),

        # ── Tabs ───────────────────────────────────────────────────────
        dcc.Tabs(
            id="main-tabs",
            value="strategy",
            style=_TABS_STYLE,
            children=[
                dcc.Tab(label="🏁 Race Strategy",    value="strategy",
                        style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE),
                dcc.Tab(label="📡 Telemetry",        value="telemetry",
                        style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE),
                dcc.Tab(label="🏆 Championship",     value="championship",
                        style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE),
                dcc.Tab(label="🔬 XAI Inspector",    value="xai",
                        style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE),
            ],
        ),

        # ── Tab content ────────────────────────────────────────────────
        html.Div(id="tab-content", style={"padding": "20px"}),
    ],
)


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    card = {"backgroundColor": BG_CARD, "borderRadius": "8px",
            "padding": "20px", "marginBottom": "16px"}
    if tab == "strategy":
        return html.Div([
            html.Div("Tire Degradation Curves & Pit Windows", style=card),
            dcc.Graph(id="strategy-chart"),
        ])
    elif tab == "telemetry":
        return html.Div([
            html.Div("Speed / Throttle / Brake Traces (FastF1)", style=card),
            dcc.Graph(id="telemetry-chart"),
        ])
    elif tab == "championship":
        return html.Div([
            html.Div("Elo Ratings & Monte Carlo Championship Probabilities", style=card),
            dcc.Graph(id="championship-chart"),
        ])
    elif tab == "xai":
        return html.Div([
            html.Div("Click a driver to inspect SHAP waterfall explanation", style=card),
            dcc.Graph(id="xai-chart"),
        ])
    return html.Div("Select a tab above")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
