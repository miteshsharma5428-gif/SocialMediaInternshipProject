import json
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

df = pd.read_csv("data/forecast_results.csv")
rec_df = pd.read_csv("data/ortools_recommendations.csv")
with open("reports/metrics_ds02.json") as f:
    metrics_data = json.load(f)

benchmarks = metrics_data["model_accuracy_benchmarks"]
ens = benchmarks["Reconciled_Ensemble"]

app = Dash(__name__)
app.title = "DS-02: Nationwide Demand Forecasting & Inventory Optimization"

app.layout = html.Div(
    style={"fontFamily": "Inter, Arial, sans-serif", "backgroundColor": "#0f172a", "color": "#f8fafc", "padding": "28px", "minHeight": "100vh"},
    children=[
        html.H1("Nationwide Demand Forecasting & Inventory Optimization (DS-02)", style={"marginBottom": "6px", "fontSize": "26px"}),
        html.P(
            "M5 Hierarchical Retail Forecasting (State → Store → Item) • 80% Quantile Prediction Intervals • Google OR-Tools Optimizer",
            style={"color": "#94a3b8", "marginTop": "0", "marginBottom": "24px"}
        ),

        # Top KPI Cards
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "24px", "flexWrap": "wrap"}, children=[
            html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "10px", "flex": "1", "border": "1px solid #334155"}, children=[
                html.Div("Reconciled Ensemble WAPE", style={"color": "#94a3b8", "fontSize": "13px"}),
                html.Div(ens["WAPE_pct"], style={"fontSize": "28px", "fontWeight": "bold", "color": "#38bdf8", "marginTop": "4px"})
            ]),
            html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "10px", "flex": "1", "border": "1px solid #334155"}, children=[
                html.Div("Reconciled Ensemble RMSSE", style={"color": "#94a3b8", "fontSize": "13px"}),
                html.Div(f"{ens['RMSSE']:.4f}", style={"fontSize": "28px", "fontWeight": "bold", "color": "#34d399", "marginTop": "4px"})
            ]),
            html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "10px", "flex": "1", "border": "1px solid #334155"}, children=[
                html.Div("OR-Tools Solver Status", style={"color": "#94a3b8", "fontSize": "13px"}),
                html.Div(metrics_data["solver_status"], style={"fontSize": "28px", "fontWeight": "bold", "color": "#a78bfa", "marginTop": "4px"})
            ]),
            html.Div(style={"backgroundColor": "#1e293b", "padding": "18px", "borderRadius": "10px", "flex": "1", "border": "1px solid #334155"}, children=[
                html.Div("Optimal 7-Day Network Order", style={"color": "#94a3b8", "fontSize": "13px"}),
                html.Div(f"{metrics_data['total_recommended_order_units']:,} units", style={"fontSize": "28px", "fontWeight": "bold", "color": "#fbbf24", "marginTop": "4px"})
            ]),
        ]),

        # Hierarchical Dropdown Selectors (State -> Store -> Item)
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "20px", "flexWrap": "wrap"}, children=[
            html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                html.Label("1. Select State:", style={"fontWeight": "bold", "fontSize": "13px", "color": "#cbd5e1"}),
                dcc.Dropdown(
                    id="state-dropdown",
                    options=[{"label": s, "value": s} for s in sorted(df["state_id"].unique())],
                    value="CA",
                    clearable=False,
                    style={"color": "#0f172a", "marginTop": "6px"}
                )
            ]),
            html.Div(style={"flex": "1", "minWidth": "240px"}, children=[
                html.Label("2. Select Store:", style={"fontWeight": "bold", "fontSize": "13px", "color": "#cbd5e1"}),
                dcc.Dropdown(
                    id="store-dropdown",
                    clearable=False,
                    style={"color": "#0f172a", "marginTop": "6px"}
                )
            ]),
            html.Div(style={"flex": "1", "minWidth": "260px"}, children=[
                html.Label("3. Select Product SKU:", style={"fontWeight": "bold", "fontSize": "13px", "color": "#cbd5e1"}),
                dcc.Dropdown(
                    id="item-dropdown",
                    options=[{"label": i, "value": i} for i in sorted(df["item_id"].unique())],
                    value=sorted(df["item_id"].unique())[0],
                    clearable=False,
                    style={"color": "#0f172a", "marginTop": "6px"}
                )
            ]),
        ]),

        # Interactive Charts Row
        html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "28px", "flexWrap": "wrap"}, children=[
            html.Div(style={"flex": "2", "minWidth": "480px", "backgroundColor": "#1e293b", "padding": "12px", "borderRadius": "10px"}, children=[
                dcc.Graph(id="forecast-graph")
            ]),
            html.Div(style={"flex": "1", "minWidth": "340px", "backgroundColor": "#1e293b", "padding": "12px", "borderRadius": "10px"}, children=[
                dcc.Graph(id="benchmark-graph")
            ]),
        ]),

        # OR-Tools Inventory Replenishment Table
        html.H3("OR-Tools Optimal Replenishment Order Recommendations (Next 7 Days)", style={"marginBottom": "12px"}),
        dash_table.DataTable(
            data=rec_df.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in rec_df.columns],
            style_header={"backgroundColor": "#1e293b", "color": "#38bdf8", "fontWeight": "bold", "border": "1px solid #334155"},
            style_cell={"backgroundColor": "#0f172a", "color": "#f8fafc", "border": "1px solid #334155", "padding": "10px", "textAlign": "left", "fontSize": "13px"},
            style_data_conditional=[
                {"if": {"column_id": "optimal_order_qty"}, "fontWeight": "bold", "color": "#fbbf24"}
            ]
        )
    ]
)

@app.callback(
    [Output("store-dropdown", "options"), Output("store-dropdown", "value")],
    Input("state-dropdown", "value")
)
def update_store_options(selected_state):
    stores = sorted(df[df["state_id"] == selected_state]["store_id"].unique())
    options = [{"label": s, "value": s} for s in stores]
    return options, stores[0]

@app.callback(
    [Output("forecast-graph", "figure"), Output("benchmark-graph", "figure")],
    [Input("store-dropdown", "value"), Input("item-dropdown", "value")]
)
def update_charts(selected_store, selected_item):
    sub = df[(df["store_id"] == selected_store) & (df["item_id"] == selected_item)].sort_values("date")

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([sub["date"], sub["date"][::-1]]),
        y=pd.concat([sub["forecast_upper_90"], sub["forecast_lower_10"][::-1]]),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.22)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Prediction Interval (P10–P90)"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=sub["date"], y=sub["sales"],
        mode="lines+markers", name="Actual Sales",
        line=dict(color="#f8fafc", width=2)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=sub["date"], y=sub["forecast_point"],
        mode="lines+markers", name="Reconciled Forecast",
        line=dict(color="#38bdf8", width=2.5, dash="dash")
    ))
    fig_forecast.update_layout(
        title=f"28-Day Demand Forecast vs. Actuals ({selected_store} | {selected_item})",
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        xaxis_title="Date",
        yaxis_title="Daily Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    model_names = [k.replace("_", " ") for k in benchmarks.keys()]
    wape_values = [round(v["WAPE"] * 100, 2) for v in benchmarks.values()]
    colors = ["#64748b", "#64748b", "#38bdf8", "#a78bfa", "#34d399"]

    fig_bench = go.Figure(data=[
        go.Bar(x=model_names, y=wape_values, marker_color=colors, text=[f"{w}%" for w in wape_values], textposition="auto")
    ])
    fig_bench.update_layout(
        title="Model Error Comparison (Lower WAPE % is Better)",
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        yaxis_title="WAPE (%)"
    )

    return fig_forecast, fig_bench

if __name__ == "__main__":
    app.run(debug=False, port=8050)
