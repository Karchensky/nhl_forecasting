"""NHL Goal Probability Model — Streamlit Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from database.db_client import get_engine, init_db
from models.evaluation import (
    calibration_table,
    compare_models,
    compute_core_metrics,
    evaluate_model,
    lift_table,
)

st.set_page_config(
    page_title="NHL Goal Probability Model",
    page_icon="🏒",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_predictions():
    engine = get_engine()
    query = """
        SELECT
            mo.player_id, mo.game_id, mo.model_version,
            mo.predicted_probability, mo.created_at,
            p.full_name as player_name, p.position,
            g.game_date, g.home_team_id, g.away_team_id, g.game_state,
            ht.abbreviation as home_team, at2.abbreviation as away_team,
            pt.abbreviation as player_team
        FROM model_outputs mo
        JOIN players p ON mo.player_id = p.player_id
        JOIN games g ON mo.game_id = g.game_id
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at2 ON g.away_team_id = at2.team_id
        LEFT JOIN teams pt ON p.current_team_id = pt.team_id
        ORDER BY mo.predicted_probability DESC
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300)
def load_odds():
    engine = get_engine()
    query = """
        SELECT
            o.player_id, o.game_id, o.sportsbook, o.market,
            o.american_odds, o.implied_probability, o.retrieved_at,
            p.full_name as player_name,
            g.game_date
        FROM odds o
        JOIN players p ON o.player_id = p.player_id
        JOIN games g ON o.game_id = g.game_id
        ORDER BY o.retrieved_at DESC
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300)
def load_historical_results():
    engine = get_engine()
    query = """
        SELECT
            pgs.player_id, pgs.game_id, pgs.goals, pgs.shots, pgs.toi_seconds,
            p.full_name, p.position,
            g.game_date, g.season
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        JOIN games g ON pgs.game_id = g.game_id
        WHERE g.game_state IN ('FINAL', 'OFF')
          AND (p.position IS NULL OR p.position != 'G')
        ORDER BY g.game_date DESC
    """
    return pd.read_sql(query, engine)


def predictions_view():
    st.header("Model Predictions")

    preds = load_predictions()
    if preds.empty:
        st.info("No predictions available yet. Run the model training pipeline first.")
        return

    model_versions = preds["model_version"].unique().tolist()
    selected_model = st.selectbox("Model Version", model_versions,
                                  index=model_versions.index("lightgbm")
                                  if "lightgbm" in model_versions else 0)

    filtered = preds[preds["model_version"] == selected_model].copy()

    if "game_date" in filtered.columns:
        filtered["game_date"] = pd.to_datetime(filtered["game_date"])
        dates = sorted(filtered["game_date"].dt.date.unique(), reverse=True)
        if dates:
            selected_date = st.selectbox("Game Date", dates)
            filtered = filtered[filtered["game_date"].dt.date == selected_date]

    min_prob = st.slider("Minimum Probability (%)", 0, 100, 10) / 100.0
    filtered = filtered[filtered["predicted_probability"] >= min_prob]

    display_cols = ["player_name", "player_team", "home_team", "away_team",
                    "predicted_probability", "position", "game_date"]
    display = filtered[[c for c in display_cols if c in filtered.columns]].copy()
    if "predicted_probability" in display.columns:
        display["predicted_probability"] = (display["predicted_probability"] * 100).round(1)
        display = display.rename(columns={"predicted_probability": "Goal Prob (%)"})

    st.dataframe(display, use_container_width=True, hide_index=True)

    if not filtered.empty:
        fig = px.histogram(filtered, x="predicted_probability", nbins=30,
                           title="Prediction Distribution")
        fig.update_xaxes(title="Predicted Probability")
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)


def value_view():
    st.header("Value Bets (+EV Opportunities)")

    preds = load_predictions()
    odds = load_odds()

    if preds.empty:
        st.info("No predictions available.")
        return
    if odds.empty:
        st.info(
            "No odds data available. Configure your Odds API key and run "
            "the odds scraper to see value opportunities."
        )
        return

    model_versions = preds["model_version"].unique().tolist()
    selected_model = st.selectbox("Model", model_versions,
                                  index=model_versions.index("lightgbm")
                                  if "lightgbm" in model_versions else 0,
                                  key="value_model")

    merged = preds[preds["model_version"] == selected_model].merge(
        odds[["player_id", "game_id", "sportsbook", "american_odds",
              "implied_probability"]],
        on=["player_id", "game_id"],
        how="inner",
    )

    if merged.empty:
        st.warning("No matching predictions + odds found.")
        return

    merged["edge"] = merged["predicted_probability"] - merged["implied_probability"]
    merged["edge_pct"] = (merged["edge"] * 100).round(2)

    min_edge = st.slider("Minimum Edge (%)", -20, 30, 2, key="min_edge") / 100.0
    value_bets = merged[merged["edge"] >= min_edge].sort_values("edge", ascending=False)

    display = value_bets[[
        "player_name", "player_team", "home_team", "away_team",
        "predicted_probability", "implied_probability", "edge_pct",
        "american_odds", "sportsbook", "game_date",
    ]].copy()
    display["predicted_probability"] = (display["predicted_probability"] * 100).round(1)
    display["implied_probability"] = (display["implied_probability"] * 100).round(1)
    display = display.rename(columns={
        "predicted_probability": "Model %",
        "implied_probability": "Market %",
        "edge_pct": "Edge %",
        "american_odds": "Odds",
    })

    n_positive = len(value_bets[value_bets["edge"] > 0])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(merged))
    col2.metric("+EV Opportunities", n_positive)
    col3.metric("Avg Edge", f"{value_bets['edge_pct'].mean():.1f}%" if not value_bets.empty else "N/A")

    st.dataframe(display, use_container_width=True, hide_index=True)


def diagnostics_view():
    st.header("Model Diagnostics")

    results = load_historical_results()
    preds = load_predictions()

    if results.empty:
        st.info("No historical data available for diagnostics.")
        return

    if not preds.empty:
        model_versions = preds["model_version"].unique().tolist()
        selected_model = st.selectbox("Model", model_versions, key="diag_model")

        merged = preds[preds["model_version"] == selected_model].merge(
            results[["player_id", "game_id", "goals"]],
            on=["player_id", "game_id"],
            how="inner",
        )

        if not merged.empty:
            y_true = (merged["goals"] >= 1).astype(int).values
            y_prob = merged["predicted_probability"].values

            metrics = compute_core_metrics(y_true, y_prob)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Log Loss", f"{metrics['log_loss']:.4f}")
            col2.metric("Brier Score", f"{metrics['brier_score']:.4f}")
            col3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            col4.metric("Base Rate", f"{metrics['base_rate']:.3f}")

            st.subheader("Calibration Curve")
            cal = calibration_table(y_true, y_prob)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=cal["predicted_mean"], y=cal["actual_rate"],
                mode="lines+markers", name="Model",
            ))
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines", name="Perfect",
                line=dict(dash="dash", color="gray"),
            ))
            fig_cal.update_layout(
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Actual Scoring Rate",
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            st.subheader("Calibration Table")
            cal_display = cal.copy()
            cal_display["actual_rate"] = (cal_display["actual_rate"] * 100).round(2)
            cal_display["predicted_mean"] = (cal_display["predicted_mean"] * 100).round(2)
            st.dataframe(cal_display, use_container_width=True, hide_index=True)

            st.subheader("Lift Chart")
            lift = lift_table(y_true, y_prob)
            fig_lift = px.bar(lift, x="decile", y="lift",
                              title="Lift by Decile (1 = highest predicted probability)")
            fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig_lift.update_xaxes(title="Decile")
            fig_lift.update_yaxes(title="Lift vs. Base Rate")
            st.plotly_chart(fig_lift, use_container_width=True)

            st.subheader("Cumulative Gains")
            fig_gains = px.line(lift, x="decile", y="cumulative_actual",
                                title="Cumulative Gains Chart")
            fig_gains.update_xaxes(title="Decile")
            fig_gains.update_yaxes(title="Cumulative % of Goals Captured")
            st.plotly_chart(fig_gains, use_container_width=True)
        else:
            st.warning("No overlapping predictions and results for evaluation.")

    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Player-Game Records", len(results))
        if "season" in results.columns:
            season_counts = results.groupby("season").size().reset_index(name="count")
            st.dataframe(season_counts, hide_index=True)
    with col2:
        if not results.empty:
            results["scored"] = (results["goals"] >= 1).astype(int)
            base_rate = results["scored"].mean()
            st.metric("Overall Scoring Rate", f"{base_rate * 100:.1f}%")


def main():
    st.title("🏒 NHL Goal Probability Model")

    init_db()

    tab1, tab2, tab3 = st.tabs(["Predictions", "Value Bets", "Diagnostics"])

    with tab1:
        predictions_view()
    with tab2:
        value_view()
    with tab3:
        diagnostics_view()


if __name__ == "__main__":
    main()
