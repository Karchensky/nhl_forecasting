"""NHL Goal Probability Model — Streamlit Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text as sa_text

from database.db_client import get_engine, init_db
from scrapers.external.odds_api import format_american_line, probability_to_american
from models.evaluation import (
    calibration_table,
    compute_core_metrics,
    lift_table,
)

st.set_page_config(
    page_title="NHL Goal Probability Model",
    layout="wide",
)

MODEL_DISPLAY = {
    "logistic_regression": ("LR %", "LR edge %"),
    "lightgbm": ("LightGBM %", "LightGBM edge %"),
    "xgboost": ("XGBoost %", "XGBoost edge %"),
}


def _primary_market_rows(odds_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, game_id), same logic as pre-refactor inner merge.

    Prefer **FanDuel** (matches ``settings.yaml`` bookmakers); if absent, use the
    most recently ``retrieved_at`` row. Do **not** average implied % across books
    (that can blur sharp vs soft lines and confuses verification against a single posted price).
    """
    if odds_df.empty:
        return pd.DataFrame(
            columns=["player_id", "game_id", "market_implied", "market_american"]
        )
    o = odds_df.copy()
    o["_book"] = o["sportsbook"].astype(str).str.lower()
    fd = o[o["_book"] == "fanduel"]
    use = fd if not fd.empty else o
    use = use.sort_values("retrieved_at", ascending=False)
    pick = use.drop_duplicates(subset=["player_id", "game_id"], keep="first")
    return pick.assign(
        market_implied=pick["implied_probability"],
        market_american=pick["american_odds"],
    )[["player_id", "game_id", "market_implied", "market_american"]]


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


def odds_table_row_count() -> int | None:
    """Raw count in ``odds`` (helps debug empty API / upsert issues)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return int(conn.execute(sa_text("SELECT COUNT(*) FROM odds")).scalar() or 0)
    except Exception:
        return None


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


def _series_to_iso_date(s: pd.Series) -> pd.Series:
    """Normalize to date-only strings yyyy-mm-dd for display."""
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def opportunities_view():
    """Predictions + market: filters, multi-model edges, ranks."""
    st.header("Opportunities")

    preds = load_predictions()
    odds = load_odds()

    if preds.empty:
        st.info("No predictions available yet. Run the model training pipeline first.")
        return

    work = preds.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    dates = sorted(work["game_date"].dropna().dt.date.unique().tolist(), reverse=True)
    if not dates:
        st.warning("No game dates in predictions.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_date = st.selectbox(
            "Date",
            options=dates,
            format_func=lambda d: d.isoformat(),
            key="opp_date",
        )
    work = work[work["game_date"].dt.date == selected_date].copy()
    if work.empty:
        st.info("No rows for the selected date.")
        return

    games_df = (
        work[["game_id", "away_team", "home_team"]]
        .drop_duplicates()
        .sort_values(["home_team", "away_team", "game_id"])
    )
    base_lbl = (
        games_df["away_team"].astype(str) + " @ " + games_df["home_team"].astype(str)
    )
    if base_lbl.duplicated().any():
        games_df = games_df.assign(label=base_lbl + " (" + games_df["game_id"].astype(str) + ")")
    else:
        games_df = games_df.assign(label=base_lbl)

    game_options = ["All games"] + games_df["label"].tolist()
    with c2:
        game_pick = st.selectbox("Game", options=game_options, key="opp_game")
    if game_pick != "All games":
        gid = int(games_df.loc[games_df["label"] == game_pick, "game_id"].iloc[0])
        work = work[work["game_id"] == gid]

    teams = sorted(work["player_team"].dropna().unique().tolist())
    with c3:
        team_pick = st.multiselect(
            "Player team",
            options=teams,
            default=teams,
            key="opp_team",
        )
    if team_pick:
        work = work[work["player_team"].isin(team_pick)]

    all_models = sorted(work["model_version"].unique().tolist())
    with c4:
        model_pick = st.multiselect(
            "Models",
            options=all_models,
            default=all_models,
            format_func=lambda m: {
                "logistic_regression": "Logistic regression",
                "lightgbm": "LightGBM",
                "xgboost": "XGBoost",
            }.get(m, m),
            key="opp_models",
        )
    if not model_pick:
        st.info("Select at least one model.")
        return

    work = work[work["model_version"].isin(model_pick)]

    id_cols = [
        "player_id",
        "game_id",
        "player_name",
        "player_team",
        "home_team",
        "away_team",
        "game_date",
    ]
    wide = work.pivot_table(
        index=[c for c in id_cols if c in work.columns],
        columns="model_version",
        values="predicted_probability",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Market: single book row per (player, game) — FanDuel first, else latest snapshot
    odds_f = odds.copy()
    if not odds_f.empty:
        odds_f["game_date"] = pd.to_datetime(odds_f["game_date"], errors="coerce")
        odds_f = odds_f[odds_f["game_date"].dt.date == selected_date]
        if not odds_f.empty:
            odds_pick = _primary_market_rows(odds_f)
            wide = wide.merge(odds_pick, on=["player_id", "game_id"], how="left")
        else:
            wide["market_implied"] = np.nan
            wide["market_american"] = np.nan
    else:
        n_db = odds_table_row_count()
        if n_db == 0:
            st.caption(
                "No odds in the database — edge and market columns will be empty. "
                "Set `ODDS_API_KEY` and run the daily job to ingest lines."
            )
        wide["market_implied"] = np.nan
        wide["market_american"] = np.nan

    def _line_from_p(p) -> str:
        if pd.isna(p):
            return "—"
        try:
            return format_american_line(probability_to_american(float(p)))
        except Exception:
            return "—"

    edge_cols = []
    for m in model_pick:
        if m not in wide.columns:
            continue
        pct_col, edge_col = MODEL_DISPLAY.get(m, (f"{m} %", f"{m} edge %"))
        wide[pct_col] = (wide[m] * 100).round(1)
        wide[edge_col] = np.where(
            wide["market_implied"].notna(),
            ((wide[m] - wide["market_implied"]) * 100).round(2),
            np.nan,
        )
        edge_cols.append(edge_col)

    wide["Market line"] = wide["market_american"].map(
        lambda x: format_american_line(int(x)) if pd.notna(x) else "—"
    )
    wide["Market %"] = np.where(
        wide["market_implied"].notna(),
        (wide["market_implied"] * 100).round(1),
        np.nan,
    )

    model_prob_cols = [m for m in model_pick if m in wide.columns]
    if model_prob_cols:
        wide["_avg_model_p"] = wide[model_prob_cols].mean(axis=1, skipna=True)
    else:
        wide["_avg_model_p"] = np.nan

    wide["Model line (avg)"] = wide["_avg_model_p"].map(_line_from_p)

    if edge_cols:
        mat = wide[edge_cols].to_numpy(dtype=float)
        wide["_best_edge"] = np.nanmax(
            np.where(np.isfinite(mat), mat, np.nan), axis=1
        )
    else:
        wide["_best_edge"] = np.nan

    wide["Edge rank (overall)"] = (
        wide["_best_edge"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    wide["Edge rank (team)"] = (
        wide.groupby("player_team", dropna=False)["_best_edge"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    wide = wide.sort_values("_best_edge", ascending=False, na_position="last")

    st.markdown(f"**Slate:** `{selected_date.isoformat()}`")

    out_cols = (
        ["Edge rank (overall)", "Edge rank (team)", "player_name", "player_team", "home_team", "away_team"]
        + [MODEL_DISPLAY[m][0] for m in model_pick if m in wide.columns]
        + ["Market line", "Market %"]
        + [MODEL_DISPLAY[m][1] for m in model_pick if m in wide.columns]
        + ["Model line (avg)"]
    )
    out_cols = [c for c in out_cols if c in wide.columns]
    display = wide[out_cols].copy()
    display = display.rename(
        columns={
            "player_name": "Player",
            "player_team": "Team",
            "home_team": "Home",
            "away_team": "Away",
        }
    )

    st.dataframe(display, width="stretch", hide_index=True)

    with st.expander("Prediction distributions (selected filters)"):
        if model_prob_cols:
            fig = go.Figure()
            for m in model_prob_cols:
                fig.add_trace(
                    go.Histogram(
                        x=wide[m].dropna(),
                        name=m.replace("_", " "),
                        opacity=0.55,
                        nbinsx=30,
                    )
                )
            fig.update_layout(
                title="Predicted probability by model",
                barmode="overlay",
                xaxis_title="Probability",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, width="stretch")


def diagnostics_view():
    st.header("Model Diagnostics")

    results = load_historical_results()
    preds = load_predictions()

    if results.empty:
        st.info("No historical data available for diagnostics.")
        return

    if "game_date" in results.columns:
        results = results.copy()
        results["game_date"] = _series_to_iso_date(results["game_date"])

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
            fig_cal.add_trace(
                go.Scatter(
                    x=cal["predicted_mean"],
                    y=cal["actual_rate"],
                    mode="lines+markers",
                    name="Model",
                )
            )
            fig_cal.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect",
                    line=dict(dash="dash", color="gray"),
                )
            )
            fig_cal.update_layout(
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Actual Scoring Rate",
            )
            st.plotly_chart(fig_cal, width="stretch")

            st.subheader("Calibration Table")
            cal_display = cal.copy()
            cal_display["actual_rate"] = (cal_display["actual_rate"] * 100).round(2)
            cal_display["predicted_mean"] = (cal_display["predicted_mean"] * 100).round(2)
            st.dataframe(cal_display, width="stretch", hide_index=True)

            st.subheader("Lift Chart")
            lift = lift_table(y_true, y_prob)
            fig_lift = px.bar(
                lift,
                x="decile",
                y="lift",
                title="Lift by Decile (1 = highest predicted probability)",
            )
            fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig_lift.update_xaxes(title="Decile")
            fig_lift.update_yaxes(title="Lift vs. Base Rate")
            st.plotly_chart(fig_lift, width="stretch")

            st.subheader("Cumulative Gains")
            fig_gains = px.line(
                lift,
                x="decile",
                y="cumulative_actual",
                title="Cumulative Gains Chart",
            )
            fig_gains.update_xaxes(title="Decile")
            fig_gains.update_yaxes(title="Cumulative % of Goals Captured")
            st.plotly_chart(fig_gains, width="stretch")
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
            results_sc = results.copy()
            results_sc["scored"] = (results_sc["goals"] >= 1).astype(int)
            base_rate = results_sc["scored"].mean()
            st.metric("Overall Scoring Rate", f"{base_rate * 100:.1f}%")


def main():
    st.title("NHL Goal Probability Model")

    init_db()

    tab1, tab2 = st.tabs(["Opportunities", "Diagnostics"])

    with tab1:
        opportunities_view()
    with tab2:
        diagnostics_view()


if __name__ == "__main__":
    main()
