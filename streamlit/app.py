"""NHL Goal Probability Model -- Streamlit Dashboard."""

import sys
from datetime import date
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

# Suppress default Streamlit top padding
st.markdown(
    "<style>div.block-container{padding-top:1.5rem;}</style>",
    unsafe_allow_html=True,
)

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
MODEL_DISPLAY = {
    "logistic_regression": ("LR %", "LR edge %"),
    "lightgbm": ("LGB %", "LGB edge %"),
    "xgboost": ("XGB %", "XGB edge %"),
}

TRAIN_SEASONS = {20202021, 20212022, 20222023, 20232024}
VAL_SEASON = 20242025
TEST_SEASON = 20252026


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _primary_market_rows(odds_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, game_id) -- FanDuel preferred, else latest."""
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
            g.game_date, g.season, g.home_team_id, g.away_team_id, g.game_state,
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


def _line_from_p(p) -> str:
    if pd.isna(p):
        return "--"
    try:
        return format_american_line(probability_to_american(float(p)))
    except Exception:
        return "--"


def _split_label(season: int) -> str:
    if season in TRAIN_SEASONS:
        return "TRAIN"
    if season == VAL_SEASON:
        return "VAL"
    if season == TEST_SEASON:
        return "TEST"
    return "OTHER"


# ---------------------------------------------------------------------------
# Tab 1 -- Opportunities
# ---------------------------------------------------------------------------

def opportunities_view():
    preds = load_predictions()
    odds = load_odds()

    if preds.empty:
        st.info("No predictions available. Run the training pipeline first.")
        return

    work = preds.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    dates = sorted(work["game_date"].dropna().dt.date.unique().tolist(), reverse=True)
    if not dates:
        st.warning("No game dates in predictions.")
        return

    # -- Filters row --
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        today = date.today()
        default_idx = 0
        if today in dates:
            default_idx = dates.index(today)
        selected_date = st.selectbox(
            "Date",
            options=dates,
            index=default_idx,
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
    base_lbl = games_df["away_team"].astype(str) + " @ " + games_df["home_team"].astype(str)
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
        team_pick = st.multiselect("Team", options=teams, default=teams, key="opp_team")
    if team_pick:
        work = work[work["player_team"].isin(team_pick)]

    all_models = sorted(work["model_version"].unique().tolist())
    with c4:
        model_pick = st.multiselect(
            "Models",
            options=all_models,
            default=all_models,
            format_func=lambda m: MODEL_NAMES.get(m, m),
            key="opp_models",
        )
    if not model_pick:
        st.info("Select at least one model.")
        return
    work = work[work["model_version"].isin(model_pick)]

    # -- Pivot to wide (one row per player-game) --
    id_cols = [
        "player_id", "game_id", "player_name", "player_team",
        "home_team", "away_team", "game_date", "position",
    ]
    wide = work.pivot_table(
        index=[c for c in id_cols if c in work.columns],
        columns="model_version",
        values="predicted_probability",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # -- Merge market odds --
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
        wide["market_implied"] = np.nan
        wide["market_american"] = np.nan

    # -- Compute display columns --
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

    wide["FD Line"] = wide["market_american"].map(
        lambda x: format_american_line(int(x)) if pd.notna(x) else "--"
    )
    wide["FD %"] = np.where(
        wide["market_implied"].notna(),
        (wide["market_implied"] * 100).round(1),
        np.nan,
    )

    model_prob_cols = [m for m in model_pick if m in wide.columns]
    if model_prob_cols:
        wide["_avg_model_p"] = wide[model_prob_cols].mean(axis=1, skipna=True)
    else:
        wide["_avg_model_p"] = np.nan

    wide["Avg model line"] = wide["_avg_model_p"].map(_line_from_p)

    if edge_cols:
        mat = wide[edge_cols].to_numpy(dtype=float)
        wide["_best_edge"] = np.nanmax(
            np.where(np.isfinite(mat), mat, np.nan), axis=1
        )
    else:
        wide["_best_edge"] = np.nan

    wide["Rank"] = (
        wide["_best_edge"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    wide["Team rank"] = (
        wide.groupby("player_team", dropna=False)["_best_edge"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    wide = wide.sort_values("_best_edge", ascending=False, na_position="last")

    # -- Summary KPIs --
    has_market = wide["market_implied"].notna()
    n_with_odds = int(has_market.sum())
    n_positive_ev = int((wide["_best_edge"] > 0).sum()) if edge_cols else 0
    avg_edge = float(wide.loc[wide["_best_edge"] > 0, "_best_edge"].mean()) if n_positive_ev else 0.0
    top_edge = float(wide["_best_edge"].max()) if edge_cols and not wide["_best_edge"].isna().all() else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Slate", selected_date.isoformat())
    k2.metric("Players w/ odds", n_with_odds)
    k3.metric("+EV plays", n_positive_ev)
    k4.metric("Avg edge (pp)", f"{avg_edge:+.1f}" if avg_edge else "--")
    k5.metric("Best edge (pp)", f"{top_edge:+.1f}" if top_edge else "--")

    # -- Main table --
    out_cols = (
        ["Rank", "Team rank", "player_name", "position", "player_team",
         "home_team", "away_team"]
        + [MODEL_DISPLAY[m][0] for m in model_pick if m in wide.columns]
        + ["FD Line", "FD %"]
        + [MODEL_DISPLAY[m][1] for m in model_pick if m in wide.columns]
        + ["Avg model line"]
    )
    out_cols = [c for c in out_cols if c in wide.columns]
    display = wide[out_cols].copy()
    display = display.rename(columns={
        "player_name": "Player",
        "position": "Pos",
        "player_team": "Team",
        "home_team": "Home",
        "away_team": "Away",
    })

    st.dataframe(display, width='stretch', hide_index=True, height=600)

    # -- Histogram expander --
    with st.expander("Prediction distributions"):
        if model_prob_cols:
            fig = go.Figure()
            for m in model_prob_cols:
                fig.add_trace(go.Histogram(
                    x=wide[m].dropna(),
                    name=MODEL_NAMES.get(m, m),
                    opacity=0.55,
                    nbinsx=30,
                ))
            fig.update_layout(
                barmode="overlay",
                xaxis_title="Probability",
                yaxis_title="Count",
                margin=dict(t=30, b=30),
                height=300,
            )
            st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab 2 -- Model Diagnostics
# ---------------------------------------------------------------------------

def diagnostics_view():
    results = load_historical_results()
    preds = load_predictions()

    if results.empty:
        st.info("No historical data for diagnostics.")
        return
    if preds.empty:
        st.info("No predictions for diagnostics.")
        return

    results = results.copy()
    results["game_date"] = pd.to_datetime(results["game_date"], errors="coerce")
    preds = preds.copy()
    preds["game_date"] = pd.to_datetime(preds["game_date"], errors="coerce")

    # -- Season / split filter --
    available_seasons = sorted(preds["season"].dropna().unique().tolist())
    season_labels = {s: f"{s} ({_split_label(s)})" for s in available_seasons}

    c1, c2 = st.columns([1, 3])
    with c1:
        default_seasons = [s for s in available_seasons if s not in TRAIN_SEASONS]
        if not default_seasons:
            default_seasons = available_seasons
        sel_seasons = st.multiselect(
            "Season(s)",
            options=available_seasons,
            default=default_seasons,
            format_func=lambda s: season_labels.get(s, str(s)),
            key="diag_seasons",
        )
    if not sel_seasons:
        st.info("Select at least one season.")
        return

    preds_f = preds[preds["season"].isin(sel_seasons)]
    results_f = results[results["season"].isin(sel_seasons)]

    split_note = ", ".join(season_labels.get(s, str(s)) for s in sorted(sel_seasons))
    any_train = any(s in TRAIN_SEASONS for s in sel_seasons)

    if any_train:
        st.warning(
            "Warning: selected seasons include TRAINING data. "
            "Diagnostics on in-sample predictions are misleading -- "
            "the model has already seen this data."
        )

    # -- Side-by-side model comparison table --
    st.subheader("Model comparison")
    model_versions = sorted(preds_f["model_version"].unique().tolist())
    comp_rows = []
    for mv in model_versions:
        merged = preds_f[preds_f["model_version"] == mv].merge(
            results_f[["player_id", "game_id", "goals"]],
            on=["player_id", "game_id"],
            how="inner",
        )
        if merged.empty:
            continue
        y_true = (merged["goals"] >= 1).astype(int).values
        y_prob = merged["predicted_probability"].values
        m = compute_core_metrics(y_true, y_prob)
        comp_rows.append({
            "Model": MODEL_NAMES.get(mv, mv),
            "Log Loss": round(m["log_loss"], 4),
            "Brier": round(m["brier_score"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Samples": m["n_samples"],
            "Base Rate": round(m["base_rate"], 4),
            "Mean Pred": round(m["mean_predicted"], 4),
            "Pred/Base": round(m["mean_predicted"] / max(m["base_rate"], 1e-6), 3),
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), width='stretch', hide_index=True)
    else:
        st.warning("No overlapping predictions and results for selected seasons.")
        return

    st.caption(f"Evaluated on: {split_note}")

    # -- Per-model detail --
    st.subheader("Detailed diagnostics")
    with c2:
        selected_model = st.selectbox(
            "Model",
            model_versions,
            format_func=lambda m: MODEL_NAMES.get(m, m),
            key="diag_model",
        )

    merged = preds_f[preds_f["model_version"] == selected_model].merge(
        results_f[["player_id", "game_id", "goals"]],
        on=["player_id", "game_id"],
        how="inner",
    )
    if merged.empty:
        st.warning("No overlapping data for this model/season selection.")
        return

    y_true = (merged["goals"] >= 1).astype(int).values
    y_prob = merged["predicted_probability"].values
    metrics = compute_core_metrics(y_true, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Log Loss", f"{metrics['log_loss']:.4f}")
    col2.metric("Brier Score", f"{metrics['brier_score']:.4f}")
    col3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col4.metric("Base Rate", f"{metrics['base_rate']:.3f}")
    col5.metric("Mean Pred", f"{metrics['mean_predicted']:.3f}")

    # -- Calibration + Lift side by side --
    left, right = st.columns(2)
    with left:
        st.markdown("**Calibration Curve**")
        cal = calibration_table(y_true, y_prob)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=cal["predicted_mean"], y=cal["actual_rate"],
            mode="lines+markers", name="Model",
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, cal["predicted_mean"].max() * 1.1],
            y=[0, cal["predicted_mean"].max() * 1.1],
            mode="lines", name="Perfect",
            line=dict(dash="dash", color="gray"),
        ))
        fig_cal.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(t=20, b=30),
            height=350,
            showlegend=True,
        )
        st.plotly_chart(fig_cal, width='stretch')

    with right:
        st.markdown("**Lift by decile**")
        lift = lift_table(y_true, y_prob)
        fig_lift = px.bar(lift, x="decile", y="lift")
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig_lift.update_layout(
            xaxis_title="Decile (1 = highest pred)",
            yaxis_title="Lift vs base rate",
            margin=dict(t=20, b=30),
            height=350,
        )
        st.plotly_chart(fig_lift, width='stretch')

    # -- Calibration table + cumulative gains --
    left2, right2 = st.columns(2)
    with left2:
        st.markdown("**Calibration Table**")
        cal_d = cal.copy()
        cal_d["actual_rate"] = (cal_d["actual_rate"] * 100).round(2)
        cal_d["predicted_mean"] = (cal_d["predicted_mean"] * 100).round(2)
        cal_d["abs_error"] = cal_d["abs_error"].round(4)
        st.dataframe(cal_d, width='stretch', hide_index=True)

    with right2:
        st.markdown("**Cumulative gains**")
        fig_gains = px.line(lift, x="decile", y="cumulative_actual")
        fig_gains.update_layout(
            xaxis_title="Decile",
            yaxis_title="Cumulative % goals captured",
            margin=dict(t=20, b=30),
            height=300,
        )
        st.plotly_chart(fig_gains, width='stretch')

    # -- Data coverage --
    with st.expander("Data coverage"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Player-game records (selected)", len(results_f))
            if "season" in results_f.columns:
                sc = results_f.groupby("season").size().reset_index(name="count")
                sc["split"] = sc["season"].map(_split_label)
                st.dataframe(sc, hide_index=True)
        with col2:
            if not results_f.empty:
                base = (results_f["goals"] >= 1).astype(int).mean()
                st.metric("Scoring rate (selected)", f"{base * 100:.1f}%")


# ---------------------------------------------------------------------------
# Tab 3 -- Edge Backtest
# ---------------------------------------------------------------------------

def backtest_view():
    preds = load_predictions()
    odds = load_odds()
    results = load_historical_results()

    if preds.empty or odds.empty or results.empty:
        st.info(
            "Backtesting requires predictions, odds, and game results. "
            "Ensure all three are populated in the database."
        )
        return

    preds = preds.copy()
    preds["game_date"] = pd.to_datetime(preds["game_date"], errors="coerce")
    odds = odds.copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce")
    results = results.copy()
    results["game_date"] = pd.to_datetime(results["game_date"], errors="coerce")

    # Pick primary market row per (player, game)
    odds_pick = _primary_market_rows(odds)

    model_versions = sorted(preds["model_version"].unique().tolist())
    sel_model = st.selectbox(
        "Model",
        model_versions,
        format_func=lambda m: MODEL_NAMES.get(m, m),
        key="bt_model",
    )

    merged = (
        preds[preds["model_version"] == sel_model]
        .merge(odds_pick, on=["player_id", "game_id"], how="inner")
        .merge(
            results[["player_id", "game_id", "goals"]],
            on=["player_id", "game_id"],
            how="inner",
        )
    )
    if merged.empty:
        st.info("No rows with both predictions and odds for completed games.")
        return

    merged["scored"] = (merged["goals"] >= 1).astype(int)
    merged["edge_pp"] = (
        (merged["predicted_probability"] - merged["market_implied"]) * 100
    )

    # Edge threshold slider
    min_edge = st.slider(
        "Minimum edge (percentage points)",
        min_value=-5.0, max_value=15.0, value=0.0, step=0.5,
        key="bt_edge",
    )
    bets = merged[merged["edge_pp"] >= min_edge].copy()
    if bets.empty:
        st.info(f"No bets at >= {min_edge:.1f}pp edge.")
        return

    # Compute flat-stake P/L using American odds
    def _payout(american: int, stake: float = 1.0) -> float:
        if american < 0:
            return stake * 100 / abs(american)
        return stake * american / 100

    bets["profit"] = bets.apply(
        lambda r: _payout(int(r["market_american"])) if r["scored"] else -1.0,
        axis=1,
    )
    bets["game_date_d"] = bets["game_date"].dt.date

    # Summary
    n_bets = len(bets)
    wins = int(bets["scored"].sum())
    total_profit = float(bets["profit"].sum())
    roi = total_profit / n_bets * 100 if n_bets else 0.0
    avg_odds = float(bets["market_american"].mean())
    avg_edge = float(bets["edge_pp"].mean())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Bets", n_bets)
    k2.metric("Wins", f"{wins} ({wins / n_bets * 100:.0f}%)" if n_bets else "0")
    k3.metric("Flat-stake ROI", f"{roi:+.1f}%")
    k4.metric("Net units", f"{total_profit:+.1f}")
    k5.metric("Avg edge (pp)", f"{avg_edge:+.1f}")
    k6.metric("Avg market odds", f"{avg_odds:+.0f}")

    # Cumulative P/L chart
    bets = bets.sort_values("game_date")
    bets["cum_profit"] = bets["profit"].cumsum()
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Scatter(
        x=list(range(1, len(bets) + 1)),
        y=bets["cum_profit"].values,
        mode="lines",
        name="Cumulative P/L",
    ))
    fig_pl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_pl.update_layout(
        xaxis_title="Bet #",
        yaxis_title="Cumulative units",
        margin=dict(t=20, b=30),
        height=350,
    )
    st.plotly_chart(fig_pl, width='stretch')

    # Edge bucket performance
    bets["edge_bucket"] = pd.cut(
        bets["edge_pp"],
        bins=[-100, 0, 2, 5, 10, 100],
        labels=["<0", "0-2", "2-5", "5-10", "10+"],
    )
    bucket_stats = bets.groupby("edge_bucket", observed=True).agg(
        bets=("scored", "count"),
        wins=("scored", "sum"),
        profit=("profit", "sum"),
        avg_edge=("edge_pp", "mean"),
    ).reset_index()
    bucket_stats["win_rate"] = (bucket_stats["wins"] / bucket_stats["bets"] * 100).round(1)
    bucket_stats["roi"] = (bucket_stats["profit"] / bucket_stats["bets"] * 100).round(1)
    bucket_stats = bucket_stats.rename(columns={
        "edge_bucket": "Edge bucket (pp)",
        "bets": "Bets",
        "wins": "Wins",
        "win_rate": "Win %",
        "profit": "Net units",
        "roi": "ROI %",
        "avg_edge": "Avg edge",
    })
    st.markdown("**Performance by edge bucket**")
    st.dataframe(bucket_stats, width='stretch', hide_index=True)

    # Recent bets detail
    with st.expander("Recent bet detail (last 50)"):
        detail = bets.sort_values("game_date", ascending=False).head(50)
        detail_cols = [
            "game_date_d", "player_name", "player_team",
            "predicted_probability", "market_implied", "edge_pp",
            "market_american", "scored", "profit",
        ]
        detail_cols = [c for c in detail_cols if c in detail.columns]
        disp = detail[detail_cols].copy()
        disp = disp.rename(columns={
            "game_date_d": "Date",
            "player_name": "Player",
            "player_team": "Team",
            "predicted_probability": "Model P",
            "market_implied": "Market P",
            "edge_pp": "Edge (pp)",
            "market_american": "Line",
            "scored": "Scored?",
            "profit": "P/L",
        })
        st.dataframe(disp, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("NHL Goal Probability Model")
    init_db()

    tab1, tab2, tab3 = st.tabs(["Opportunities", "Diagnostics", "Backtest"])

    with tab1:
        opportunities_view()
    with tab2:
        diagnostics_view()
    with tab3:
        backtest_view()


if __name__ == "__main__":
    main()
