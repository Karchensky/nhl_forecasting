"""NHL Goal Probability Model -- Streamlit Dashboard."""

import pickle
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

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


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


@st.cache_data(ttl=600)
def load_shot_events_summary():
    """Load shot event aggregates for xG diagnostics."""
    engine = get_engine()
    query = """
        SELECT
            se.event_type, se.shot_type, se.is_goal,
            se.distance, se.angle, se.x_coord, se.y_coord,
            se.situation_code, se.period,
            g.season
        FROM shot_events se
        JOIN games g ON se.game_id = g.game_id
        WHERE g.game_state IN ('FINAL', 'OFF')
          AND se.period_type != 'SO'
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


def _load_saved_model_artifact(name: str) -> dict | None:
    """Load a saved model pickle for feature importance inspection."""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_xg_artifact() -> dict | None:
    """Load the saved xG model artifact."""
    path = MODEL_DIR / "xg_model.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


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
    c1, c2, c3, c4, c5 = st.columns(5)
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

    positions = sorted(work["position"].dropna().unique().tolist())
    with c4:
        pos_pick = st.multiselect("Position", options=positions, default=positions, key="opp_pos")
    if pos_pick:
        work = work[work["position"].isin(pos_pick)]

    all_models = sorted(work["model_version"].unique().tolist())
    default_models = [m for m in all_models if m != "logistic_regression"] or all_models
    with c5:
        model_pick = st.multiselect(
            "Models",
            options=all_models,
            default=default_models,
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

    # Average edge across selected models
    if edge_cols:
        edge_mat = wide[edge_cols].to_numpy(dtype=float)
        wide["Avg edge %"] = np.round(np.nanmean(
            np.where(np.isfinite(edge_mat), edge_mat, np.nan), axis=1
        ), 2)
        wide["_best_edge"] = np.nanmax(
            np.where(np.isfinite(edge_mat), edge_mat, np.nan), axis=1
        )
    else:
        wide["Avg edge %"] = np.nan
        wide["_best_edge"] = np.nan

    # Model consensus: how many models agree on +EV
    if edge_cols:
        edge_mat = wide[edge_cols].to_numpy(dtype=float)
        wide["Consensus"] = np.nansum(edge_mat > 0, axis=1).astype(int)
        wide["Consensus"] = wide["Consensus"].astype(str) + "/" + str(len(edge_cols))
    else:
        wide["Consensus"] = "--"

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

    # Build game label for game rank grouping
    wide["_game_label"] = (
        wide["away_team"].astype(str) + " @ " + wide["home_team"].astype(str)
    )
    wide["Game rank"] = (
        wide.groupby("_game_label", dropna=False)["_best_edge"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    wide = wide.sort_values("_best_edge", ascending=False, na_position="last")

    # -- Slicer row: edge + rank filters --
    st.markdown("---")
    f1, f2, f3, f4 = st.columns(4)
    avg_edge_vals = wide["Avg edge %"].dropna()
    edge_min = float(avg_edge_vals.min()) if not avg_edge_vals.empty else -20.0
    edge_max = float(avg_edge_vals.max()) if not avg_edge_vals.empty else 10.0
    with f1:
        edge_range = st.slider(
            "Avg edge % range",
            min_value=edge_min, max_value=edge_max,
            value=(edge_min, edge_max), step=0.5,
            key="opp_edge_range",
        )
    with f2:
        max_rank = int(wide["Rank"].max()) if wide["Rank"].notna().any() else 100
        rank_limit = st.slider(
            "Overall rank (top N)", min_value=1, max_value=max_rank,
            value=max_rank, step=1, key="opp_rank_limit",
        )
    with f3:
        max_team_rank = int(wide["Team rank"].max()) if wide["Team rank"].notna().any() else 30
        team_rank_limit = st.slider(
            "Team rank (top N)", min_value=1, max_value=max_team_rank,
            value=max_team_rank, step=1, key="opp_team_rank_limit",
        )
    with f4:
        max_game_rank = int(wide["Game rank"].max()) if wide["Game rank"].notna().any() else 30
        game_rank_limit = st.slider(
            "Game rank (top N)", min_value=1, max_value=max_game_rank,
            value=max_game_rank, step=1, key="opp_game_rank_limit",
        )

    # Apply slicer filters
    mask = pd.Series(True, index=wide.index)
    mask &= wide["Avg edge %"].between(edge_range[0], edge_range[1]) | wide["Avg edge %"].isna()
    mask &= wide["Rank"].le(rank_limit) | wide["Rank"].isna()
    mask &= wide["Team rank"].le(team_rank_limit) | wide["Team rank"].isna()
    mask &= wide["Game rank"].le(game_rank_limit) | wide["Game rank"].isna()
    wide = wide[mask]

    # -- Summary KPIs --
    has_market = wide["market_implied"].notna()
    n_players = len(wide)
    n_with_odds = int(has_market.sum())
    n_positive_ev = int((wide["_best_edge"] > 0).sum()) if edge_cols else 0
    avg_edge = float(wide.loc[wide["_best_edge"] > 0, "_best_edge"].mean()) if n_positive_ev else 0.0
    top_edge = float(wide["_best_edge"].max()) if edge_cols and not wide["_best_edge"].isna().all() else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Slate", selected_date.isoformat())
    k2.metric("Players", n_players)
    k3.metric("w/ FD odds", n_with_odds)
    k4.metric("+EV plays", n_positive_ev)
    k5.metric("Avg +EV edge", f"{avg_edge:+.1f}pp" if avg_edge else "--")
    k6.metric("Best edge", f"{top_edge:+.1f}pp" if top_edge else "--")

    # -- Main table --
    out_cols = (
        ["Rank", "Team rank", "Game rank", "player_name", "position",
         "player_team", "home_team", "away_team"]
        + [MODEL_DISPLAY[m][0] for m in model_pick if m in wide.columns]
        + ["FD Line", "Avg model line", "FD %"]
        + [MODEL_DISPLAY[m][1] for m in model_pick if m in wide.columns]
        + ["Avg edge %", "Consensus"]
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

    # -- Visuals --
    # Edge distribution histogram
    has_edge = wide["Avg edge %"].notna()
    if has_edge.any():
        st.markdown("**Edge vs. FanDuel**")
        edge_data = wide.loc[has_edge, "Avg edge %"]
        fig_edge = go.Figure()
        fig_edge.add_trace(go.Histogram(
            x=edge_data,
            nbinsx=25,
            marker_color="cornflowerblue",
            hovertemplate="Edge: %{x:.1f}pp<br>Players: %{y}<extra></extra>",
        ))
        fig_edge.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1.5)
        fig_edge.update_layout(
            title="Edge vs. FanDuel",
            xaxis_title="% Difference between FanDuel Odds vs. Model Odds",
            yaxis_title="Number of Players",
            margin=dict(t=40, b=40),
            height=350,
        )
        st.plotly_chart(fig_edge, width='stretch')

    # Model vs. FanDuel scatter
    with st.expander("Model vs. FanDuel scatter", expanded=True):
        scatter_has_market = wide["market_implied"].notna()
        scatter_data = wide[scatter_has_market]
        if scatter_data.empty:
            st.info("No players with both model predictions and FanDuel odds.")
        else:
            scatter_models = model_prob_cols
            sc_sel = st.selectbox(
                "Model to compare",
                scatter_models,
                format_func=lambda m: MODEL_NAMES.get(m, m),
                key="opp_scatter_model",
            )
            fd_probs = scatter_data["market_implied"]
            model_probs = scatter_data[sc_sel]
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=fd_probs,
                y=model_probs,
                mode="markers",
                marker=dict(size=6, opacity=0.6, color="cornflowerblue"),
                text=scatter_data["player_name"],
                hovertemplate=(
                    "%{text}<br>"
                    "FanDuel: %{x:.1%}<br>"
                    + MODEL_NAMES.get(sc_sel, sc_sel)
                    + ": %{y:.1%}<extra></extra>"
                ),
            ))
            max_val = max(
                fd_probs.max() if not fd_probs.empty else 0.1,
                model_probs.max() if not model_probs.empty else 0.1,
            ) * 1.05
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(dash="dash", color="gray"),
                showlegend=False,
            ))
            fig_scatter.update_layout(
                title="Predicted vs. FanDuel Odds",
                xaxis_title="FanDuel (% Chance to Score)",
                yaxis_title=MODEL_NAMES.get(sc_sel, sc_sel) + " (% Chance to Score)",
                margin=dict(t=40, b=30),
                height=400,
            )
            st.plotly_chart(fig_scatter, width='stretch')

    # Prediction distributions
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
    model_merged_cache = {}
    for mv in model_versions:
        merged = preds_f[preds_f["model_version"] == mv].merge(
            results_f[["player_id", "game_id", "goals"]],
            on=["player_id", "game_id"],
            how="inner",
        )
        if merged.empty:
            continue
        model_merged_cache[mv] = merged
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

    merged = model_merged_cache.get(selected_model)
    if merged is None or merged.empty:
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

    # -- Prediction distribution overlay --
    st.markdown("**Prediction distribution (all models)**")
    fig_dist = go.Figure()
    for mv in model_versions:
        mv_merged = model_merged_cache.get(mv)
        if mv_merged is None:
            continue
        fig_dist.add_trace(go.Histogram(
            x=mv_merged["predicted_probability"],
            name=MODEL_NAMES.get(mv, mv),
            opacity=0.5,
            nbinsx=40,
        ))
    fig_dist.update_layout(
        barmode="overlay",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        margin=dict(t=20, b=30),
        height=300,
    )
    st.plotly_chart(fig_dist, width='stretch')

    # -- Per-position breakdown --
    with st.expander("Performance by position"):
        merged_with_pos = merged.copy()
        if "position" in merged_with_pos.columns:
            pos_groups = {"Forward": ["C", "L", "R", "LW", "RW"], "Defense": ["D"]}
            rows_pos = []
            for label, pos_list in pos_groups.items():
                mask = merged_with_pos["position"].isin(pos_list)
                subset = merged_with_pos[mask]
                if len(subset) < 50:
                    continue
                yt = (subset["goals"] >= 1).astype(int).values
                yp = subset["predicted_probability"].values
                m = compute_core_metrics(yt, yp)
                rows_pos.append({
                    "Position": label,
                    "Samples": m["n_samples"],
                    "Base Rate": f"{m['base_rate']:.3f}",
                    "Mean Pred": f"{m['mean_predicted']:.3f}",
                    "Log Loss": f"{m['log_loss']:.4f}",
                    "AUC": f"{m['roc_auc']:.4f}",
                    "Brier": f"{m['brier_score']:.4f}",
                })
            if rows_pos:
                st.dataframe(pd.DataFrame(rows_pos), width='stretch', hide_index=True)
            else:
                st.info("Not enough data per position group.")

    # -- Feature importance --
    with st.expander("Feature importance (top 30)"):
        artifact = _load_saved_model_artifact(selected_model)
        if artifact and "model" in artifact:
            model_obj = artifact["model"]
            feature_cols = artifact.get("feature_cols", [])
            importances = None

            if selected_model == "lightgbm":
                importances = model_obj.feature_importance(importance_type="gain")
            elif selected_model == "xgboost":
                score_dict = model_obj.get_score(importance_type="gain")
                importances = [score_dict.get(f, 0) for f in feature_cols]
            elif selected_model == "logistic_regression":
                if hasattr(model_obj, "coef_"):
                    importances = np.abs(model_obj.coef_[0])

            if importances is not None and len(importances) == len(feature_cols):
                imp_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False).head(30)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                )
                fig_imp.update_layout(
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=20, b=30, l=200),
                    height=max(400, len(imp_df) * 18),
                )
                st.plotly_chart(fig_imp, width='stretch')
            else:
                st.info("Could not extract feature importances for this model type.")
        else:
            st.info(f"No saved model artifact found for {MODEL_NAMES.get(selected_model, selected_model)}.")

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
# Tab 3 -- xG Model Diagnostics
# ---------------------------------------------------------------------------

def xg_diagnostics_view():
    st.subheader("Expected Goals (xG) Model")

    artifact = _load_xg_artifact()
    if artifact is None:
        st.info("No trained xG model found. Run `python -m models.xg_model` first.")
        return

    model_results = artifact.get("results", {})
    feature_cols = artifact.get("feature_cols", [])
    model = artifact.get("model")

    # -- Performance metrics table --
    st.markdown("**Model performance**")
    perf_rows = []
    for split in ["train", "val", "test"]:
        r = model_results.get(split, {})
        if r:
            perf_rows.append({
                "Split": split.upper(),
                "AUC": round(r["auc"], 4),
                "Log Loss": round(r["log_loss"], 4),
                "Brier": round(r["brier"], 4),
            })
    if perf_rows:
        st.dataframe(pd.DataFrame(perf_rows), width='stretch', hide_index=True)
    else:
        st.warning("No evaluation results stored in xG model artifact.")

    # -- Feature importance --
    if model is not None and feature_cols:
        st.markdown("**Top 20 features (gain)**")
        imp_vals = model.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": imp_vals,
        }).sort_values("Importance", ascending=False).head(20)

        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=30, l=180),
            height=max(400, len(imp_df) * 20),
        )
        st.plotly_chart(fig_imp, width='stretch')

    # -- Shot data analysis --
    shots_df = load_shot_events_summary()
    if shots_df.empty:
        st.info("No shot event data. Run backfill_pbp first.")
        return

    st.markdown("---")
    st.subheader("Shot data overview")

    k1, k2, k3, k4 = st.columns(4)
    total_shots = len(shots_df)
    total_goals = int(shots_df["is_goal"].sum())
    goal_rate = total_goals / total_shots * 100 if total_shots else 0
    n_seasons = shots_df["season"].nunique()
    k1.metric("Total shots", f"{total_shots:,}")
    k2.metric("Goals", f"{total_goals:,}")
    k3.metric("Goal rate", f"{goal_rate:.2f}%")
    k4.metric("Seasons", n_seasons)

    # -- Goal rate by shot type --
    left, right = st.columns(2)
    with left:
        st.markdown("**Goal rate by shot type**")
        st_df = (
            shots_df[shots_df["shot_type"].notna()]
            .groupby("shot_type")
            .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"))
            .reset_index()
        )
        st_df["goal_pct"] = (st_df["goals"] / st_df["shots"] * 100).round(2)
        st_df = st_df.sort_values("goal_pct", ascending=False)

        fig_st = px.bar(st_df, x="goal_pct", y="shot_type", orientation="h",
                        text="shots", labels={"goal_pct": "Goal %", "shot_type": "Shot Type"})
        fig_st.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=30, l=120),
            height=350,
        )
        st.plotly_chart(fig_st, width='stretch')

    with right:
        st.markdown("**Goal rate by game situation**")
        sit_df = shots_df[shots_df["situation_code"].notna()].copy()
        sit_labels = {
            "1551": "5v5",
            "1541": "Home PP (5v4)",
            "1451": "Away PP (5v4)",
            "1531": "Home PP (5v3)",
            "1351": "Away PP (5v3)",
            "0551": "Home EN",
            "1550": "Away EN",
        }
        sit_df["situation"] = sit_df["situation_code"].map(sit_labels).fillna("Other")
        sit_agg = (
            sit_df.groupby("situation")
            .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"))
            .reset_index()
        )
        sit_agg["goal_pct"] = (sit_agg["goals"] / sit_agg["shots"] * 100).round(2)
        sit_agg = sit_agg[sit_agg["shots"] >= 100].sort_values("goal_pct", ascending=False)

        fig_sit = px.bar(sit_agg, x="goal_pct", y="situation", orientation="h",
                         text="shots", labels={"goal_pct": "Goal %", "situation": "Situation"})
        fig_sit.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=30, l=120),
            height=350,
        )
        st.plotly_chart(fig_sit, width='stretch')

    # -- Distance/angle vs goal rate --
    left2, right2 = st.columns(2)
    with left2:
        st.markdown("**Goal rate by distance**")
        dist_df = shots_df[shots_df["distance"].notna()].copy()
        dist_df["dist_bin"] = pd.cut(dist_df["distance"], bins=range(0, 100, 5))
        dist_agg = (
            dist_df.groupby("dist_bin", observed=True)
            .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"))
            .reset_index()
        )
        dist_agg["goal_pct"] = (dist_agg["goals"] / dist_agg["shots"] * 100).round(2)
        dist_agg["distance"] = dist_agg["dist_bin"].apply(lambda x: x.mid)

        fig_dist = px.line(dist_agg, x="distance", y="goal_pct",
                           labels={"distance": "Distance (ft)", "goal_pct": "Goal %"},
                           markers=True)
        fig_dist.update_layout(margin=dict(t=20, b=30), height=300)
        st.plotly_chart(fig_dist, width='stretch')

    with right2:
        st.markdown("**Goal rate by angle**")
        angle_df = shots_df[shots_df["angle"].notna()].copy()
        angle_df["angle_bin"] = pd.cut(angle_df["angle"], bins=range(0, 95, 5))
        angle_agg = (
            angle_df.groupby("angle_bin", observed=True)
            .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"))
            .reset_index()
        )
        angle_agg["goal_pct"] = (angle_agg["goals"] / angle_agg["shots"] * 100).round(2)
        angle_agg["angle"] = angle_agg["angle_bin"].apply(lambda x: x.mid)

        fig_angle = px.line(angle_agg, x="angle", y="goal_pct",
                            labels={"angle": "Angle (degrees)", "goal_pct": "Goal %"},
                            markers=True)
        fig_angle.update_layout(margin=dict(t=20, b=30), height=300)
        st.plotly_chart(fig_angle, width='stretch')

    # -- Shot location heatmap --
    with st.expander("Shot location heatmap"):
        loc_df = shots_df[shots_df["x_coord"].notna() & shots_df["y_coord"].notna()].copy()
        loc_df["x_abs"] = loc_df["x_coord"].abs()

        fig_heat = go.Figure()
        fig_heat.add_trace(go.Histogram2d(
            x=loc_df["x_abs"],
            y=loc_df["y_coord"],
            colorscale="YlOrRd",
            nbinsx=50,
            nbinsy=50,
            colorbar=dict(title="Shots"),
        ))
        fig_heat.update_layout(
            xaxis_title="X (distance from center ice)",
            yaxis_title="Y (lateral position)",
            margin=dict(t=20, b=30),
            height=400,
            width=600,
        )
        st.plotly_chart(fig_heat, width='stretch')

    # -- Per-period goal rate --
    with st.expander("Goal rate by period"):
        period_df = (
            shots_df[shots_df["period"].between(1, 5)]
            .groupby("period")
            .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"))
            .reset_index()
        )
        period_df["goal_pct"] = (period_df["goals"] / period_df["shots"] * 100).round(2)
        period_df["period_label"] = period_df["period"].map(
            {1: "1st", 2: "2nd", 3: "3rd", 4: "OT", 5: "SO"}
        )
        fig_period = px.bar(period_df, x="period_label", y="goal_pct",
                            text="shots",
                            labels={"period_label": "Period", "goal_pct": "Goal %"})
        fig_period.update_layout(margin=dict(t=20, b=30), height=300)
        st.plotly_chart(fig_period, width='stretch')

    # -- Season-over-season shot volume --
    with st.expander("Shots and goals by season"):
        season_df = (
            shots_df.groupby("season")
            .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"))
            .reset_index()
        )
        season_df["goal_pct"] = (season_df["goals"] / season_df["shots"] * 100).round(2)
        st.dataframe(season_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4 -- Edge Backtest
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

    c1, c2 = st.columns([1, 3])
    model_versions = sorted(preds["model_version"].unique().tolist())
    with c1:
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

    with c2:
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

    # Daily P/L
    with st.expander("Daily P/L breakdown"):
        daily = bets.groupby("game_date_d").agg(
            bets=("scored", "count"),
            wins=("scored", "sum"),
            profit=("profit", "sum"),
            avg_edge=("edge_pp", "mean"),
        ).reset_index()
        daily["cum_profit"] = daily["profit"].cumsum()
        daily["roi"] = (daily["profit"] / daily["bets"] * 100).round(1)

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily["game_date_d"],
            y=daily["profit"],
            name="Daily P/L",
            marker_color=np.where(daily["profit"] >= 0, "#2ecc71", "#e74c3c"),
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily["game_date_d"],
            y=daily["cum_profit"],
            name="Cumulative",
            mode="lines+markers",
            yaxis="y2",
        ))
        fig_daily.update_layout(
            yaxis=dict(title="Daily units"),
            yaxis2=dict(title="Cumulative units", overlaying="y", side="right"),
            margin=dict(t=20, b=30),
            height=350,
            legend=dict(x=0, y=1.1, orientation="h"),
        )
        st.plotly_chart(fig_daily, width='stretch')

        daily_disp = daily.rename(columns={
            "game_date_d": "Date", "bets": "Bets", "wins": "Wins",
            "profit": "P/L", "avg_edge": "Avg Edge (pp)", "roi": "ROI %",
            "cum_profit": "Cum P/L",
        })
        st.dataframe(daily_disp, width='stretch', hide_index=True)

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
# Tab 5 -- Data Pipeline Status
# ---------------------------------------------------------------------------

def pipeline_status_view():
    st.subheader("Data pipeline status")

    engine = get_engine()

    with engine.connect() as conn:
        # Games coverage
        games_by_season = conn.execute(sa_text("""
            SELECT season,
                   COUNT(*) as total,
                   SUM(CASE WHEN game_state IN ('FINAL', 'OFF') THEN 1 ELSE 0 END) as completed
            FROM games
            WHERE game_type IN (2, 3)
            GROUP BY season ORDER BY season
        """)).fetchall()

        # Shot events coverage
        pbp_by_season = conn.execute(sa_text("""
            SELECT g.season,
                   COUNT(DISTINCT se.game_id) as games_with_pbp,
                   COUNT(*) as total_shots
            FROM shot_events se
            JOIN games g ON se.game_id = g.game_id
            GROUP BY g.season ORDER BY g.season
        """)).fetchall()

        # Player stats coverage
        pgs_by_season = conn.execute(sa_text("""
            SELECT g.season,
                   COUNT(DISTINCT pgs.game_id) as games,
                   COUNT(*) as player_game_rows
            FROM player_game_stats pgs
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY g.season ORDER BY g.season
        """)).fetchall()

        # Predictions coverage
        pred_by_season = conn.execute(sa_text("""
            SELECT g.season,
                   COUNT(DISTINCT mo.game_id) as games,
                   COUNT(*) as predictions,
                   COUNT(DISTINCT mo.model_version) as models
            FROM model_outputs mo
            JOIN games g ON mo.game_id = g.game_id
            GROUP BY g.season ORDER BY g.season
        """)).fetchall()

        # Odds coverage
        odds_stats = conn.execute(sa_text("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT game_id) as games,
                   MIN(retrieved_at) as first_capture,
                   MAX(retrieved_at) as last_capture
            FROM odds
        """)).fetchone()

        # Model files
        model_files = list(MODEL_DIR.glob("*.pkl"))

    # -- Games table --
    st.markdown("**Games by season**")
    games_df = pd.DataFrame(games_by_season, columns=["Season", "Total", "Completed"])
    games_df["Split"] = games_df["Season"].map(lambda s: _split_label(int(s)))
    st.dataframe(games_df, width='stretch', hide_index=True)

    # -- PBP coverage --
    left, right = st.columns(2)
    with left:
        st.markdown("**Play-by-play (shot events) coverage**")
        if pbp_by_season:
            pbp_df = pd.DataFrame(pbp_by_season, columns=["Season", "Games w/ PBP", "Total Shots"])
            st.dataframe(pbp_df, width='stretch', hide_index=True)
        else:
            st.info("No PBP data. Run backfill_pbp.")

    with right:
        st.markdown("**Player stats coverage**")
        if pgs_by_season:
            pgs_df = pd.DataFrame(pgs_by_season, columns=["Season", "Games", "Player-Game Rows"])
            st.dataframe(pgs_df, width='stretch', hide_index=True)
        else:
            st.info("No player stats data.")

    # -- Predictions + Odds + Models --
    left2, mid2, right2 = st.columns(3)
    with left2:
        st.markdown("**Predictions**")
        if pred_by_season:
            pred_df = pd.DataFrame(pred_by_season, columns=["Season", "Games", "Predictions", "Models"])
            st.dataframe(pred_df, width='stretch', hide_index=True)
        else:
            st.info("No predictions yet.")

    with mid2:
        st.markdown("**Odds data**")
        if odds_stats and odds_stats[0] > 0:
            st.metric("Total odds rows", f"{odds_stats[0]:,}")
            st.metric("Games with odds", odds_stats[1])
            st.metric("First capture", str(odds_stats[2])[:19] if odds_stats[2] else "N/A")
            st.metric("Last capture", str(odds_stats[3])[:19] if odds_stats[3] else "N/A")
        else:
            st.info("No odds data captured yet.")

    with right2:
        st.markdown("**Saved models**")
        for mf in sorted(model_files):
            size_kb = mf.stat().st_size / 1024
            st.text(f"{mf.name} ({size_kb:.0f} KB)")
        if not model_files:
            st.info("No saved models.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("NHL Goal Probability Model")
    init_db()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Opportunities", "Model Diagnostics", "xG Model", "Backtest", "Pipeline Status"
    ])

    with tab1:
        opportunities_view()
    with tab2:
        diagnostics_view()
    with tab3:
        xg_diagnostics_view()
    with tab4:
        backtest_view()
    with tab5:
        pipeline_status_view()


if __name__ == "__main__":
    main()
