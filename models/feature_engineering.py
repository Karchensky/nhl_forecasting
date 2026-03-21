"""Feature engineering pipeline for NHL goal probability model.

Builds features for each (player, game) pair using ONLY data available
before that game (strict temporal causality — no leakage).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.db_client import get_engine
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("models.features")


def _load_tables() -> dict[str, pd.DataFrame]:
    """Load core tables into DataFrames."""
    engine = get_engine()
    tables = {}
    for name in ("players", "games", "player_game_stats", "goalie_game_stats",
                  "team_game_stats"):
        tables[name] = pd.read_sql_table(name, engine)
    return tables


def _build_player_game_base(tables: dict) -> pd.DataFrame:
    """Create the base dataset: one row per (skater, completed game) with target."""
    pgs = tables["player_game_stats"].copy()
    games = tables["games"].copy()
    players = tables["players"].copy()

    games = games[games["game_state"].isin(["FINAL", "OFF"])].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    df = pgs.merge(games[["game_id", "game_date", "season", "home_team_id",
                           "away_team_id", "game_type"]],
                   on="game_id", how="inner")

    goalies = set(players.loc[players["position"] == "G", "player_id"])
    df = df[~df["player_id"].isin(goalies)].copy()

    df["scored"] = (df["goals"] >= 1).astype(int)

    df["is_home"] = (df["team_id"] == df["home_team_id"]).astype(int)
    df["opponent_team_id"] = np.where(
        df["team_id"] == df["home_team_id"],
        df["away_team_id"],
        df["home_team_id"],
    )

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    return df


def _rolling_player_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling player-level features over prior N games."""
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()

    stat_cols = ["goals", "assists", "points", "shots", "hits", "blocked_shots",
                 "pim", "toi_seconds", "pp_goals", "takeaways", "giveaways"]

    for w in windows:
        grp = df.groupby("player_id")
        for col in stat_cols:
            shifted = grp[col].shift(1)
            rolling = shifted.rolling(window=w, min_periods=1)
            df[f"{col}_avg_{w}g"] = rolling.mean().values

        shifted_goals = grp["goals"].shift(1)
        shifted_shots = grp["shots"].shift(1)
        goals_sum = shifted_goals.rolling(window=w, min_periods=1).sum()
        shots_sum = shifted_shots.rolling(window=w, min_periods=1).sum()
        df[f"shooting_pct_{w}g"] = np.where(shots_sum > 0, goals_sum / shots_sum, 0.0)

        shifted_toi = grp["toi_seconds"].shift(1)
        toi_sum = shifted_toi.rolling(window=w, min_periods=1).sum()
        df[f"goals_per_60_{w}g"] = np.where(
            toi_sum > 0, goals_sum / (toi_sum / 3600.0), 0.0
        )
        df[f"shots_per_60_{w}g"] = np.where(
            toi_sum > 0, shots_sum / (toi_sum / 3600.0), 0.0
        )

        shifted_scored = grp["scored"].shift(1)
        df[f"scoring_rate_{w}g"] = (
            shifted_scored.rolling(window=w, min_periods=1).mean().values
        )

    return df


def _season_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding season-level player features (everything before current game)."""
    df = df.sort_values(["player_id", "season", "game_date", "game_id"]).copy()

    grp = df.groupby(["player_id", "season"])

    df["season_goals_cum"] = grp["goals"].apply(
        lambda x: x.shift(1).expanding().sum()
    ).values
    df["season_shots_cum"] = grp["shots"].apply(
        lambda x: x.shift(1).expanding().sum()
    ).values
    df["season_games_played"] = grp["goals"].apply(
        lambda x: x.shift(1).expanding().count()
    ).values

    season_toi_cum = grp["toi_seconds"].apply(
        lambda x: x.shift(1).expanding().sum()
    ).values

    df["season_shooting_pct"] = np.where(
        df["season_shots_cum"] > 0,
        df["season_goals_cum"] / df["season_shots_cum"],
        0.0,
    )
    df["season_goals_per_60"] = np.where(
        season_toi_cum > 0,
        df["season_goals_cum"] / (season_toi_cum / 3600.0),
        0.0,
    )
    df["season_scoring_rate"] = grp["scored"].apply(
        lambda x: x.shift(1).expanding().mean()
    ).values

    return df


def _team_rolling_features(df: pd.DataFrame, tables: dict, windows: list[int]) -> pd.DataFrame:
    """Rolling team-level features (team's recent form)."""
    tgs = tables["team_game_stats"].copy()
    games = tables["games"][["game_id", "game_date"]].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    tgs = tgs.merge(games, on="game_id", how="inner")
    tgs = tgs.sort_values(["team_id", "game_date"]).copy()

    for w in windows:
        grp = tgs.groupby("team_id")
        shifted_goals = grp["goals"].shift(1)
        shifted_shots = grp["shots"].shift(1)
        shifted_pp_g = grp["pp_goals"].shift(1)
        shifted_pp_o = grp["pp_opportunities"].shift(1)
        shifted_won = grp["won"].shift(1).astype(float)

        tgs[f"team_goals_avg_{w}g"] = shifted_goals.rolling(w, min_periods=1).mean().values
        tgs[f"team_shots_avg_{w}g"] = shifted_shots.rolling(w, min_periods=1).mean().values
        pp_g_sum = shifted_pp_g.rolling(w, min_periods=1).sum()
        pp_o_sum = shifted_pp_o.rolling(w, min_periods=1).sum()
        tgs[f"team_pp_pct_{w}g"] = np.where(pp_o_sum > 0, pp_g_sum / pp_o_sum, 0.0)
        tgs[f"team_win_pct_{w}g"] = shifted_won.rolling(w, min_periods=1).mean().values

    team_feat_cols = [c for c in tgs.columns
                      if c.startswith("team_") and c not in ("team_id",)]
    team_feats = tgs[["team_id", "game_id"] + team_feat_cols].copy()

    df = df.merge(team_feats, left_on=["team_id", "game_id"],
                  right_on=["team_id", "game_id"], how="left")
    return df


def _opponent_rolling_features(df: pd.DataFrame, tables: dict, windows: list[int]) -> pd.DataFrame:
    """Rolling opponent-level features (opponent defensive strength)."""
    tgs = tables["team_game_stats"].copy()
    games = tables["games"][["game_id", "game_date"]].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    tgs = tgs.merge(games, on="game_id", how="inner")

    # Self-join to get opponent's goals/shots in same game
    tgs_opp = tgs[["team_id", "game_id", "goals", "shots"]].rename(
        columns={"team_id": "opp_tid", "goals": "opp_goals_in_game",
                 "shots": "opp_shots_in_game"})
    tgs = tgs.merge(tgs_opp, on="game_id", how="inner")
    tgs = tgs[tgs["team_id"] != tgs["opp_tid"]].copy()

    tgs["goals_allowed"] = tgs["opp_goals_in_game"]
    tgs["shots_allowed"] = tgs["opp_shots_in_game"]
    tgs = tgs.sort_values(["team_id", "game_date"]).copy()

    for w in windows:
        grp = tgs.groupby("team_id")
        shifted_ga = grp["goals_allowed"].shift(1)
        shifted_sa = grp["shots_allowed"].shift(1)
        shifted_pp_o = grp["pp_opportunities"].shift(1)
        shifted_pp_g = grp["pp_goals"].shift(1)

        tgs[f"opp_ga_avg_{w}g"] = shifted_ga.rolling(w, min_periods=1).mean().values
        tgs[f"opp_sa_avg_{w}g"] = shifted_sa.rolling(w, min_periods=1).mean().values

        pk_opp = shifted_pp_o.rolling(w, min_periods=1).sum()
        pk_ga = shifted_pp_g.rolling(w, min_periods=1).sum()
        tgs[f"opp_pk_pct_{w}g"] = np.where(
            pk_opp > 0, 1.0 - (pk_ga / pk_opp), 0.8
        )

    opp_cols = [c for c in tgs.columns if c.startswith("opp_") and c != "opp_tid"
                and c not in ("opp_goals_in_game", "opp_shots_in_game")]
    opp_feats = tgs[["team_id", "game_id"] + opp_cols].copy()
    opp_feats = opp_feats.rename(columns={"team_id": "opponent_team_id"})

    df = df.merge(opp_feats, on=["opponent_team_id", "game_id"], how="left")
    return df


def _goalie_features(df: pd.DataFrame, tables: dict, windows: list[int]) -> pd.DataFrame:
    """Opponent starting goalie recent performance."""
    ggs = tables["goalie_game_stats"].copy()
    games = tables["games"][["game_id", "game_date"]].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    ggs = ggs.merge(games, on="game_id", how="inner")

    starters = ggs[ggs["started"] == True].copy()
    starters = starters.sort_values(["player_id", "game_date"]).copy()

    starters["save_pct_clean"] = starters["save_pct"].fillna(0.9)

    for w in windows:
        grp = starters.groupby("player_id")
        shifted_sv = grp["save_pct_clean"].shift(1)
        shifted_ga = grp["goals_against"].shift(1)
        starters[f"goalie_sv_pct_{w}g"] = shifted_sv.rolling(w, min_periods=1).mean().values
        starters[f"goalie_ga_avg_{w}g"] = shifted_ga.rolling(w, min_periods=1).mean().values

    goalie_cols = ["player_id", "team_id", "game_id"] + [
        c for c in starters.columns if c.startswith("goalie_")
    ]
    goalie_feats = starters[goalie_cols].rename(
        columns={"player_id": "opp_goalie_id", "team_id": "opponent_team_id"}
    )

    df = df.merge(goalie_feats, on=["opponent_team_id", "game_id"], how="left")
    return df


def _game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rest days, back-to-back, schedule density."""
    df = df.sort_values(["player_id", "game_date"]).copy()

    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(7).clip(0, 30)
    df["is_back_to_back"] = (df["rest_days"] <= 1).astype(int)

    # Vectorized games-in-last-7-days using team schedule
    team_dates = df[["team_id", "game_date"]].drop_duplicates()
    team_dates = team_dates.sort_values(["team_id", "game_date"])

    counts = []
    for _, row in team_dates.iterrows():
        mask = (
            (team_dates["team_id"] == row["team_id"])
            & (team_dates["game_date"] >= row["game_date"] - timedelta(days=7))
            & (team_dates["game_date"] < row["game_date"])
        )
        counts.append(mask.sum())
    team_dates["games_last_7d"] = counts

    df = df.merge(team_dates, on=["team_id", "game_date"], how="left",
                  suffixes=("", "_dup"))
    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    df.drop(columns=dup_cols, inplace=True)
    df["games_last_7d"] = df["games_last_7d"].fillna(0).astype(int)

    df.drop(columns=["prev_game_date"], inplace=True)
    return df


def _player_profile_features(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Static player attributes."""
    players = tables["players"][["player_id", "position", "birth_date"]].copy()

    pos_map = {"C": 0, "L": 1, "R": 2, "D": 3, "LW": 1, "RW": 2}
    players["position_code"] = players["position"].map(pos_map).fillna(0).astype(int)
    players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")

    df = df.merge(players[["player_id", "position_code", "birth_date"]],
                  on="player_id", how="left", suffixes=("", "_player"))

    df["age"] = ((df["game_date"] - df["birth_date"]).dt.days / 365.25).fillna(27)
    df.drop(columns=["birth_date"], inplace=True, errors="ignore")

    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction terms between player scoring ability and opponent weakness."""
    if "goals_per_60_10g" in df.columns and "opp_ga_avg_10g" in df.columns:
        df["player_g60_x_opp_ga"] = df["goals_per_60_10g"] * df["opp_ga_avg_10g"].fillna(3.0)

    if "pp_goals_avg_10g" in df.columns and "opp_pk_pct_10g" in df.columns:
        df["player_ppg_x_opp_pk_weak"] = (
            df["pp_goals_avg_10g"] * (1.0 - df["opp_pk_pct_10g"].fillna(0.8))
        )

    return df


def build_feature_matrix(seasons: list[int] = None) -> pd.DataFrame:
    """Build the full feature matrix for model training/inference.

    Returns a DataFrame with one row per (player, game) with all features
    and the binary target 'scored'.
    """
    cfg = load_config()
    windows = cfg["model"]["rolling_windows"]

    logger.info("Loading tables...")
    tables = _load_tables()

    logger.info("Building base dataset...")
    df = _build_player_game_base(tables)
    initial_rows = len(df)
    logger.info("Base: %d rows", initial_rows)

    if seasons:
        df = df[df["season"].isin(seasons)].copy()
        logger.info("Filtered to seasons %s: %d rows", seasons, len(df))

    logger.info("Computing rolling player features...")
    df = _rolling_player_features(df, windows)

    logger.info("Computing season player features...")
    df = _season_player_features(df)

    logger.info("Computing team features...")
    df = _team_rolling_features(df, tables, windows)

    logger.info("Computing opponent features...")
    df = _opponent_rolling_features(df, tables, windows)

    logger.info("Computing goalie features...")
    df = _goalie_features(df, tables, windows)

    logger.info("Computing game context features...")
    df = _game_context_features(df)

    logger.info("Computing player profile features...")
    df = _player_profile_features(df, tables)

    logger.info("Computing interaction features...")
    df = _interaction_features(df)

    non_feature_cols = [
        "player_id", "game_id", "team_id", "game_date", "season",
        "home_team_id", "away_team_id", "opponent_team_id", "game_type",
        "goals", "assists", "points", "shots", "hits", "blocked_shots",
        "pim", "plus_minus", "toi_seconds", "pp_toi_seconds", "sh_toi_seconds",
        "ev_toi_seconds", "pp_goals", "sh_goals", "gw_goals", "ot_goals",
        "faceoff_wins", "faceoff_losses", "takeaways", "giveaways",
        "scored", "opp_goalie_id",
    ]

    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    logger.info("Feature matrix: %d rows x %d features", len(df), len(feature_cols))
    logger.info("Features: %s", feature_cols)

    return df
