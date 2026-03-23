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

# NHL Web API ``gameType``: 1 = preseason, 2 = regular season, 3 = playoffs.
# Schedule ingest already skips type 1; this also drops any preseason rows in DB.
COMPETITIVE_GAME_TYPES = (2, 3)


def _games_competitive_for_joins(tables: dict) -> pd.DataFrame:
    """``game_id`` + ``game_date`` for regular + playoff games only."""
    g = tables["games"][["game_id", "game_date", "game_type"]].copy()
    g = g[g["game_type"].isin(COMPETITIVE_GAME_TYPES)]
    g["game_date"] = pd.to_datetime(g["game_date"])
    return g[["game_id", "game_date"]]


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
    games = games[games["game_type"].isin(COMPETITIVE_GAME_TYPES)].copy()
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
    """Compute rolling player-level features over prior N games.

    Includes core stats, enrichment-based opportunity/deployment features,
    and advanced analytics (Corsi, zone starts, PDO) when available.
    """
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()

    core_stat_cols = [
        "goals", "assists", "points", "shots", "hits", "blocked_shots",
        "pim", "toi_seconds", "pp_goals", "takeaways", "giveaways",
    ]

    enrichment_cols = [
        "pp_toi_seconds", "sh_toi_seconds", "ev_toi_seconds",
        "shifts", "corsi_for", "corsi_against",
        "fenwick_for", "fenwick_against",
        "oz_start_pct", "dz_start_pct",
        "total_shot_attempts", "missed_shots",
        "pp_shots", "pp_individual_sat_for",
        "es_goals_for", "es_goals_against",
        "pdo", "on_ice_shooting_pct",
    ]

    available_enrichment = [c for c in enrichment_cols if c in df.columns]
    all_avg_cols = core_stat_cols + available_enrichment

    new_cols = {}
    for w in windows:
        grp = df.groupby("player_id")

        for col in all_avg_cols:
            shifted = grp[col].shift(1)
            new_cols[f"{col}_avg_{w}g"] = shifted.rolling(w, min_periods=1).mean().values

        shifted_goals = grp["goals"].shift(1)
        shifted_shots = grp["shots"].shift(1)
        goals_sum = shifted_goals.rolling(w, min_periods=1).sum()
        shots_sum = shifted_shots.rolling(w, min_periods=1).sum()
        new_cols[f"shooting_pct_{w}g"] = np.where(
            shots_sum > 0, goals_sum / shots_sum, 0.0
        )

        shifted_toi = grp["toi_seconds"].shift(1)
        toi_sum = shifted_toi.rolling(w, min_periods=1).sum()
        new_cols[f"goals_per_60_{w}g"] = np.where(
            toi_sum > 0, goals_sum / (toi_sum / 3600.0), 0.0
        )
        new_cols[f"shots_per_60_{w}g"] = np.where(
            toi_sum > 0, shots_sum / (toi_sum / 3600.0), 0.0
        )

        shifted_scored = grp["scored"].shift(1)
        new_cols[f"scoring_rate_{w}g"] = (
            shifted_scored.rolling(w, min_periods=1).mean().values
        )

        if "pp_toi_seconds" in df.columns:
            pp_toi_sum = grp["pp_toi_seconds"].shift(1).rolling(w, min_periods=1).sum()
            new_cols[f"pp_toi_share_{w}g"] = np.where(
                toi_sum > 0, pp_toi_sum / toi_sum, 0.0
            )
            new_cols[f"pp_goals_per_60_{w}g"] = np.where(
                pp_toi_sum > 0,
                grp["pp_goals"].shift(1).rolling(w, min_periods=1).sum() / (pp_toi_sum / 3600.0),
                0.0,
            )

        if "corsi_for" in df.columns:
            cf_sum = grp["corsi_for"].shift(1).rolling(w, min_periods=1).sum()
            ca_sum = grp["corsi_against"].shift(1).rolling(w, min_periods=1).sum()
            new_cols[f"corsi_diff_{w}g"] = cf_sum - ca_sum
            total_corsi = cf_sum + ca_sum
            new_cols[f"corsi_pct_{w}g"] = np.where(
                total_corsi > 0, cf_sum / total_corsi, 0.5
            )

        if "fenwick_for" in df.columns:
            ff_sum = grp["fenwick_for"].shift(1).rolling(w, min_periods=1).sum()
            fa_sum = grp["fenwick_against"].shift(1).rolling(w, min_periods=1).sum()
            total_fen = ff_sum + fa_sum
            new_cols[f"fenwick_pct_{w}g"] = np.where(
                total_fen > 0, ff_sum / total_fen, 0.5
            )

        if "es_goals_for" in df.columns:
            esgf = grp["es_goals_for"].shift(1).rolling(w, min_periods=1).sum()
            esga = grp["es_goals_against"].shift(1).rolling(w, min_periods=1).sum()
            new_cols[f"es_goal_diff_{w}g"] = esgf - esga

        if "total_shot_attempts" in df.columns:
            tsa_sum = grp["total_shot_attempts"].shift(1).rolling(w, min_periods=1).sum()
            new_cols[f"shot_attempts_per_60_{w}g"] = np.where(
                toi_sum > 0, tsa_sum / (toi_sum / 3600.0), 0.0
            )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
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
    games = _games_competitive_for_joins(tables)
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

        if "pp_toi_seconds" in tgs.columns:
            spp = grp["pp_toi_seconds"].shift(1)
            tgs[f"team_pp_toi_rollsum_{w}g"] = spp.rolling(w, min_periods=1).sum().values
        if "pp_opportunities_actual" in tgs.columns:
            spo = grp["pp_opportunities_actual"].shift(1)
            tgs[f"team_pp_opps_rollsum_{w}g"] = spo.rolling(w, min_periods=1).sum().values

    team_feat_cols = [c for c in tgs.columns
                      if c.startswith("team_") and c not in ("team_id",)]
    team_feats = tgs[["team_id", "game_id"] + team_feat_cols].copy()

    df = df.merge(team_feats, left_on=["team_id", "game_id"],
                  right_on=["team_id", "game_id"], how="left")
    return df


def _pp_share_of_team_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Player's share of team power-play TOI over prior N games (deployment)."""
    if "pp_toi_seconds" not in df.columns:
        return df

    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()
    grp = df.groupby("player_id")
    new_cols = {}
    for w in windows:
        new_cols[f"_player_pp_toi_sum_{w}g"] = (
            grp["pp_toi_seconds"].shift(1).rolling(w, min_periods=1).sum().values
        )
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    for w in windows:
        tcol = f"team_pp_toi_rollsum_{w}g"
        pcol = f"_player_pp_toi_sum_{w}g"
        if tcol not in df.columns:
            continue
        df[f"pp_toi_share_of_team_{w}g"] = df[pcol] / np.maximum(
            df[tcol].fillna(0), 1.0
        )

    drop_internal = [c for c in df.columns if c.startswith("_player_pp_toi_sum_")]
    df.drop(columns=drop_internal, inplace=True, errors="ignore")
    return df


def _teammate_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mean prior scoring rate of teammates in the same game (lineup quality proxy)."""
    col = "goals_per_60_10g"
    if col not in df.columns:
        return df

    gsum = df.groupby(["game_id", "team_id"])[col].transform("sum")
    gcnt = df.groupby(["game_id", "team_id"])[col].transform("count")
    denom = np.maximum(gcnt - 1, 1)
    df["teammate_mean_g60_10g"] = np.where(
        gcnt > 1, (gsum - df[col]) / denom, 0.0
    )

    if "shots_per_60_10g" in df.columns:
        df["player_sh60_x_teammate_g60"] = (
            df["shots_per_60_10g"].fillna(0) * df["teammate_mean_g60_10g"]
        )

    if "pp_toi_seconds_avg_10g" in df.columns:
        df["pp_toi_avg_x_teammate_g60"] = (
            df["pp_toi_seconds_avg_10g"].fillna(0) * df["teammate_mean_g60_10g"]
        )

    return df


def _opponent_rolling_features(df: pd.DataFrame, tables: dict, windows: list[int]) -> pd.DataFrame:
    """Rolling opponent-level features (opponent defensive strength)."""
    tgs = tables["team_game_stats"].copy()
    games = _games_competitive_for_joins(tables)
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
    games = _games_competitive_for_joins(tables)
    ggs = ggs.merge(games, on="game_id", how="inner")

    starters = ggs[ggs["started"] == True].copy()
    # Multiple rows with started=True for one team-game breaks merge (pandas crash);
    # keep the primary starter (most TOI).
    if "toi_seconds" not in starters.columns:
        starters["toi_seconds"] = 0
    starters = starters.sort_values(
        ["team_id", "game_id", "toi_seconds"],
        ascending=[True, True, False],
        na_position="last",
    )
    starters = starters.groupby(["team_id", "game_id"], as_index=False).first()
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
    # Must be exactly one row per (opponent_team_id, game_id) for left merge
    dup_n = goalie_feats.duplicated(subset=["opponent_team_id", "game_id"]).sum()
    if dup_n:
        logger.warning("Dropping %d duplicate goalie rows per team-game", int(dup_n))
        goalie_feats = goalie_feats.drop_duplicates(
            subset=["opponent_team_id", "game_id"], keep="first"
        )

    for col in ("opponent_team_id", "game_id"):
        goalie_feats[col] = pd.to_numeric(goalie_feats[col], errors="coerce").astype("int64")

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

    df["game_dow"] = df["game_date"].dt.dayofweek.fillna(3).astype(int)

    df.drop(columns=["prev_game_date"], inplace=True)
    return df


def _opponent_schedule_context(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Opponent team rest, back-to-back, and schedule density (mirror of player context)."""
    g = tables["games"].copy()
    g = g[g["game_state"].isin(["FINAL", "OFF"])].copy()
    g = g[g["game_type"].isin(COMPETITIVE_GAME_TYPES)].copy()
    if g.empty:
        df["opponent_rest_days"] = 7.0
        df["opponent_is_back_to_back"] = 0
        df["opponent_games_last_7d"] = 0
        return df

    g["game_date"] = pd.to_datetime(g["game_date"])
    h = g[["game_date", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    aw = g[["game_date", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    sched = pd.concat([h, aw], ignore_index=True).drop_duplicates(
        subset=["team_id", "game_date"]
    )
    sched = sched.sort_values(["team_id", "game_date"])
    sched["prev_dt"] = sched.groupby("team_id")["game_date"].shift(1)
    sched["opponent_rest_days"] = (
        (sched["game_date"] - sched["prev_dt"]).dt.days.fillna(7).clip(0, 30)
    )
    sched["opponent_is_back_to_back"] = (sched["opponent_rest_days"] <= 1).astype(int)

    opp_dates = sched.sort_values(["team_id", "game_date"])
    counts = []
    for _, row in opp_dates.iterrows():
        mask = (
            (opp_dates["team_id"] == row["team_id"])
            & (opp_dates["game_date"] >= row["game_date"] - timedelta(days=7))
            & (opp_dates["game_date"] < row["game_date"])
        )
        counts.append(int(mask.sum()))
    opp_dates["opponent_games_last_7d"] = counts

    opp_sched = opp_dates.rename(columns={"team_id": "opponent_team_id"})
    merge_cols = [
        "opponent_team_id", "game_date", "opponent_rest_days",
        "opponent_is_back_to_back", "opponent_games_last_7d",
    ]
    df = df.merge(opp_sched[merge_cols], on=["opponent_team_id", "game_date"], how="left")
    df["opponent_rest_days"] = df["opponent_rest_days"].fillna(7.0)
    df["opponent_is_back_to_back"] = df["opponent_is_back_to_back"].fillna(0).astype(int)
    df["opponent_games_last_7d"] = df["opponent_games_last_7d"].fillna(0).astype(int)
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


def _home_road_split_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling stats split by home/road venue."""
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()

    for w in windows:
        for loc_val, loc_label in [(1, "home"), (0, "road")]:
            mask = df["is_home"] == loc_val
            temp = df[["player_id", "game_date", "goals", "shots", "toi_seconds"]].copy()
            temp.loc[~mask, ["goals", "shots", "toi_seconds"]] = np.nan

            grp = temp.groupby("player_id")
            sg = grp["goals"].shift(1)
            ss = grp["shots"].shift(1)
            st = grp["toi_seconds"].shift(1)

            g_sum = sg.rolling(w, min_periods=1).sum()
            s_sum = ss.rolling(w, min_periods=1).sum()
            t_sum = st.rolling(w, min_periods=1).sum()

            df[f"{loc_label}_goals_avg_{w}g"] = sg.rolling(w, min_periods=1).mean().values
            df[f"{loc_label}_shots_avg_{w}g"] = ss.rolling(w, min_periods=1).mean().values
            df[f"{loc_label}_g_per_60_{w}g"] = np.where(
                t_sum > 0, g_sum / (t_sum / 3600.0), 0.0
            )

            if "pp_toi_seconds" in df.columns:
                temp_pp = df[["player_id", "game_date", "pp_toi_seconds"]].copy()
                temp_pp.loc[~mask, "pp_toi_seconds"] = np.nan
                gpp = temp_pp.groupby("player_id")
                spp = gpp["pp_toi_seconds"].shift(1)
                df[f"{loc_label}_pp_toi_avg_{w}g"] = (
                    spp.rolling(w, min_periods=1).mean().values
                )

            if "pp_shots" in df.columns:
                temp_ps = df[["player_id", "game_date", "pp_shots"]].copy()
                temp_ps.loc[~mask, "pp_shots"] = np.nan
                gps = temp_ps.groupby("player_id")
                sps = gps["pp_shots"].shift(1)
                df[f"{loc_label}_pp_shots_avg_{w}g"] = (
                    sps.rolling(w, min_periods=1).mean().values
                )

            if "total_shot_attempts" in df.columns:
                temp_tsa = df[["player_id", "game_date", "total_shot_attempts"]].copy()
                temp_tsa.loc[~mask, "total_shot_attempts"] = np.nan
                gts = temp_tsa.groupby("player_id")
                sts = gts["total_shot_attempts"].shift(1)
                df[f"{loc_label}_shot_attempts_avg_{w}g"] = (
                    sts.rolling(w, min_periods=1).mean().values
                )

    return df


def _vs_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Player historical performance vs the specific upcoming opponent."""
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()

    career_vs = (
        df.groupby(["player_id", "opponent_team_id"])
        .apply(
            lambda g: g.assign(
                vs_opp_goals_cum=g["goals"].shift(1).expanding().sum(),
                vs_opp_games_cum=g["goals"].shift(1).expanding().count(),
                vs_opp_shots_cum=g["shots"].shift(1).expanding().sum(),
            ),
            include_groups=False,
        )
    )

    if career_vs.empty:
        df["vs_opp_goals_per_game"] = 0.0
        df["vs_opp_shooting_pct"] = 0.0
        return df

    df["vs_opp_goals_cum"] = career_vs["vs_opp_goals_cum"].values
    df["vs_opp_games_cum"] = career_vs["vs_opp_games_cum"].values
    df["vs_opp_shots_cum"] = career_vs["vs_opp_shots_cum"].values

    df["vs_opp_goals_per_game"] = np.where(
        df["vs_opp_games_cum"] > 0,
        df["vs_opp_goals_cum"] / df["vs_opp_games_cum"],
        0.0,
    )
    df["vs_opp_shooting_pct"] = np.where(
        df["vs_opp_shots_cum"] > 0,
        df["vs_opp_goals_cum"] / df["vs_opp_shots_cum"],
        0.0,
    )
    df.drop(columns=["vs_opp_goals_cum", "vs_opp_games_cum", "vs_opp_shots_cum"],
            inplace=True)

    return df


def _games_since_last_goal_block(prev_scored: np.ndarray) -> np.ndarray:
    """Consecutive games without a goal before each row (capped at 50)."""
    n = len(prev_scored)
    out = np.zeros(n, dtype=np.float64)
    run = 0.0
    for i in range(n):
        out[i] = min(run, 50.0)
        v = prev_scored[i]
        if np.isnan(v):
            run = min(run + 1.0, 50.0)
        elif v >= 1:
            run = 0.0
        else:
            run = min(run + 1.0, 50.0)
    return out


def _streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum/streak indicators."""
    df = df.sort_values(["player_id", "game_date", "game_id"]).copy()

    shifted_scored = df.groupby("player_id")["scored"].shift(1)
    df["scored_last_1"] = shifted_scored.fillna(0).values
    df["scored_last_3_sum"] = df.groupby("player_id")["scored"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )

    shifted_goals = df.groupby("player_id")["goals"].shift(1)
    df["goals_last_3_sum"] = df.groupby("player_id")["goals"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["goals_last_1"] = shifted_goals.fillna(0).values

    shifted_shots = df.groupby("player_id")["shots"].shift(1)
    df["shots_trend_5g"] = df.groupby("player_id")["shots"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0.0,
            raw=True,
        )
    )

    df["_prev_sc_align"] = df.groupby("player_id")["scored"].shift(1)
    df["games_since_last_goal"] = (
        df.groupby("player_id", group_keys=False)["_prev_sc_align"]
        .transform(lambda s: _games_since_last_goal_block(s.to_numpy(dtype=float)))
    )
    df.drop(columns=["_prev_sc_align"], inplace=True)

    if "pp_toi_seconds" in df.columns:
        df["pp_toi_trend_5g"] = df.groupby("player_id")["pp_toi_seconds"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0.0,
                raw=True,
            )
        )

    if "oz_start_pct" in df.columns:
        df["oz_start_pct_trend_5g"] = df.groupby("player_id")["oz_start_pct"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0.0,
                raw=True,
            )
        )

    return df


def _enhanced_opponent_features(df: pd.DataFrame, tables: dict, windows: list[int]) -> pd.DataFrame:
    """Enhanced opponent defensive features using enrichment data."""
    tgs = tables["team_game_stats"].copy()
    games = _games_competitive_for_joins(tables)
    tgs = tgs.merge(games, on="game_id", how="inner")
    tgs = tgs.sort_values(["team_id", "game_date"]).copy()

    for w in windows:
        grp = tgs.groupby("team_id")

        if "pk_pct" in tgs.columns:
            shifted = grp["pk_pct"].shift(1)
            tgs[f"opp_pk_pct_actual_{w}g"] = shifted.rolling(w, min_periods=1).mean().values

        if "goals_5v5" in tgs.columns:
            shifted = grp["goals_5v5"].shift(1)
            tgs[f"opp_goals_5v5_avg_{w}g"] = shifted.rolling(w, min_periods=1).mean().values

        if "goals_against_5v5" in tgs.columns:
            shifted = grp["goals_against_5v5"].shift(1)
            tgs[f"opp_ga_5v5_avg_{w}g"] = shifted.rolling(w, min_periods=1).mean().values

        if "pp_opportunities_actual" in tgs.columns:
            shifted = grp["pp_opportunities_actual"].shift(1)
            tgs[f"opp_pp_opps_avg_{w}g"] = shifted.rolling(w, min_periods=1).mean().values

        if "times_shorthanded" in tgs.columns:
            shifted = grp["times_shorthanded"].shift(1)
            tgs[f"opp_times_sh_avg_{w}g"] = shifted.rolling(w, min_periods=1).mean().values

    enh_cols = [c for c in tgs.columns if c.startswith("opp_") and "_avg_" in c]
    if not enh_cols:
        return df

    opp_feats = tgs[["team_id", "game_id"] + enh_cols].copy()
    opp_feats = opp_feats.rename(columns={"team_id": "opponent_team_id"})

    df = df.merge(opp_feats, on=["opponent_team_id", "game_id"], how="left")
    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction terms between player scoring ability and opponent weakness."""
    if "goals_per_60_10g" in df.columns and "opp_ga_avg_10g" in df.columns:
        df["player_g60_x_opp_ga"] = (
            df["goals_per_60_10g"] * df["opp_ga_avg_10g"].fillna(3.0)
        )

    if "pp_goals_avg_10g" in df.columns and "opp_pk_pct_10g" in df.columns:
        df["player_ppg_x_opp_pk_weak"] = (
            df["pp_goals_avg_10g"] * (1.0 - df["opp_pk_pct_10g"].fillna(0.8))
        )

    if "pp_toi_seconds_avg_10g" in df.columns and "opp_times_sh_avg_10g" in df.columns:
        df["pp_toi_x_opp_penalties"] = (
            df["pp_toi_seconds_avg_10g"].fillna(0)
            * df["opp_times_sh_avg_10g"].fillna(3.0)
        )

    if "corsi_pct_10g" in df.columns and "opp_ga_5v5_avg_10g" in df.columns:
        df["corsi_x_opp_ga_5v5"] = (
            df["corsi_pct_10g"].fillna(0.5) * df["opp_ga_5v5_avg_10g"].fillna(2.5)
        )

    if "shots_per_60_10g" in df.columns and "vs_opp_goals_per_game" in df.columns:
        df["shot_rate_x_vs_opp"] = (
            df["shots_per_60_10g"].fillna(0) * df["vs_opp_goals_per_game"].fillna(0)
        )

    if "shots_per_60_10g" in df.columns and "opponent_rest_days" in df.columns:
        df["sh60_x_opp_fatigue"] = (
            df["shots_per_60_10g"].fillna(0)
            * (7.0 - df["opponent_rest_days"].fillna(7).clip(0, 7))
        )

    if "is_home" in df.columns and "opponent_is_back_to_back" in df.columns:
        df["home_x_opp_b2b"] = (
            df["is_home"].astype(float)
            * df["opponent_is_back_to_back"].astype(float)
        )

    if "vs_opp_goals_per_game" in df.columns and "is_home" in df.columns:
        df["vs_opp_goals_x_home"] = (
            df["vs_opp_goals_per_game"].fillna(0) * df["is_home"].astype(float)
        )

    return df


def _build_upcoming_game_base(tables: dict) -> pd.DataFrame:
    """Build rows for upcoming (not yet played) games using last known rosters.

    For each upcoming game, we create one row per skater who played in that
    team's most recent completed game.
    """
    games = tables["games"].copy()
    pgs = tables["player_game_stats"].copy()
    players = tables["players"].copy()

    upcoming = games[~games["game_state"].isin(["FINAL", "OFF"])].copy()
    upcoming = upcoming[upcoming["game_type"].isin(COMPETITIVE_GAME_TYPES)].copy()
    if upcoming.empty:
        return pd.DataFrame()

    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"])

    goalies = set(players.loc[players["position"] == "G", "player_id"])
    completed = games[games["game_state"].isin(["FINAL", "OFF"])].copy()
    completed = completed[completed["game_type"].isin(COMPETITIVE_GAME_TYPES)].copy()
    completed["game_date"] = pd.to_datetime(completed["game_date"])

    rows = []
    for _, game in upcoming.iterrows():
        for side, team_col, opp_col in [
            ("home", "home_team_id", "away_team_id"),
            ("away", "away_team_id", "home_team_id"),
        ]:
            tid = game[team_col]
            team_games = completed[
                (completed["home_team_id"] == tid) | (completed["away_team_id"] == tid)
            ].sort_values("game_date")
            if team_games.empty:
                continue
            last_game_id = team_games.iloc[-1]["game_id"]
            roster = pgs[
                (pgs["game_id"] == last_game_id) & (pgs["team_id"] == tid)
            ]["player_id"].unique()
            roster = [p for p in roster if p not in goalies]
            for pid in roster:
                rows.append({
                    "player_id": pid,
                    "game_id": game["game_id"],
                    "team_id": tid,
                    "goals": 0, "assists": 0, "points": 0,
                    "shots": 0, "hits": 0, "blocked_shots": 0,
                    "pim": 0, "plus_minus": 0, "toi_seconds": 0,
                    "pp_toi_seconds": 0, "sh_toi_seconds": 0,
                    "ev_toi_seconds": 0, "pp_goals": 0, "sh_goals": 0,
                    "gw_goals": 0, "ot_goals": 0,
                    "faceoff_wins": 0, "faceoff_losses": 0,
                    "takeaways": 0, "giveaways": 0,
                    "game_date": game["game_date"],
                    "season": game["season"],
                    "home_team_id": game["home_team_id"],
                    "away_team_id": game["away_team_id"],
                    "game_type": game["game_type"],
                    "scored": 0,
                    "is_home": int(tid == game["home_team_id"]),
                    "opponent_team_id": game[opp_col],
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_feature_matrix_with_upcoming(seasons: list[int] = None) -> pd.DataFrame:
    """Build feature matrix that includes upcoming (unplayed) games.

    Upcoming game rows get features based entirely on historical data.
    """
    cfg = load_config()
    windows = cfg["model"]["rolling_windows"]

    logger.info("Loading tables...")
    tables = _load_tables()

    logger.info("Building base dataset (completed games)...")
    df = _build_player_game_base(tables)

    upcoming_df = _build_upcoming_game_base(tables)
    if not upcoming_df.empty:
        logger.info("Adding %d upcoming game-player rows", len(upcoming_df))
        df = pd.concat([df, upcoming_df], ignore_index=True)

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    if seasons:
        df = df[df["season"].isin(seasons)].copy()
        logger.info("Filtered to seasons %s: %d rows", seasons, len(df))

    df = _run_full_feature_pipeline(df, tables, windows)

    logger.info("Feature matrix (with upcoming): %d rows", len(df))
    return df


def _run_full_feature_pipeline(
    df: pd.DataFrame, tables: dict, windows: list[int]
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""
    logger.info("Computing rolling player features...")
    df = _rolling_player_features(df, windows)

    logger.info("Computing season player features...")
    df = _season_player_features(df)

    logger.info("Computing team features...")
    df = _team_rolling_features(df, tables, windows)

    logger.info("Computing PP share-of-team deployment features...")
    df = _pp_share_of_team_features(df, windows)

    logger.info("Computing teammate offensive context...")
    df = _teammate_strength_features(df)

    logger.info("Computing opponent features...")
    df = _opponent_rolling_features(df, tables, windows)

    logger.info("Computing goalie features...")
    df = _goalie_features(df, tables, windows)

    logger.info("Computing game context features...")
    df = _game_context_features(df)

    logger.info("Computing opponent schedule context...")
    df = _opponent_schedule_context(df, tables)

    logger.info("Computing player profile features...")
    df = _player_profile_features(df, tables)

    logger.info("Computing home/road split features...")
    df = _home_road_split_features(df, windows)

    logger.info("Computing vs-opponent features...")
    df = _vs_opponent_features(df)

    logger.info("Computing streak features...")
    df = _streak_features(df)

    logger.info("Computing enhanced opponent features...")
    df = _enhanced_opponent_features(df, tables, windows)

    logger.info("Computing interaction features...")
    df = _interaction_features(df)

    return df


RAW_STAT_COLS = [
    "goals", "assists", "points", "shots", "hits", "blocked_shots",
    "pim", "plus_minus", "toi_seconds", "pp_toi_seconds", "sh_toi_seconds",
    "ev_toi_seconds", "pp_goals", "sh_goals", "gw_goals", "ot_goals",
    "faceoff_wins", "faceoff_losses", "takeaways", "giveaways",
    "shifts", "ot_toi_seconds", "time_on_ice_per_shift",
    "corsi_for", "corsi_against", "fenwick_for", "fenwick_against",
    "oz_start_pct", "dz_start_pct", "individual_corsi_for",
    "individual_shots_for_per60", "on_ice_shooting_pct", "on_ice_save_pct",
    "pdo", "pp_shots", "pp_individual_sat_for", "pp_toi_pct_per_game",
    "missed_shots", "total_shot_attempts", "first_goals", "empty_net_goals",
    "es_goals_for", "es_goals_against",
    "oz_faceoffs", "dz_faceoffs", "nz_faceoffs", "total_faceoffs",
    "ev_faceoff_pct",
]

NON_FEATURE_COLS = [
    "player_id", "game_id", "team_id", "game_date", "season",
    "home_team_id", "away_team_id", "opponent_team_id", "game_type",
    "scored", "opp_goalie_id",
] + RAW_STAT_COLS


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

    df = _run_full_feature_pipeline(df, tables, windows)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info("Feature matrix: %d rows x %d features", len(df), len(feature_cols))
    logger.info("Features: %s", feature_cols)

    return df
