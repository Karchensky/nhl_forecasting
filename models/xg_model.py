"""Expected Goals (xG) model: shot-level binary classifier.

Predicts the probability that a given shot attempt becomes a goal,
using spatial features (distance, angle), shot type, game state
(period, situation code), and shooter context.

Usage:
    python -m models.xg_model              # train + evaluate
    python -m models.xg_model --evaluate   # evaluate saved model only
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_client import get_engine
from utils.config import PROJECT_ROOT, load_config
from utils.logger import get_logger

logger = get_logger("models.xg_model")

MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

XG_MODEL_PATH = MODEL_DIR / "xg_model.pkl"

# ---------------------------------------------------------------------------
# Feature engineering for shot-level data
# ---------------------------------------------------------------------------

# Shot type one-hot encoding order
SHOT_TYPES = [
    "wrist", "snap", "slap", "backhand", "tip-in",
    "deflected", "wrap-around", "poke", "bat", "between-legs", "cradle",
]


def _parse_situation_code(code: str | None) -> dict:
    """Parse NHL situation code (e.g., '1551') into strength features.

    Format: [away_goalie][away_skaters][home_skaters][home_goalie]
    """
    if not code or len(str(code)) != 4:
        return {
            "shooter_skaters": 5, "opponent_skaters": 5,
            "is_pp": 0, "is_sh": 0, "is_empty_net": 0,
        }
    s = str(code)
    away_g, away_sk, home_sk, home_g = int(s[0]), int(s[1]), int(s[2]), int(s[3])
    # We don't know which side the shooter is on from the code alone;
    # this gets resolved in _build_xg_features using team context.
    return {
        "away_goalie": away_g, "away_skaters": away_sk,
        "home_skaters": home_sk, "home_goalie": home_g,
    }


def _load_shot_data() -> pd.DataFrame:
    """Load shot events joined with game context."""
    engine = get_engine()
    query = """
        SELECT
            se.*,
            g.season,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score
        FROM shot_events se
        JOIN games g ON se.game_id = g.game_id
        WHERE g.game_state IN ('FINAL', 'OFF')
          AND g.game_type IN (2, 3)
        ORDER BY g.game_date, se.game_id, se.event_id
    """
    df = pd.read_sql(query, engine)
    logger.info("Loaded %d shot events from %d games",
                len(df), df["game_id"].nunique())
    return df


def _build_xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for xG model from raw shot events."""
    feat = pd.DataFrame(index=df.index)

    # Spatial features
    feat["distance"] = df["distance"].fillna(df["distance"].median())
    feat["angle"] = df["angle"].fillna(df["angle"].median())
    feat["distance_sq"] = feat["distance"] ** 2
    feat["angle_sq"] = feat["angle"] ** 2
    feat["dist_x_angle"] = feat["distance"] * feat["angle"]

    # Raw coordinates (normalized: offensive zone always positive x)
    feat["x_coord"] = df["x_coord"].abs().fillna(0)
    feat["y_coord"] = df["y_coord"].fillna(0).abs()

    # Shot type one-hot
    for st in SHOT_TYPES:
        feat[f"shot_{st}"] = (df["shot_type"] == st).astype(int)
    feat["shot_type_missing"] = df["shot_type"].isna().astype(int)

    # Period and time
    feat["period"] = df["period"].clip(1, 5)
    feat["time_in_period"] = df["time_in_period_seconds"].fillna(0)
    feat["is_overtime"] = (df["period_type"] == "OT").astype(int)

    # Situation / strength
    sit_data = df["situation_code"].apply(_parse_situation_code).apply(pd.Series)

    # Determine shooter side (home vs away)
    is_home_shooter = (df["team_id"] == df["home_team_id"]).astype(int)
    feat["is_home_shooter"] = is_home_shooter

    shooter_skaters = np.where(
        is_home_shooter, sit_data["home_skaters"], sit_data["away_skaters"]
    )
    opp_skaters = np.where(
        is_home_shooter, sit_data["away_skaters"], sit_data["home_skaters"]
    )
    opp_goalie = np.where(
        is_home_shooter, sit_data["away_goalie"], sit_data["home_goalie"]
    )

    feat["shooter_skaters"] = shooter_skaters
    feat["opponent_skaters"] = opp_skaters
    feat["skater_diff"] = shooter_skaters - opp_skaters
    feat["is_pp"] = (feat["skater_diff"] > 0).astype(int)
    feat["is_sh"] = (feat["skater_diff"] < 0).astype(int)
    feat["is_empty_net"] = (opp_goalie == 0).astype(int)

    # Empty net indicator from goalie_id (more reliable)
    feat["goalie_absent"] = df["goalie_id"].isna().astype(int)

    # Zone code
    feat["is_offensive_zone"] = (df["zone_code"] == "O").astype(int)

    # -----------------------------------------------------------------------
    # Shot sequence / prior-event features (sorted within each game)
    # -----------------------------------------------------------------------
    df_sorted = df.sort_values(["game_id", "event_id"]).copy()
    game_grp = df_sorted.groupby("game_id")

    prev_time = game_grp["time_in_period_seconds"].shift(1)
    prev_period = game_grp["period"].shift(1)
    same_period = df_sorted["period"] == prev_period

    # Time since last shot (any team) — continuous, capped at 120s
    time_diff = df_sorted["time_in_period_seconds"] - prev_time
    time_since_last = np.where(same_period, time_diff, 120.0)
    time_since_last = np.clip(np.where(np.isnan(time_since_last), 120.0, time_since_last), 0, 120)
    feat["time_since_last_shot"] = time_since_last

    # Rebound: shot within 3 seconds of a prior shot-on-goal or save
    feat["is_rebound"] = ((same_period) & (time_diff <= 3) & (time_diff >= 0)).astype(int)

    # Rush: shot 3-5 seconds after prior event (quick transition)
    feat["is_rush"] = ((same_period) & (time_diff <= 5) & (time_diff > 3)).astype(int)

    # Prior shot was by same team (sustained pressure) vs opposite team
    prev_team = game_grp["team_id"].shift(1)
    feat["same_team_prior_shot"] = (df_sorted["team_id"] == prev_team).astype(int).values

    # Prior shot was a goal (changes game dynamics)
    prev_is_goal = game_grp["is_goal"].shift(1).fillna(0)
    feat["prior_shot_was_goal"] = prev_is_goal.astype(int).values

    # Distance change from previous shot (large negative = rush towards net)
    prev_dist = game_grp["distance"].shift(1)
    feat["distance_change"] = np.where(
        same_period,
        (df_sorted["distance"] - prev_dist).fillna(0).values,
        0.0,
    )

    # Angle change from previous shot (cross-ice passes)
    prev_angle = game_grp["angle"].shift(1)
    feat["angle_change"] = np.where(
        same_period,
        np.abs((df_sorted["angle"] - prev_angle).fillna(0).values),
        0.0,
    )

    # Shot flurry: count of shots in same game within last 10 seconds
    # (high-danger sustained offensive zone time)
    game_period_time = df_sorted["period"] * 1200 + df_sorted["time_in_period_seconds"]
    feat["shots_last_10s"] = 0  # default
    for gid, grp_idx in df_sorted.groupby("game_id").groups.items():
        gpt = game_period_time.loc[grp_idx].values
        counts = np.zeros(len(gpt), dtype=int)
        j = 0
        for i in range(len(gpt)):
            while j < i and gpt[i] - gpt[j] > 10:
                j += 1
            counts[i] = i - j  # shots in the 10s window before this one
        feat.loc[grp_idx, "shots_last_10s"] = counts

    # -----------------------------------------------------------------------
    # Score state (relative to shooting team) — derived from cumulative goals
    # in the play-by-play, NOT the final game score.
    # -----------------------------------------------------------------------
    if "home_team_id" in df.columns:
        # Build running score from goal events within each game
        is_goal_sorted = df_sorted["is_goal"].astype(int)
        is_home_team_shot = (df_sorted["team_id"] == df_sorted["home_team_id"]).astype(int)
        home_goals = (is_goal_sorted * is_home_team_shot)
        away_goals = (is_goal_sorted * (1 - is_home_team_shot))
        # Cumsum of goals BEFORE current event (shift so current goal isn't counted)
        cum_home = home_goals.groupby(df_sorted["game_id"]).cumsum() - home_goals
        cum_away = away_goals.groupby(df_sorted["game_id"]).cumsum() - away_goals
        score_diff = np.where(
            is_home_team_shot, cum_home - cum_away, cum_away - cum_home
        )
        feat["score_diff"] = np.clip(score_diff, -5, 5)
        feat["is_trailing"] = (score_diff < 0).astype(int)
        feat["is_tied"] = (score_diff == 0).astype(int)

    return feat


XG_FEATURE_COLS = None  # Set after first training


def train_xg_model(test_season: int | None = None) -> dict:
    """Train the xG model on shot-level data.

    Uses season-based train/val/test split. By default, uses the most
    recent complete season as test, prior season as validation, rest as train.
    """
    cfg = load_config()
    all_seasons = cfg["seasons"]

    df = _load_shot_data()
    if df.empty:
        logger.error("No shot events found. Run backfill_pbp first.")
        return {}

    features = _build_xg_features(df)
    target = df["is_goal"].astype(int)

    feature_cols = list(features.columns)

    # Season split
    if test_season is None:
        test_season = all_seasons[-1]

    val_season = all_seasons[-2] if len(all_seasons) > 1 else test_season
    train_seasons = [s for s in all_seasons if s != test_season and s != val_season]
    if not train_seasons:
        train_seasons = [s for s in all_seasons if s != test_season]

    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"] == val_season
    test_mask = df["season"] == test_season

    X_train = features.loc[train_mask, feature_cols]
    y_train = target.loc[train_mask]
    X_val = features.loc[val_mask, feature_cols]
    y_val = target.loc[val_mask]
    X_test = features.loc[test_mask, feature_cols]
    y_test = target.loc[test_mask]

    logger.info("Train: %d shots (%d goals, %.2f%% rate) from seasons %s",
                len(X_train), y_train.sum(),
                100 * y_train.mean(), train_seasons)
    logger.info("Val:   %d shots from season %d", len(X_val), val_season)
    logger.info("Test:  %d shots from season %d", len(X_test), test_season)

    # LightGBM
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    # Calibrate on validation set
    val_preds_raw = model.predict(X_val)
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_preds_raw, y_val)

    # Evaluate
    results = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        preds_raw = model.predict(X)
        preds_cal = calibrator.predict(preds_raw)
        preds_cal = np.clip(preds_cal, 1e-7, 1 - 1e-7)

        auc = roc_auc_score(y, preds_cal)
        ll = log_loss(y, preds_cal)
        brier = brier_score_loss(y, preds_cal)

        results[name] = {"auc": auc, "log_loss": ll, "brier": brier}
        logger.info("xG %s: AUC=%.4f  LogLoss=%.4f  Brier=%.4f  mean_pred=%.4f  actual=%.4f",
                     name, auc, ll, brier, preds_cal.mean(), y.mean())

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info("Top 15 xG features:\n%s", imp.head(15).to_string(index=False))

    # Save
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "results": results,
    }
    with open(XG_MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Saved xG model to %s", XG_MODEL_PATH)

    return results


def load_xg_model() -> dict | None:
    """Load the saved xG model artifact."""
    if not XG_MODEL_PATH.exists():
        logger.warning("No xG model found at %s", XG_MODEL_PATH)
        return None
    with open(XG_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_xg(shot_df: pd.DataFrame) -> np.ndarray:
    """Predict xG for a DataFrame of shot events.

    Returns array of calibrated xG probabilities.
    """
    artifact = load_xg_model()
    if artifact is None:
        raise FileNotFoundError("No trained xG model found. Run train_xg_model() first.")

    model = artifact["model"]
    calibrator = artifact["calibrator"]
    feature_cols = artifact["feature_cols"]

    features = _build_xg_features(shot_df)
    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0

    preds_raw = model.predict(features[feature_cols])
    preds_cal = calibrator.predict(preds_raw)
    return np.clip(preds_cal, 1e-7, 1 - 1e-7)


def compute_player_xg_totals(season: int | None = None) -> pd.DataFrame:
    """Compute per-player-per-game xG totals from shot events.

    Returns DataFrame with columns: player_id, game_id, xg_total, shots, goals,
    xg_per_shot, goals_above_expected.
    """
    engine = get_engine()
    season_filter = ""
    params = {}
    if season:
        season_filter = "AND g.season = :season"
        params["season"] = season

    query = f"""
        SELECT se.*, g.season, g.game_date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score
        FROM shot_events se
        JOIN games g ON se.game_id = g.game_id
        WHERE g.game_state IN ('FINAL', 'OFF')
          AND g.game_type IN (2, 3)
          {season_filter}
        ORDER BY g.game_date, se.game_id, se.event_id
    """
    from sqlalchemy import text as sa_text
    df = pd.read_sql(sa_text(query), engine, params=params)
    if df.empty:
        return pd.DataFrame()

    xg_probs = predict_xg(df)
    df["xg"] = xg_probs

    # Aggregate per shooter per game
    player_xg = (
        df.groupby(["shooter_id", "game_id"])
        .agg(
            xg_total=("xg", "sum"),
            shots=("xg", "count"),
            goals=("is_goal", "sum"),
        )
        .reset_index()
        .rename(columns={"shooter_id": "player_id"})
    )
    player_xg["xg_per_shot"] = player_xg["xg_total"] / player_xg["shots"].clip(lower=1)
    player_xg["goals_above_expected"] = player_xg["goals"] - player_xg["xg_total"]

    return player_xg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xG model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate saved model only (no training)")
    args = parser.parse_args()

    if args.evaluate:
        artifact = load_xg_model()
        if artifact:
            print("Saved model results:")
            for name, metrics in artifact["results"].items():
                print(f"  {name}: {metrics}")
        else:
            print("No saved model found.")
    else:
        train_xg_model()
