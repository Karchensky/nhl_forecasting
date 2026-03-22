"""Inference module: generate predictions for upcoming games."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

from database.db_client import get_session
from database.ingestion import upsert_model_output
from models.feature_engineering import build_feature_matrix
from models.training import load_model
from utils.logger import get_logger

logger = get_logger("models.inference")


def predict_with_model(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using a saved model."""
    saved = load_model(model_name)
    feature_cols = saved["feature_cols"]
    X = df[feature_cols].fillna(0)

    if model_name == "logistic_regression":
        scaler = saved["scaler"]
        model = saved["model"]
        X_scaled = scaler.transform(X.values)
        probs = model.predict_proba(X_scaled)[:, 1]
    elif model_name == "lightgbm":
        model = saved["model"]
        raw_probs = model.predict(X)
        calibrator = saved.get("calibrator")
        probs = calibrator.predict(raw_probs) if calibrator else raw_probs
    elif model_name == "xgboost":
        model = saved["model"]
        dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
        raw_probs = model.predict(dmatrix)
        calibrator = saved.get("calibrator")
        probs = calibrator.predict(raw_probs) if calibrator else raw_probs
    else:
        raise ValueError(f"Unknown model: {model_name}")

    result = df[["player_id", "game_id", "team_id", "game_date", "season",
                  "is_home", "opponent_team_id"]].copy()
    result["predicted_probability"] = probs
    result["model_version"] = model_name

    return result


def predict_upcoming(model_name: str = "lightgbm") -> pd.DataFrame:
    """Predict goal probabilities for upcoming/today's games.

    Builds features for the current season (completed games provide the
    rolling history) and then generates predictions for all rows, including
    any upcoming games that have been ingested into the games table.
    """
    from utils.config import load_config
    from models.feature_engineering import build_feature_matrix_with_upcoming

    cfg = load_config()
    test_season = cfg["model"]["test_season"]

    logger.info("Building features for season %d (with upcoming)...", test_season)
    try:
        df = build_feature_matrix_with_upcoming(seasons=[test_season])
    except Exception:
        logger.info("Falling back to standard feature matrix...")
        df = build_feature_matrix(seasons=[test_season])

    if df.empty:
        logger.warning("No data for season %d", test_season)
        return pd.DataFrame()

    today = date.today()
    tomorrow = today + timedelta(days=1)
    logger.info("Total rows: %d, date range: %s to %s",
                len(df), df["game_date"].min(), df["game_date"].max())

    gd = pd.to_datetime(df["game_date"]).dt.normalize()
    horizon_mask = (gd.dt.date >= today) & (gd.dt.date <= tomorrow)
    df_horizon = df[horizon_mask].copy()
    if df_horizon.empty:
        logger.warning(
            "No rows for today/tomorrow in season %s; nothing to predict.",
            test_season,
        )
        return pd.DataFrame()

    logger.info("Predicting for %d rows (today + next day only)", len(df_horizon))
    predictions = predict_with_model(model_name, df_horizon)
    return predictions


def store_predictions(predictions: pd.DataFrame):
    """Store predictions in the model_outputs table."""
    if predictions.empty:
        return
    BATCH = 500
    total = len(predictions)
    for start in range(0, total, BATCH):
        chunk = predictions.iloc[start:start + BATCH]
        with get_session() as session:
            for _, row in chunk.iterrows():
                upsert_model_output(session, {
                    "player_id": int(row["player_id"]),
                    "game_id": int(row["game_id"]),
                    "model_version": row["model_version"],
                    "predicted_probability": float(row["predicted_probability"]),
                })
    logger.info("Stored %d predictions", total)
