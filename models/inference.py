"""Inference module: generate predictions for upcoming games."""

from datetime import date, datetime

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

from database.db_client import get_session
from database.ingestion import upsert_model_output
from models.feature_engineering import build_feature_matrix
from models.training import NON_FEATURE_COLS, load_model
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
    """Predict goal probabilities for upcoming/today's games."""
    from utils.config import load_config
    cfg = load_config()
    test_season = cfg["model"]["test_season"]

    logger.info("Building features for season %d...", test_season)
    df = build_feature_matrix(seasons=[test_season])

    if df.empty:
        logger.warning("No data for season %d", test_season)
        return pd.DataFrame()

    today = date.today()
    future_mask = df["game_date"] >= pd.Timestamp(today)
    if future_mask.any():
        logger.info("Found %d future game-player rows", future_mask.sum())

    predictions = predict_with_model(model_name, df)
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
