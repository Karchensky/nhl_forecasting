"""Model training pipeline for NHL goal probability prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from models.feature_engineering import build_feature_matrix
from utils.config import load_config, PROJECT_ROOT
from utils.logger import get_logger

logger = get_logger("models.training")

MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = [
    "player_id", "game_id", "team_id", "game_date", "season",
    "home_team_id", "away_team_id", "opponent_team_id", "game_type",
    "goals", "assists", "points", "shots", "hits", "blocked_shots",
    "pim", "plus_minus", "toi_seconds", "pp_toi_seconds", "sh_toi_seconds",
    "ev_toi_seconds", "pp_goals", "sh_goals", "gw_goals", "ot_goals",
    "faceoff_wins", "faceoff_losses", "takeaways", "giveaways",
    "scored", "opp_goalie_id",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by season into train / validation / test sets."""
    cfg = load_config()
    train_seasons = cfg["model"]["train_seasons"]
    val_season = cfg["model"]["validation_season"]
    test_season = cfg["model"]["test_season"]

    train = df[df["season"].isin(train_seasons)].copy()
    val = df[df["season"] == val_season].copy()
    test = df[df["season"] == test_season].copy()

    logger.info("Train: %d rows (%s)", len(train), train_seasons)
    logger.info("Val:   %d rows (%d)", len(val), val_season)
    logger.info("Test:  %d rows (%d)", len(test), test_season)

    return train, val, test


def train_logistic_baseline(train: pd.DataFrame, val: pd.DataFrame,
                            feature_cols: list[str]) -> dict:
    """Train a logistic regression baseline model."""
    logger.info("Training Logistic Regression baseline...")

    X_train = train[feature_cols].fillna(0).values
    y_train = train["scored"].values
    X_val = val[feature_cols].fillna(0).values
    y_val = val["scored"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    )
    model.fit(X_train_s, y_train)

    train_probs = model.predict_proba(X_train_s)[:, 1]
    val_probs = model.predict_proba(X_val_s)[:, 1]

    logger.info("LR train mean prob: %.4f, val mean prob: %.4f",
                train_probs.mean(), val_probs.mean())

    result = {
        "name": "logistic_regression",
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "train_probs": train_probs,
        "val_probs": val_probs,
        "y_train": y_train,
        "y_val": y_val,
    }

    _save_model(result, "logistic_regression")
    return result


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame,
                   feature_cols: list[str]) -> dict:
    """Train a LightGBM model with calibration."""
    logger.info("Training LightGBM...")

    X_train = train[feature_cols].fillna(0)
    y_train = train["scored"].values
    X_val = val[feature_cols].fillna(0)
    y_val = val["scored"].values

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    raw_train_probs = model.predict(X_train)
    raw_val_probs = model.predict(X_val)

    logger.info("LGB raw train mean: %.4f, val mean: %.4f",
                raw_train_probs.mean(), raw_val_probs.mean())

    # Calibrate with isotonic regression
    logger.info("Calibrating LightGBM with isotonic regression...")
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_train_probs, y_train)

    cal_train_probs = iso.predict(raw_train_probs)
    cal_val_probs = iso.predict(raw_val_probs)

    logger.info("LGB calibrated train mean: %.4f, val mean: %.4f",
                cal_train_probs.mean(), cal_val_probs.mean())

    result = {
        "name": "lightgbm",
        "model": model,
        "calibrator": iso,
        "feature_cols": feature_cols,
        "train_probs": cal_train_probs,
        "val_probs": cal_val_probs,
        "raw_train_probs": raw_train_probs,
        "raw_val_probs": raw_val_probs,
        "y_train": y_train,
        "y_val": y_val,
    }

    _save_model(result, "lightgbm")
    return result


def train_xgboost(train: pd.DataFrame, val: pd.DataFrame,
                  feature_cols: list[str]) -> dict:
    """Train an XGBoost model with calibration."""
    logger.info("Training XGBoost...")

    X_train = train[feature_cols].fillna(0)
    y_train = train["scored"].values
    X_val = val[feature_cols].fillna(0)
    y_val = val["scored"].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    raw_train_probs = model.predict(dtrain)
    raw_val_probs = model.predict(dval)

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_train_probs, y_train)

    cal_train_probs = iso.predict(raw_train_probs)
    cal_val_probs = iso.predict(raw_val_probs)

    logger.info("XGB calibrated train mean: %.4f, val mean: %.4f",
                cal_train_probs.mean(), cal_val_probs.mean())

    result = {
        "name": "xgboost",
        "model": model,
        "calibrator": iso,
        "feature_cols": feature_cols,
        "train_probs": cal_train_probs,
        "val_probs": cal_val_probs,
        "y_train": y_train,
        "y_val": y_val,
    }

    _save_model(result, "xgboost")
    return result


def _save_model(result: dict, name: str):
    path = MODEL_DIR / f"{name}.pkl"
    to_save = {k: v for k, v in result.items()
               if k not in ("train_probs", "val_probs", "raw_train_probs",
                            "raw_val_probs", "y_train", "y_val")}
    with open(path, "wb") as f:
        pickle.dump(to_save, f)
    logger.info("Saved model to %s", path)


def load_model(name: str) -> dict:
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training():
    """Full training pipeline."""
    logger.info("Building feature matrix...")
    df = build_feature_matrix()
    feature_cols = get_feature_columns(df)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    train, val, test = prepare_splits(df)

    results = {}
    results["logistic_regression"] = train_logistic_baseline(train, val, feature_cols)
    results["lightgbm"] = train_lightgbm(train, val, feature_cols)
    results["xgboost"] = train_xgboost(train, val, feature_cols)

    return results, df, train, val, test


if __name__ == "__main__":
    run_training()
