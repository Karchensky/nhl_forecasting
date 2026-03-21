"""Model training pipeline for NHL goal probability prediction.

Supports rolling-window cross-validation, time-weighted samples,
and Optuna hyperparameter tuning.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import optuna
import xgboost as xgb

from models.feature_engineering import NON_FEATURE_COLS, build_feature_matrix
from utils.config import load_config, PROJECT_ROOT
from utils.logger import get_logger

logger = get_logger("models.training")

MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS_TRAINING = list(NON_FEATURE_COLS)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS_TRAINING]


# ---------------------------------------------------------------------------
# Time-weighted sample generation
# ---------------------------------------------------------------------------

def compute_sample_weights(df: pd.DataFrame, half_life_days: int = 365) -> np.ndarray:
    """Exponential decay weights based on game recency.

    More recent games get higher weight. half_life_days controls the decay rate.
    """
    max_date = df["game_date"].max()
    days_ago = (max_date - df["game_date"]).dt.days.values.astype(float)
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights


# ---------------------------------------------------------------------------
# Rolling-window cross-validation
# ---------------------------------------------------------------------------

def rolling_window_cv_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_games: int = 500,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate temporally ordered train/val splits for rolling-window CV.

    Splits are based on game_date ordering. Each fold uses everything before
    the validation window as training data.
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    unique_dates = sorted(df["game_date"].unique())
    n_dates = len(unique_dates)

    val_size = max(1, n_dates // (n_folds + 1))
    splits = []

    for fold in range(n_folds):
        val_start_idx = n_dates - (n_folds - fold) * val_size
        val_end_idx = val_start_idx + val_size

        if val_start_idx < min_train_games:
            continue

        val_start_date = unique_dates[val_start_idx]
        val_end_date = unique_dates[min(val_end_idx, n_dates - 1)]

        train_mask = df["game_date"] < val_start_date
        val_mask = (df["game_date"] >= val_start_date) & (df["game_date"] <= val_end_date)

        train_split = df[train_mask].copy()
        val_split = df[val_mask].copy()

        if len(train_split) > 0 and len(val_split) > 0:
            splits.append((train_split, val_split))

    logger.info("Generated %d rolling-window CV folds", len(splits))
    return splits


# ---------------------------------------------------------------------------
# Optuna objective for LightGBM
# ---------------------------------------------------------------------------

def _lgb_optuna_objective(
    trial: optuna.Trial,
    cv_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    use_weights: bool = True,
    half_life_days: int = 365,
) -> float:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "verbose": -1,
        "seed": 42,
    }

    fold_losses = []
    for train_df, val_df in cv_splits:
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["scored"].values
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["scored"].values

        w = compute_sample_weights(train_df, half_life_days) if use_weights else None

        dtrain = lgb.Dataset(X_train, label=y_train, weight=w)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        val_probs = model.predict(X_val)
        fold_losses.append(log_loss(y_val, val_probs))

    return np.mean(fold_losses)


def tune_lightgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int = 50,
    n_folds: int = 5,
    half_life_days: int = 365,
) -> dict:
    """Run Optuna hyperparameter tuning for LightGBM."""
    logger.info("Starting Optuna tuning for LightGBM (%d trials)...", n_trials)

    cv_splits = rolling_window_cv_splits(df, n_folds=n_folds)
    if not cv_splits:
        logger.warning("No valid CV splits generated, using default params.")
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="lgb_tune")

    study.optimize(
        lambda trial: _lgb_optuna_objective(
            trial, cv_splits, feature_cols,
            use_weights=True, half_life_days=half_life_days,
        ),
        n_trials=n_trials,
    )

    logger.info("Best trial: logloss=%.5f, params=%s",
                study.best_trial.value, study.best_trial.params)
    return study.best_trial.params


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

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
    """Train a logistic regression baseline model with time weights."""
    logger.info("Training Logistic Regression baseline...")

    X_train = train[feature_cols].fillna(0).values
    y_train = train["scored"].values
    X_val = val[feature_cols].fillna(0).values
    y_val = val["scored"].values

    weights = compute_sample_weights(train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    )
    model.fit(X_train_s, y_train, sample_weight=weights)

    train_probs = model.predict_proba(X_train_s)[:, 1]
    val_probs = model.predict_proba(X_val_s)[:, 1]

    logger.info("LR train logloss: %.4f, val logloss: %.4f",
                log_loss(y_train, train_probs), log_loss(y_val, val_probs))

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


def train_lightgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    tuned_params: dict | None = None,
    half_life_days: int = 365,
) -> dict:
    """Train a LightGBM model with time weights and optional tuned params."""
    logger.info("Training LightGBM...")

    X_train = train[feature_cols].fillna(0)
    y_train = train["scored"].values
    X_val = val[feature_cols].fillna(0)
    y_val = val["scored"].values

    weights = compute_sample_weights(train, half_life_days)

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    default_params = {
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

    if tuned_params:
        default_params.update(tuned_params)
        default_params["metric"] = ["binary_logloss", "auc"]
        default_params["verbose"] = -1
        default_params["seed"] = 42

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    raw_train_probs = model.predict(X_train)
    raw_val_probs = model.predict(X_val)

    logger.info("LGB raw train logloss: %.4f, val logloss: %.4f",
                log_loss(y_train, raw_train_probs), log_loss(y_val, raw_val_probs))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_train_probs, y_train)

    cal_train_probs = iso.predict(raw_train_probs)
    cal_val_probs = iso.predict(raw_val_probs)

    logger.info("LGB calibrated val logloss: %.4f, AUC: %.4f",
                log_loss(y_val, cal_val_probs), roc_auc_score(y_val, cal_val_probs))

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


def train_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    half_life_days: int = 365,
) -> dict:
    """Train an XGBoost model with time weights and calibration."""
    logger.info("Training XGBoost...")

    X_train = train[feature_cols].fillna(0)
    y_train = train["scored"].values
    X_val = val[feature_cols].fillna(0)
    y_val = val["scored"].values

    weights = compute_sample_weights(train, half_life_days)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights, feature_names=feature_cols)
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

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_train_probs, y_train)

    cal_train_probs = iso.predict(raw_train_probs)
    cal_val_probs = iso.predict(raw_val_probs)

    logger.info("XGB calibrated val logloss: %.4f, AUC: %.4f",
                log_loss(y_val, cal_val_probs), roc_auc_score(y_val, cal_val_probs))

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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_training(tune: bool = True, n_trials: int = 50):
    """Full training pipeline with optional Optuna tuning."""
    logger.info("Building feature matrix...")
    df = build_feature_matrix()
    feature_cols = get_feature_columns(df)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    train, val, test = prepare_splits(df)

    tuned_params = {}
    if tune:
        train_val = pd.concat([train, val], ignore_index=True)
        tuned_params = tune_lightgbm(
            train_val, feature_cols, n_trials=n_trials, n_folds=5
        )

    results = {}
    results["logistic_regression"] = train_logistic_baseline(train, val, feature_cols)
    results["lightgbm"] = train_lightgbm(
        train, val, feature_cols, tuned_params=tuned_params
    )
    results["xgboost"] = train_xgboost(train, val, feature_cols)

    return results, df, train, val, test


if __name__ == "__main__":
    import sys
    do_tune = "--no-tune" not in sys.argv
    run_training(tune=do_tune)
