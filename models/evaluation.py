"""Model evaluation with calibration analysis, segmentation, and lift charts."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger("models.evaluation")


def compute_core_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute log loss, Brier score, and ROC-AUC."""
    return {
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "n_samples": len(y_true),
        "base_rate": y_true.mean(),
        "mean_predicted": y_prob.mean(),
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Bucketed calibration: predicted vs actual scoring rate per decile."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        count=("y_true", "count"),
        actual_rate=("y_true", "mean"),
        predicted_mean=("y_prob", "mean"),
        predicted_min=("y_prob", "min"),
        predicted_max=("y_prob", "max"),
    ).reset_index()
    grouped["abs_error"] = (grouped["predicted_mean"] - grouped["actual_rate"]).abs()
    return grouped


def lift_table(y_true: np.ndarray, y_prob: np.ndarray,
               n_bins: int = 10) -> pd.DataFrame:
    """Compute lift by probability decile."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=n_bins, labels=False) + 1

    base_rate = y_true.mean()
    grouped = df.groupby("decile").agg(
        count=("y_true", "count"),
        actual_rate=("y_true", "mean"),
        predicted_mean=("y_prob", "mean"),
    ).reset_index()
    grouped["lift"] = grouped["actual_rate"] / base_rate
    grouped["cumulative_actual"] = (
        df.groupby("decile")["y_true"].sum().cumsum().values
        / y_true.sum()
    )
    return grouped


def monotonicity_check(cal_table: pd.DataFrame) -> dict:
    """Check if calibration buckets show monotonic increase in actual rate."""
    rates = cal_table["actual_rate"].values
    is_monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    violations = sum(1 for i in range(len(rates) - 1) if rates[i] > rates[i + 1])
    return {
        "is_monotonic": is_monotonic,
        "violations": violations,
        "total_bins": len(rates),
    }


def evaluate_model(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Full evaluation suite for a model."""
    logger.info("Evaluating %s...", name)

    metrics = compute_core_metrics(y_true, y_prob)
    logger.info(
        "%s — Log Loss: %.4f | Brier: %.4f | AUC: %.4f | Base Rate: %.4f",
        name, metrics["log_loss"], metrics["brier_score"],
        metrics["roc_auc"], metrics["base_rate"],
    )

    cal = calibration_table(y_true, y_prob)
    logger.info("%s calibration table:\n%s", name, cal.to_string(index=False))

    mono = monotonicity_check(cal)
    logger.info(
        "%s monotonicity: %s (%d violations / %d bins)",
        name, mono["is_monotonic"], mono["violations"], mono["total_bins"],
    )

    lift = lift_table(y_true, y_prob)
    logger.info("%s lift table:\n%s", name, lift.to_string(index=False))

    return {
        "name": name,
        "metrics": metrics,
        "calibration": cal,
        "monotonicity": mono,
        "lift": lift,
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Side-by-side comparison of model metrics."""
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "model": r["name"],
            "log_loss": m["log_loss"],
            "brier_score": m["brier_score"],
            "roc_auc": m["roc_auc"],
            "base_rate": m["base_rate"],
            "mean_predicted": m["mean_predicted"],
            "monotonic": r["monotonicity"]["is_monotonic"],
        })
    comparison = pd.DataFrame(rows)
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))
    return comparison
