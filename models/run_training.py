"""Run the full training + evaluation pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.feature_engineering import build_feature_matrix
from models.training import (
    get_feature_columns,
    prepare_splits,
    train_logistic_baseline,
    train_lightgbm,
    train_xgboost,
)
from models.evaluation import evaluate_model, compare_models
from utils.logger import get_logger

logger = get_logger("run_training")


def main():
    logger.info("=== Building Feature Matrix ===")
    df = build_feature_matrix()
    feature_cols = get_feature_columns(df)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols[:10])

    logger.info("\n=== Preparing Train/Val/Test Splits ===")
    train, val, test = prepare_splits(df)

    if len(train) < 100:
        logger.error("Not enough training data (%d rows). Need more boxscores.", len(train))
        return

    logger.info("\n=== Training Models ===")
    results = {}

    logger.info("\n--- Logistic Regression ---")
    results["lr"] = train_logistic_baseline(train, val, feature_cols)

    logger.info("\n--- LightGBM ---")
    results["lgb"] = train_lightgbm(train, val, feature_cols)

    logger.info("\n--- XGBoost ---")
    results["xgb"] = train_xgboost(train, val, feature_cols)

    logger.info("\n=== Evaluating Models on Validation Set ===")
    eval_results = []
    for key, r in results.items():
        ev = evaluate_model(r["name"], r["y_val"], r["val_probs"])
        eval_results.append(ev)

    logger.info("\n=== Model Comparison ===")
    comparison = compare_models(eval_results)
    print("\n" + comparison.to_string(index=False))

    logger.info("\n=== Feature Importance (LightGBM) ===")
    lgb_model = results["lgb"]["model"]
    importance = lgb_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("\nTop 20 features by gain:")
    for feat, imp in feat_imp[:20]:
        print(f"  {feat:40s} {imp:.1f}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
