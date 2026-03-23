"""Generate predictions and store in DB.

By default, scores only the test season (config ``model.test_season``).
Pass ``--season 20242025`` to score a specific season, or ``--all`` to
score every season (generates in-sample predictions for training data --
useful for calibration diagnostics but NOT for honest evaluation).
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.feature_engineering import build_feature_matrix
from models.inference import predict_with_model, store_predictions
from models.training import get_feature_columns
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("generate_predictions")


def main():
    parser = argparse.ArgumentParser(description="Generate and store model predictions.")
    parser.add_argument(
        "--season", type=int, default=None,
        help="Score a specific season (e.g. 20252026). Default: test season from config.",
    )
    parser.add_argument(
        "--all", action="store_true", dest="score_all",
        help="Score ALL seasons (including training). Produces in-sample predictions.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.score_all:
        seasons = None  # no filter
        logger.info("Scoring ALL seasons (includes in-sample training data)")
    elif args.season:
        seasons = [args.season]
        logger.info("Scoring season %d", args.season)
    else:
        seasons = [cfg["model"]["test_season"]]
        logger.info("Scoring test season %d", seasons[0])

    df = build_feature_matrix(seasons=seasons)
    feature_cols = get_feature_columns(df)
    logger.info("Dataset: %d rows, %d features", len(df), len(feature_cols))

    if df.empty:
        logger.warning("No data to score.")
        return

    for model_name in ("logistic_regression", "lightgbm", "xgboost"):
        logger.info("Generating predictions with %s...", model_name)
        preds = predict_with_model(model_name, df)
        logger.info("Storing %d predictions for %s", len(preds), model_name)
        store_predictions(preds)

    logger.info("All predictions stored!")


if __name__ == "__main__":
    main()
