"""Generate predictions for all available data and store in DB."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.feature_engineering import build_feature_matrix
from models.inference import predict_with_model, store_predictions
from models.training import get_feature_columns
from utils.logger import get_logger

logger = get_logger("generate_predictions")


def main():
    logger.info("Building feature matrix for current season...")
    from utils.config import load_config
    cfg = load_config()

    df = build_feature_matrix()
    feature_cols = get_feature_columns(df)
    logger.info("Full dataset: %d rows, %d features", len(df), len(feature_cols))

    for model_name in ("logistic_regression", "lightgbm", "xgboost"):
        logger.info("Generating predictions with %s...", model_name)
        preds = predict_with_model(model_name, df)
        logger.info("Storing %d predictions for %s", len(preds), model_name)
        store_predictions(preds)

    logger.info("All predictions stored!")


if __name__ == "__main__":
    main()
