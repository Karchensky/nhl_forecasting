"""Print top feature importances for README refresh (requires trained models + DB)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from models.feature_engineering import build_feature_matrix
from models.training import get_feature_columns, load_model


def _norm_pct(arr: np.ndarray) -> np.ndarray:
    s = float(arr.sum())
    return (arr / s * 100) if s > 0 else arr


def feature_importance_frame(top_n: int = 15):
    df = build_feature_matrix()
    fc = get_feature_columns(df)

    lr = load_model("logistic_regression")
    lgb = load_model("lightgbm")
    xgb = load_model("xgboost")

    lr_imp = np.abs(lr["model"].coef_[0])
    lgb_imp = np.array(lgb["model"].feature_importance(importance_type="gain"))
    xgb_scores = xgb["model"].get_score(importance_type="gain")
    xgb_imp = np.array([xgb_scores.get(f, 0.0) for f in fc])

    lr_pct = _norm_pct(lr_imp)
    lgb_pct = _norm_pct(lgb_imp)
    xgb_pct = _norm_pct(xgb_imp)

    rank = lr_pct + lgb_pct + xgb_pct
    order = np.argsort(-rank)[:top_n]

    rows = []
    for i in order:
        rows.append({
            "feature": fc[i],
            "logistic_regression_pct": round(float(lr_pct[i]), 1),
            "lightgbm_pct": round(float(lgb_pct[i]), 1),
            "xgboost_pct": round(float(xgb_pct[i]), 1),
        })
    return rows


def main():
    for r in feature_importance_frame(20):
        print(
            f"| `{r['feature']}` | {r['logistic_regression_pct']} | "
            f"{r['lightgbm_pct']} | {r['xgboost_pct']} |"
        )


if __name__ == "__main__":
    main()
