"""Feature importances for README / analysis (requires trained models + DB)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.feature_engineering import build_feature_matrix
from models.training import get_feature_columns, load_model


def _norm_pct(arr: np.ndarray) -> np.ndarray:
    s = float(arr.sum())
    return (arr / s * 100) if s > 0 else arr


def feature_importance_rows(
    top_n: int | None = None,
) -> list[dict]:
    """All features (or first top_n) sorted by combined importance signal."""
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

    combined = lr_pct + lgb_pct + xgb_pct
    order = np.argsort(-combined)
    if top_n is not None:
        order = order[:top_n]

    rows = []
    for i in order:
        rows.append({
            "feature": fc[i],
            "logistic_regression_pct": round(float(lr_pct[i]), 2),
            "lightgbm_pct": round(float(lgb_pct[i]), 2),
            "xgboost_pct": round(float(xgb_pct[i]), 2),
            "combined_pct": round(float(combined[i]), 2),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Print feature importance markdown table rows.")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Only print the top N features (default: all features)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown table to this file instead of stdout",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write a CSV to this path",
    )
    args = parser.parse_args()

    rows = feature_importance_rows(top_n=args.top)
    lines = [
        "| Feature | LR % | LightGBM % | XGBoost % | Combined % |",
        "|---------|-----:|-----------:|----------:|-----------:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['feature']}` | {r['logistic_regression_pct']} | "
            f"{r['lightgbm_pct']} | {r['xgboost_pct']} | {r['combined_pct']} |"
        )
    text = "\n".join(lines) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {args.out}")
    else:
        print(text)

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
