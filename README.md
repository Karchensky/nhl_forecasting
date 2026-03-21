# NHL goal probability & +EV tooling

Proprietary models estimate the probability that a skater scores in a given game. Outputs are stored in SQLite and compared to **FanDuel** anytime-goal markets (via [The Odds API](https://the-odds-api.com/)) to surface positive expected value (`model P − implied P`).

The `reference/` folder is not part of the active pipeline; this README describes the code under the repo root only.

---

## Quick start

```text
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

1. **Config** — Edit `config/settings.yaml` as needed. Optional overrides: `config/settings_local.yaml` (gitignored if you add it).
2. **Secrets** — Put `ODDS_API_KEY=...` in a **`.env` file in the project root** (see [Environment variables](#environment-variables)).
3. **Database** — Paths default to `data/nhl_forecasting.db`. Run ingestion / training flows that populate tables (see below).
4. **Dashboard** — `streamlit run streamlit/app.py`

---

## Project layout

| Path | Role |
|------|------|
| `config/settings.yaml` | NHL API, Odds API, DB path, seasons, team list, model split seasons. |
| `database/` | SQLAlchemy models, `db_client` (engine/session, `init_db`), `ingestion` upserts. |
| `scrapers/nhl_api/` | Schedule/boxscore fetch + parsers; roster helpers; backfill utilities. |
| `scrapers/external/odds_api.py` | FanDuel player anytime-goal odds → `odds` table. |
| `models/feature_engineering.py` | Strictly causal features per (player, game): rolling windows, season-to-date, team/opp/goalie context, interactions. |
| `models/training.py` | Trains **logistic regression** (scaled), **LightGBM**, **XGBoost**; isotonic calibration for tree models; saves `models/saved/*.pkl`. |
| `models/run_training.py` | End-to-end train + validation metrics + printed LightGBM importances. |
| `models/inference.py` | Load a pickle, score a feature matrix, `store_predictions` → `model_outputs`. |
| `models/generate_predictions.py` | Score **all** rows in the full feature matrix with every model (batch refresh). |
| `models/evaluation.py` | Log loss, Brier, ROC-AUC, calibration / lift helpers for diagnostics. |
| `models/feature_importance_report.py` | Regenerates markdown table rows for top features (needs DB + saved models). |
| `scheduler/daily_job.py` | **Daily pipeline:** recent completed games → boxscores; odds fetch; predictions for **all three** models on the current test season. |
| `streamlit/app.py` | Predictions (multi-model), value bets, diagnostics. |
| `utils/config.py` | Loads YAML, merges `settings_local`, reads **`ODDS_API_KEY`** from the environment (after loading `.env`). |
| `utils/logger.py` | Shared logging. |

---

## Daily scheduler

`scheduler/daily_job.py` is intended to run once per day (Task Scheduler, cron, or CI):

1. `init_db()`
2. **`update_recent_games`** — Last N days of schedule per team; final games get full boxscore → `player_game_stats`, goalies, team stats.
3. **`fetch_odds`** — If `ODDS_API_KEY` is set, pulls events and `player_goal_scorer_anytime` for FanDuel.
4. **`run_predictions`** — Builds features for the configured **test season** and writes `model_outputs` for logistic regression, LightGBM, and XGBoost.

Run manually: `python scheduler/daily_job.py`

---

## Feature importance (by model)

Values are **each model’s share of its own signal**: \|coefficients\| (logistic) and **gain** (tree models), renormalized to **100% per model** so you can compare which features each learner emphasizes. *Not* comparable as absolute effect sizes across model types.

Top drivers from the current saved checkpoints (refresh with `python models/feature_importance_report.py` after retraining):

| Feature | Logistic regression % | LightGBM % | XGBoost % |
|---------|----------------------:|-------------:|----------:|
| `season_scoring_rate` | 5.4 | 28.1 | 22.5 |
| `position_code` | 7.4 | 13.7 | 10.7 |
| `shots_avg_20g` | 3.8 | 8.7 | 4.7 |
| `goals_avg_20g` | 2.2 | 2.5 | 6.1 |
| `toi_seconds_avg_5g` | 2.8 | 5.4 | 2.4 |
| `shots_per_60_20g` | 5.7 | 1.5 | 1.3 |
| `points_avg_20g` | 0.0 | 3.0 | 5.4 |
| `shooting_pct_20g` | 6.4 | 0.4 | 0.5 |
| `toi_seconds_avg_20g` | 3.3 | 2.6 | 1.3 |
| `blocked_shots_avg_20g` | 5.0 | 1.1 | 0.7 |
| `toi_seconds_avg_10g` | 1.2 | 3.2 | 1.9 |
| `shots_avg_5g` | 2.2 | 0.9 | 1.2 |
| `season_games_played` | 3.1 | 0.5 | 0.5 |
| `opp_sa_avg_20g` | 2.1 | 1.3 | 0.6 |
| `season_goals_per_60` | 2.1 | 0.5 | 1.3 |
