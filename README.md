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

**Important:** use the same venv for training, `scheduler/daily_job.py`, and Streamlit. If `daily_job` fails with `ModuleNotFoundError: optuna`, run `pip install -r requirements.txt` again inside `.venv` (Optuna is only imported when you run hyperparameter tuning, but older installs may be missing it).

1. **Config** — Edit `config/settings.yaml` as needed. Optional overrides: `config/settings_local.yaml` (gitignored if you add it).
2. **Secrets** — Put `ODDS_API_KEY=...` in a **`.env` file in the project root** (see [Environment variables](#environment-variables)).
3. **Database** — Paths default to `data/nhl_forecasting.db`. Run ingestion / migrations / training flows below.
4. **Dashboard** — `streamlit run streamlit/app.py`

### Streamlit “Connection error” / disconnects

Common causes:

- **Browser tab outlived the server** — If you stop the terminal or Streamlit crashes, refresh won’t recover. Run `streamlit run streamlit/app.py` again and open the **Local URL** (`http://localhost:8501`).
- **Use Local URL on the same PC** — The Network URL can fail from other devices if firewall/Wi‑Fi blocks the port.
- **Proxy / corporate VPN** — Can block WebSockets; try another network or browser.
- **`.env` warnings** — `python-dotenv could not parse statement starting at line 1` means line 1 of `.env` isn’t valid `KEY=value` text (BOM/UTF-16/quotes). Save `.env` as **UTF-8**, one variable per line: `ODDS_API_KEY=your_key_here` (no spaces around `=`). This warning is usually harmless if the key still loads.

---

## From scratch (new machine or empty DB)

High-level order of operations:

| Step | What | Command / notes |
|------|------|-----------------|
| 1 | Create venv, install deps | `pip install -r requirements.txt` |
| 2 | Configure seasons, paths, splits | `config/settings.yaml` (`seasons`, `model.train_seasons`, `validation_season`, `test_season`) |
| 3 | Initialize DB schema | `python -c "from database.db_client import init_db; init_db()"` |
| 4 | Add Stats API columns (existing DBs) | `python -m database.migrate_add_stats_columns` (safe to re-run; skips existing columns) |
| 5 | Ingest NHL Web API data | Run roster/schedule/boxscore backfills under `scrapers/nhl_api/` (e.g. `backfill.py`, `fetch_missing_boxscores.py`) until `games` + `player_game_stats` cover your seasons |
| 6 | Enrich with NHL Stats API | `python -m scrapers.nhl_stats_api.backfill` (long run: many HTTP requests; resume-aware). Optional: `--no-resume` to force refresh |
| 7 | Train models | `python models/run_training.py` (see [Retraining models](#retraining-models)) |
| 8 | Daily automation | `python scheduler/daily_job.py` or schedule it (boxscores, Stats API patch for completed games, odds, **today + tomorrow** predictions only) |

**Data sources**

- **Web API** (`api-web.nhle.com`): schedules, boxscores, rosters — primary game/skater facts.
- **Stats API** (`api.nhle.com/stats/rest/en`): per-game PP/SH/EV TOI, Corsi/Fenwick, zone starts, PDO, shot attempts, team PP/PK/5v5 splits — used for segmentation / opportunity features.

---

## Retraining models

After new boxscores, a Stats API backfill, or feature-code changes, refresh checkpoints:

```text
python models/run_training.py
```

- **Default:** builds the full causal feature matrix (~200+ features: rolling windows, season-to-date, home/road, vs-opponent, deployment, Corsi/PDO, enhanced opponent metrics, interactions, teammate context, PP share of team PP time), then:
  - **Optuna** tunes LightGBM hyperparameters (rolling time-based CV; can take **hours**).
  - Trains **logistic regression** (weighted), **LightGBM**, **XGBoost** with **time-decay sample weights** and **isotonic calibration** on tree models.
  - Writes `models/saved/logistic_regression.pkl`, `lightgbm.pkl`, `xgboost.pkl`.

**Faster iteration (skip tuning):**

```text
python models/run_training.py --no-tune
```

**Refresh the feature-importance table** (for docs / sanity checks):

```text
python models/feature_importance_report.py
```

Copy the printed markdown rows into the table below if you want the README to match your latest run.

**Optional: backfill predictions for all historical rows** (heavy; useful for Streamlit diagnostics on past games):

```text
python models/generate_predictions.py
```

The **daily scheduler** only scores **today and tomorrow** to limit DB churn and API-aligned “current slate” views.

---

## Train / validation / test seasons (config) vs. recency weighting

In `config/settings.yaml`, `model` defines:

| Key | Role |
|-----|------|
| `train_seasons` | Full seasons used to **fit** each model (e.g. 2020–21 … 2023–24). |
| `validation_season` | Held-out season for **early stopping** (LightGBM/XGBoost) and the **metrics printed** at the end of `run_training.py` (e.g. 2024–25). |
| `test_season` | Current slate season for **daily predictions** and `build_feature_matrix_with_upcoming` (e.g. **20252026** while the year is in progress). |

**Why not only random splits?** Game rows are **time-ordered**; random splits would **leak** future games into training and inflate scores. Season-based `train_seasons` + `validation_season` keeps a clean temporal gap for honest validation metrics.

**Recent data is still prioritized** inside training:

- **Sample weights** — `compute_sample_weights()` applies exponential decay by `game_date` (more recent player-games count more when fitting LR / LightGBM / XGBoost).
- **Optuna** (optional, when you run **without** `--no-tune`) — Tunes **only LightGBM** hyperparameters using **rolling time-based CV** on `train_seasons` ∪ `validation_season`, not random shuffles. It does **not** replace season splits; it finds better tree settings **given** those splits and weights.

**Incomplete current season (e.g. 20252026):** That’s normal. It’s the **target season for inference**; training still uses completed prior seasons + validation on the last full season. When the league year rolls over, update `train_seasons`, `validation_season`, and `test_season` in YAML and retrain.

**After adding or changing features**, retrain so `models/saved/*.pkl` `feature_cols` match the new matrix:  
`python models/run_training.py --no-tune` (or full run with Optuna).

---

## Validation performance (reference — re-run training to refresh)

Metrics below are on **`validation_season`** from a representative training run (Stats API–enriched features, `--no-tune`). Your numbers will change after retraining.

| Model | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Notes |
|-------|------------|---------|-----------|--------|
| Logistic regression | ~0.390 | ~0.119 | ~0.704 | Weighted, scaled; no isotonic. |
| LightGBM | ~0.395 | ~0.119 | ~0.708 | Raw val logloss ~0.388; **isotonic** calibration can raise logloss slightly while improving probability shape. |
| XGBoost | ~0.389 | ~0.119 | ~0.708 | Weighted + isotonic. |

For stricter segmentation / hyperparameters, run **`python models/run_training.py`** (with Optuna) overnight, then update this table from the script output.

---

## Project layout

| Path | Role |
|------|------|
| `config/settings.yaml` | NHL API, Stats API URL, Odds API, DB path, seasons, team list, model split seasons, rolling windows. |
| `database/` | SQLAlchemy models, `db_client`, `ingestion`, `migrate_add_stats_columns.py` |
| `scrapers/nhl_api/` | Schedule/boxscore fetch + parsers; roster helpers; backfill utilities. |
| `scrapers/nhl_stats_api/` | Stats API client, parsers, season/team backfill, game-level enrichment. |
| `scrapers/external/odds_api.py` | FanDuel player anytime-goal odds → `odds` table (rate-limited). |
| `models/feature_engineering.py` | Causal features per (player, game): Web + Stats API fields, rolling 5/10/20g, home/road, vs-opponent, streaks, teammate/PP-share deployment, opponent 5v5/PK, interactions. |
| `models/training.py` | Weighted training, rolling CV for Optuna, saves `models/saved/*.pkl`. |
| `models/run_training.py` | End-to-end train + validation metrics + LightGBM gain importances. |
| `models/inference.py` | Load pickle, score features; `predict_upcoming` = today + tomorrow only. |
| `models/generate_predictions.py` | Score **all** rows in full matrix (optional full-history refresh). |
| `models/evaluation.py` | Log loss, Brier, ROC-AUC, calibration / lift. |
| `models/feature_importance_report.py` | Markdown table rows for top features. |
| `scheduler/daily_job.py` | Recent + upcoming games, boxscores, Stats API enrich on completed games, odds, predictions for all three models (horizon: today + tomorrow). |
| `streamlit/app.py` | Predictions, value bets, diagnostics. |
| `utils/config.py` | YAML + `settings_local` + `.env` (`ODDS_API_KEY`). |
| `utils/logger.py` | Shared logging. |

---

## Daily scheduler

`scheduler/daily_job.py` — run once per day (Task Scheduler, cron, or CI):

1. `init_db()`
2. **`update_recent_games`** — Last N days + short lookahead; finals get boxscores.
3. **`enrich_games_with_stats_api`** — Stats API reports for those completed games.
4. **`fetch_odds`** — FanDuel anytime goal props when `ODDS_API_KEY` is set (throttled).
5. **`run_predictions`** — `build_feature_matrix_with_upcoming` for the configured **test season**, then scores **only calendar today + next day** for LR, LightGBM, XGBoost → `model_outputs`.

Run manually: `python scheduler/daily_job.py`

---

## Feature importance (by model)

Values are **each model’s share of its own signal**: \|coefficients\| (logistic) and **gain** (tree models), renormalized to **100% per model**. Refresh after retraining:

`python models/feature_importance_report.py`

Recent run (Stats-API-enriched features + segmentation):

| Feature | Logistic regression % | LightGBM % | XGBoost % |
|---------|----------------------:|-------------:|----------:|
| `season_scoring_rate` | 1.6 | 19.4 | 11.2 |
| `position_code` | 2.6 | 9.9 | 3.7 |
| `pp_individual_sat_for_avg_20g` | 0.1 | 9.2 | 5.6 |
| `shots_avg_20g` | 1.3 | 5.7 | 3.8 |
| `goals_avg_20g` | 0.5 | 0.2 | 6.8 |
| `pp_toi_share_20g` | 2.0 | 0.2 | 5.1 |
| `pp_toi_share_of_team_5g` | 0.6 | 3.1 | 3.0 |
| `shots_per_60_20g` | 2.7 | 0.7 | 0.5 |
| `toi_seconds_avg_5g` | 1.9 | 1.2 | 0.7 |
| `pp_toi_share_of_team_20g` | 0.6 | 1.3 | 1.5 |
| `pp_toi_share_5g` | 2.1 | 0.6 | 0.6 |
| `toi_seconds_avg_20g` | 1.0 | 1.1 | 1.0 |
| `opp_sa_avg_20g` | 1.1 | 1.4 | 0.4 |
| `blocked_shots_avg_20g` | 2.2 | 0.2 | 0.3 |
| `pp_toi_seconds_avg_5g` | 0.9 | 0.7 | 1.0 |
| `shooting_pct_20g` | 2.0 | 0.2 | 0.3 |
| `season_goals_per_60` | 0.9 | 0.9 | 0.6 |
| `corsi_against_avg_20g` | 1.9 | 0.2 | 0.3 |
| `vs_opp_goals_per_game` | 1.5 | 0.3 | 0.4 |
| `corsi_for_avg_20g` | 1.1 | 0.5 | 0.6 |

---

## Environment variables

- **`ODDS_API_KEY`** — Required for odds ingestion (can live in `.env` at repo root; loaded by `utils/config.py`).

---

## Segmentation / opportunity features (summary)

The feature pipeline emphasizes **repeatable opportunity** (deployment, volume, matchup), not only past goals:

- **Stats API:** PP/SH/EV TOI, shifts, Corsi/Fenwick, zone starts, PDO, shot attempts, PP shot metrics, EV on-ice goals, faceoff zones; team PP TOI / PK / 5v5 goals.
- **Rolling 3 / 5 / 10 / 20 games** (`model.rolling_windows`) on the above + classic boxscore stats — **3-game** window adds short “heat” signal.
- **Home vs road** rolling offense **and** PP TOI, PP shots, and shot attempts by venue.
- **vs current opponent** historical goal and shooting rates.
- **Streaks / trends:** goals in last 3 games, **games since last goal**, shot / PP TOI / **OZ start %** linear trends over 5 games.
- **PP share of team** (`pp_toi_share_of_team_*`): player’s share of team power-play time over prior games.
- **Teammate context:** average teammate `goals_per_60_10g` in the same lineup + interactions with shot rate and PP TOI.
- **Opponent defense:** shots/goals against, PK, **5v5 GA**, penalty frequency when available.
- **Opponent schedule:** **opponent rest days**, **opponent back-to-back**, **opponent games in last 7 days** (fatigue / travel proxy).
- **Calendar:** **day-of-week** (`game_dow`).
- **Interactions:** e.g. shot rate × opponent fatigue, home × opponent B2B, vs-opponent rate × home.

All features are built so only information **strictly before** the game is used (shifted rolling / expanding season stats).
