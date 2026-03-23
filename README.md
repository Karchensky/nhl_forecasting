# NHL goal probability & +EV tooling

Proprietary models estimate the probability that a skater scores in a given game. Outputs are stored in SQLite and compared to **FanDuel** anytime-goal markets (via [The Odds API](https://the-odds-api.com/)) to surface positive expected value (`model P − implied P`).

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
| 8 | Daily automation | `python scheduler/daily_job.py` or schedule it (boxscores, Stats API patch for completed games, odds, **today-only** predictions) |

**Preseason vs regular/playoffs**

- **Schedule ingest** (`parse_schedule`) only stores **regular season (2)** and **playoffs (3)**; **preseason (1)** is skipped at parse time.
- **Feature matrix** (`_build_player_game_base` and related joins) also filters to **`game_type` ∈ {2, 3}**, so any preseason rows that reached the DB (e.g. old backfills) are **not** used for training or validation.

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
  - Trains **logistic regression** (weighted + scaled + **Platt calibration on validation logits** `decision_function` — not on `predict_proba`), **LightGBM**, **XGBoost** with **time-decay sample weights** and **isotonic calibration** on tree models.
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

The **daily scheduler** only scores **today’s** games to avoid stale rolling features for “tomorrow” before tonight’s results exist.

---

## Train / validation / test seasons (config) vs. recency weighting

In `config/settings.yaml`, `model` defines:

| Key | Role |
|-----|------|
| `train_seasons` | Full seasons used to **fit** each model (e.g. 2020–21 … 2023–24). |
| `validation_season` | Held-out season for **early stopping** (LightGBM/XGBoost) and the **metrics printed** at the end of `run_training.py` (e.g. 2024–25). |
| `test_season` | Current slate season for **daily predictions** and `build_feature_matrix_with_upcoming` (e.g. **20252026** while the year is in progress). |


**Recent data is prioritized** inside training:

- **Sample weights** — `compute_sample_weights()` applies exponential decay by `game_date` (more recent player-games count more when fitting LR / LightGBM / XGBoost).
- **Optuna** (optional, when you run **without** `--no-tune`) — Tunes **only LightGBM** hyperparameters using **rolling time-based CV** on `train_seasons` ∪ `validation_season`, not random shuffles. It does **not** replace season splits; it finds better tree settings **given** those splits and weights.

---

## Validation performance (reference — re-run training to refresh)

Metrics below are on **`validation_season`** (**20242025** in current config): **50,321** player-game rows. Training rows **185,437**; test season holdout **40,067** rows (not scored in this table). The full feature matrix built in that run was **275,825** rows × **309** features. Refresh after each retrain from the `run_training.py` “Model comparison” block.

**Example:** `python models/run_training.py --no-tune` (2026-03-22 evening), **309** features, LightGBM best iteration **~43** (early stopping), XGBoost **~159** rounds.

| Model | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Mean pred (val) | Notes |
|-------|------------|---------|-----------|-----------------|--------|
| Logistic regression | 0.3891 | 0.1188 | 0.7036 | 0.1497 | **Platt** fit on val **logits**; val base rate **~0.1496**. |
| LightGBM | 0.3922 | 0.1191 | 0.7071 | 0.1423 | **Isotonic** on **train** raw scores. |
| XGBoost | 0.3903 | 0.1187 | 0.7079 | 0.1381 | **Isotonic** on train raw scores. |

For stricter segmentation / hyperparameters, run **`python models/run_training.py`** (with Optuna) overnight, then update this table from the script output.

See also [Train / validation / test seasons](#train--validation--test-seasons-config-vs-recency-weighting).

---

## Project layout

| Path | Role |
|------|------|
| `config/settings.yaml` | NHL API, Stats API URL, Odds API, DB path, seasons, team list, model split seasons, rolling windows. |
| `config/odds_team_map.yaml` | NHL abbrev → display names for Odds API matching + optional label aliases (Utah, Montréal, etc.). |
| `database/` | SQLAlchemy models, `db_client`, `ingestion`, `migrate_add_stats_columns.py` |
| `scrapers/nhl_api/` | Schedule/boxscore fetch + parsers; roster helpers; backfill utilities. |
| `scrapers/nhl_stats_api/` | Stats API client, parsers, season/team backfill, game-level enrichment. |
| `scrapers/external/odds_api.py` | FanDuel player anytime-goal odds → `odds` table (rate-limited). |
| `models/feature_engineering.py` | Causal features per (player, game): Web + Stats API fields, rolling 5/10/20g, home/road, vs-opponent, streaks, teammate/PP-share deployment, opponent 5v5/PK, interactions. |
| `models/training.py` | Weighted training, rolling CV for Optuna, saves `models/saved/*.pkl`. |
| `models/run_training.py` | End-to-end train + validation metrics + LightGBM gain importances. |
| `models/inference.py` | Load pickle, score features; `predict_upcoming` = **today’s games only**. LR calibration is **Platt on logits** (`decision_function`), not on `predict_proba` (avoids OOD probability pile-up). |
| `models/generate_predictions.py` | Score **all** rows in full matrix (optional full-history refresh). |
| `models/evaluation.py` | Log loss, Brier, ROC-AUC, calibration / lift. |
| `models/feature_importance_report.py` | Full feature table (or `--top N`) → stdout, `--out` markdown, `--csv`. |
| `docs/database-erd.md` | Mermaid ERD for SQLite schema (teams, games, stats, odds, model_outputs). |
| `scheduler/daily_job.py` | Recent + upcoming games, boxscores, Stats API enrich on completed games, odds, predictions for all three models (**today only**). |
| `streamlit/app.py` | **Opportunities** tab (date / game / team / model filters; model %, market %, edges, lines, ranks) and **Diagnostics**. |
| `utils/config.py` | YAML + `settings_local` + `.env` (`ODDS_API_KEY`). |
| `utils/logger.py` | Shared logging. |

---

## Daily scheduler

`scheduler/daily_job.py` — run once per day (Task Scheduler, cron, or CI):

1. `init_db()`
2. **`update_recent_games`** — Last N days + short lookahead; finals get boxscores.
3. **`enrich_games_with_stats_api`** — Stats API reports for those completed games.
4. **`fetch_odds`** — FanDuel anytime goal props when `ODDS_API_KEY` is set (throttled). Only **today’s** `games.game_date` rows are matched; extra Odds API slates are ignored. Team strings use `config/odds_team_map.yaml` because `teams.full_name` is often a 3-letter abbrev from roster ingest. **No odds upsert** when `games.game_state` is **LIVE**, **CRIT**, **OFF**, or **FINAL** (keeps pregame lines; avoids overwriting with live prices).
5. **`run_predictions`** — `build_feature_matrix_with_upcoming` for the configured **test season**, then scores **only calendar today** for LR, LightGBM, XGBoost → `model_outputs`.

Run manually: `python scheduler/daily_job.py`

---

## Database schema (ERD)

See **[docs/database-erd.md](docs/database-erd.md)** for a Mermaid entity-relationship diagram

---

## Feature importance (by model)

Values are **each model’s share of its own signal**, renormalized to **100% per model**: logistic uses **\|coef\| / scaler scale** (approx. sensitivity per one unit in the *original* feature, **not** causal importance), while trees use **gain**. **Do not add LR % to tree %** — the report sorts by **Trees sum %** = LightGBM % + XGBoost % (two gain shares; max ~200). Compare LR % only to other LR %. After changing calibration, **retrain** (`run_training.py`) so Streamlit picks up the new `logistic_regression.pkl`.

```text
# All features, markdown file + CSV
python models/feature_importance_report.py --out docs/feature_importance.md --csv docs/feature_importance.csv

# Top 30 only to stdout
python models/feature_importance_report.py --top 30
```

Snapshot (top features only — regenerate files above after retraining):

Recent run (2026-03-22, `--no-tune` + `feature_importance_report.py`, Stats-API-enriched **309** features; same train/val as table above):

| Feature | LR % | LightGBM % | XGBoost % | Trees sum % |
|---------|-----:|-----------:|----------:|------------:|
| `season_scoring_rate` | 2.12 | 23.27 | 10.23 | 33.5 |
| `position_code` | 0.38 | 10.52 | 3.28 | 13.79 |
| `pp_individual_sat_for_avg_20g` | 0.16 | 7.31 | 5.53 | 12.85 |
| `shots_avg_20g` | 0.36 | 4.86 | 3.54 | 8.4 |
| `pp_toi_share_of_team_5g` | 0.01 | 2.14 | 2.99 | 5.13 |
| `pp_toi_share_20g` | 5.84 | 2.95 | 0.37 | 3.31 |
| `pp_toi_share_of_team_20g` | 0.01 | 1.72 | 1.17 | 2.88 |
| `points_avg_20g` | 0.22 | 0.44 | 2.06 | 2.5 |
| `pp_toi_share_of_team_3g` | 0.10 | 0.58 | 1.75 | 2.33 |
| `pp_toi_seconds_avg_5g` | 0.0 | 0.71 | 1.56 | 2.27 |
| `shots_per_60_20g` | 0.21 | 1.53 | 0.73 | 2.26 |
| `toi_seconds_avg_3g` | 0.0 | 1.26 | 0.86 | 2.12 |
| `toi_seconds_avg_20g` | 0.0 | 1.21 | 0.75 | 1.96 |
| `season_goals_per_60` | 0.23 | 0.66 | 0.97 | 1.63 |
| `opp_sa_avg_20g` | 0.07 | 1.16 | 0.27 | 1.43 |
| `toi_seconds_avg_5g` | 0.0 | 0.68 | 0.62 | 1.3 |
| `pp_toi_share_of_team_10g` | 0.02 | 0.40 | 0.87 | 1.27 |
| `pp_individual_sat_for_avg_10g` | 0.45 | 0.55 | 0.63 | 1.18 |
| `pp_toi_share_5g` | 3.08 | 0.74 | 0.43 | 1.17 |
| `shots_avg_10g` | 0.23 | 0.19 | 0.96 | 1.15 |
| `toi_seconds_avg_10g` | 0.0 | 0.58 | 0.55 | 1.13 |

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