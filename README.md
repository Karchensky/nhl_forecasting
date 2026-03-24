# NHL Goal Probability Model

This project estimates the probability that any NHL skater scores at least one goal in a given game. It trains three models (Logistic Regression, LightGBM, XGBoost) on five seasons of game-level data, compares predictions against FanDuel's anytime goal scorer odds (pulled from [The Odds API](https://the-odds-api.com/)), and surfaces potential +EV (positive expected value) betting opportunities in a Streamlit dashboard.

The goal is straightforward: if our model thinks a player has a 25% chance to score but FanDuel is offering +450 (implied 18%), that's an edge worth investigating.

---

## How it works

1. **Data collection** -- Two NHL APIs provide game schedules, boxscores, play-by-play shot data, and advanced stats (Corsi, Fenwick, zone starts, PP deployment, etc.) going back to the 2020-21 season. The Odds API provides FanDuel player prop lines for today's games.

2. **Feature engineering** -- For each (player, game) row, the pipeline computes ~309 features: rolling averages across 3/5/10/20-game windows, season-to-date stats, opponent defensive quality, goalie performance, schedule context (rest days, back-to-backs), home/road splits, streak indicators, and interaction terms. All features are strictly causal (no future data leakage).

3. **Model training** -- Three models are trained on seasons 2020-21 through 2023-24, validated on 2024-25, and tested on 2025-26 (the current season). Time-decay weighting gives recent games more influence. LightGBM can be Optuna-tuned with rolling temporal cross-validation. Calibration is applied post-training (Platt scaling for LR, isotonic regression for tree models).

4. **Daily predictions** -- A scheduler script fetches today's games, enriches them with advanced stats, pulls FanDuel odds, generates predictions for all three models, and stores everything in SQLite.

5. **Dashboard** -- A Streamlit app shows today's slate with model probabilities, FanDuel implied probabilities, and the edge (model % - market %) for each player. It also has diagnostics and backtest tabs.

---

## Setup

### Prerequisites

- Python 3.11+
- An API key from [The Odds API](https://the-odds-api.com/) (free tier gives 500 requests/month, which is plenty)

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Configuration

1. **Edit `config/settings.yaml`** to set your seasons, database path, and model split seasons.

2. **Create a `.env` file** in the project root with your Odds API key:

   ```text
   ODDS_API_KEY=your_key_here
   ```

3. **Initialize the database:**
   ```bash
   python -c "from database.db_client import init_db; init_db()"
   python -m database.migrate_add_stats_columns
   ```

### Backfilling historical data (first time only)

This populates your database with several seasons of game data. It takes a while due to API rate limits.

```bash
# 1. Fetch rosters, schedules, and boxscores from the NHL Web API
python -m scrapers.nhl_api.backfill

# 2. Fill in any missing boxscores
python -m scrapers.nhl_api.fetch_missing_boxscores

# 3. Enrich with advanced stats from the NHL Stats API (Corsi, Fenwick, etc.)
#    This is the longest step -- thousands of API calls. It's resume-aware,
#    so you can stop and restart safely.
python -m scrapers.nhl_stats_api.backfill

# 4. Backfill play-by-play shot events (for the xG model).
#    Resume-aware -- safe to stop and restart. Uses 2 workers to stay under API rate limits.
python -m scrapers.nhl_api.backfill_pbp --workers 2
```

### Training models

```bash
# 1. Train the xG model first (it feeds into the main models as features)
python -m models.xg_model

# 2. Full training with Optuna hyperparameter tuning (can take 2-4 hours)
python models/run_training.py

# Quick training without tuning (5-10 minutes)
python models/run_training.py --no-tune
```

This saves model files to `models/saved/`: `xg_model.pkl`, `logistic_regression.pkl`, `lightgbm.pkl`, `xgboost.pkl`.

---

## Daily workflow

Once you have historical data and trained models, your daily routine looks like this:

```bash
# 1. Run the daily job (fetches new games, enriches stats, pulls odds, generates predictions)
python scheduler/daily_job.py

# 2. Open the dashboard
streamlit run streamlit/app.py
```

The daily job does the following in order:

1. Fetches games from the last few days + upcoming schedule
2. Pulls boxscores for any newly completed games
3. Enriches completed games with Stats API advanced stats
4. Ingests play-by-play shot events for newly completed games (feeds the xG model)
5. Fetches FanDuel anytime goal scorer odds for today's games
6. Generates predictions for today's games using all three models

Run it before puck drop. The Streamlit dashboard will then show today's slate with edges.

### Other useful commands

```bash
# Refresh rosters (useful at trade deadline or start of season)
python -m scrapers.nhl_api.refresh_rosters

# Backfill predictions for all historical games (for the Diagnostics tab)
python models/generate_predictions.py --all

# Generate predictions for a specific season only
python models/generate_predictions.py --season 20252026

# Feature importance report
python models/feature_importance_report.py --top 30
```

---

## Project structure

```text
nhl_forecasting/
|-- config/
|   |-- settings.yaml            # Seasons, API URLs, model split config, rolling windows
|   |-- odds_team_map.yaml       # Maps NHL abbreviations to Odds API display names
|
|-- data/
|   |-- nhl_forecasting.db       # SQLite database (all game data, odds, predictions)
|
|-- database/
|   |-- models.py                # SQLAlchemy ORM (Teams, Players, Games, Stats, Odds, ModelOutput)
|   |-- db_client.py             # Engine/session factory (SQLite with WAL mode)
|   |-- ingestion.py             # Upsert helpers for idempotent data loading
|   |-- migrate_add_stats_columns.py  # Adds Stats API enrichment columns to existing DBs
|
|-- scrapers/
|   |-- nhl_api/                 # NHL Web API (api-web.nhle.com)
|   |   |-- client.py            # Schedule, boxscore, roster, play-by-play endpoints
|   |   |-- parsers.py           # Parse boxscore JSON into player/goalie/team stats
|   |   |-- backfill.py          # Bulk historical data ingestion
|   |   |-- backfill_pbp.py     # Resume-aware play-by-play shot event backfill
|   |   |-- fetch_missing_boxscores.py
|   |   |-- refresh_rosters.py
|   |
|   |-- nhl_stats_api/           # NHL Stats API (api.nhle.com/stats/rest/en)
|   |   |-- client.py            # Advanced stats reports (Corsi, Fenwick, zone starts, etc.)
|   |   |-- parsers.py           # 9 skater reports + 4 team reports
|   |   |-- backfill.py          # Resume-aware bulk enrichment
|   |
|   |-- external/
|       |-- odds_api.py          # The Odds API client, odds normalization, FanDuel fetch
|       |-- odds_matching.py     # Fuzzy name matching between Odds API and NHL DB
|
|-- models/
|   |-- feature_engineering.py   # ~350 features: rolling stats, xG, opponent quality, context, etc.
|   |-- xg_model.py             # Expected goals model (shot-level LightGBM classifier)
|   |-- training.py              # Train LR/LGB/XGB with calibration and time-decay weights
|   |-- inference.py             # Load models, score today's games, store predictions
|   |-- evaluation.py            # Logloss, Brier, AUC, calibration tables, lift tables
|   |-- generate_predictions.py  # Batch scoring (all seasons or specific season)
|   |-- run_training.py          # End-to-end: build features, tune, train, evaluate
|   |-- saved/                   # Pickled model files
|
|-- scheduler/
|   |-- daily_job.py             # Orchestrates the full daily pipeline
|
|-- streamlit/
|   |-- app.py                   # Dashboard: Opportunities, Diagnostics, Backtest tabs
|
|-- utils/
|   |-- config.py                # YAML config loader with .env support
|   |-- logger.py                # Shared logging setup
```

---

## The three models

| Model | Role | Calibration | Notes |
|-------|------|-------------|-------|
| **Logistic Regression** | Baseline | Platt scaling (CV out-of-fold on training logits) | L1 regularization auto-selects ~266 of 309 features. Predictions clipped to 2-45% as a safety net for OOD inputs. Conservative but stable. |
| **LightGBM** | Primary | Isotonic regression on validation set | Optuna-tunable. Default params produce predictions tightly concentrated near the 15% base rate; tuning widens the spread. |
| **XGBoost** | Primary | Isotonic regression on validation set | Best raw performance (lowest logloss, highest AUC). Predictions track FanDuel lines closely on out-of-sample data. |

### Test season performance (2025-26, 40K out-of-sample predictions)

| Model | Logloss | AUC | Mean Pred | Base Rate |
|-------|---------|-----|-----------|-----------|
| LR | 0.398 | 0.689 | 0.134 | 0.151 |
| LGB (default) | 0.396 | 0.694 | 0.145 | 0.151 |
| XGB | 0.395 | 0.695 | 0.143 | 0.151 |

These are honest out-of-sample numbers -- the test season was never seen during training or validation.

---

## Expected Goals (xG) model

The xG model is a shot-level LightGBM binary classifier that predicts the probability of each shot becoming a goal. It uses play-by-play data from the NHL API and feeds into the main goal-scoring models as features.

### xG shot features

- **Spatial** -- distance to net, angle, x/y coordinates, distance squared, angle squared, distance x angle interaction
- **Shot type** -- one-hot encoding of wrist, snap, slap, backhand, tip-in, deflected, wrap-around, poke, bat, between-legs, cradle
- **Game state** -- period, time in period, overtime flag, score differential (trailing/tied/leading), strength state (PP/SH/5v5), empty net
- **Shot sequence** -- time since last shot (any team), rebound (shot within 3s of prior), rush (3-5s), same-team prior shot (sustained pressure), prior shot was a goal, distance/angle change from prior shot, shots in last 10 seconds (flurry)

The xG model is trained once on all available shot data, calibrated via isotonic regression on a validation season, and saved to `models/saved/xg_model.pkl`.

### xG features in the main models

Per-player-per-game xG totals are aggregated and turned into rolling features at all four windows (3/5/10/20 games):

- **xG total** and **xG per 60** -- how many expected goals the player generates from shot quality + volume
- **Goals above expected** -- actual goals minus xG (positive = elite finisher, negative = unlucky or poor finisher)
- **Finishing percentage** -- actual goals / xG (sustainability indicator)
- **xG trend** -- short-window xG vs long-window xG (is xG rising or falling?)
- **Finishing consistency** -- std dev of goals-above-expected (low = reliable, high = streaky)
- **Teammate xG quality** -- mean xG of teammates in the same game (offensive environment quality)
- **Opponent xG against** -- how much xG the opponent allows per game (defensive weakness)
- **xG x opponent interaction** -- player xG rate x opponent xG-against (matchup quality)

---

## Feature overview

The ~350 features fall into these groups:

- **Rolling player stats** (4 windows: 3/5/10/20 games) -- goals, assists, shots, TOI, PP time, Corsi/Fenwick, zone starts, shooting %, etc.
- **xG-derived** (4 windows) -- xG totals, xG/60, goals above expected, finishing %, xG trend, teammate/opponent xG context
- **Season cumulative** -- goals, shots, games played, shooting %, goals per 60
- **Team rolling** -- team goals, shots, PP %, win rate
- **PP deployment** -- player's share of team power play time
- **Teammate context** -- average linemate scoring rate and xG quality
- **Opponent defense** (4 windows) -- goals against, shots against, PK %, 5v5 GA, xG against
- **Goalie quality** (4 windows) -- opposing goalie save % and GA average
- **Schedule context** -- rest days, back-to-back, games in last 7 days (player and opponent)
- **Player profile** -- position, age
- **Home/road splits** (4 windows) -- venue-specific goal and shot rates
- **vs. opponent history** -- career scoring rate against this team
- **Streaks/momentum** -- recent scoring, shot trends, games since last goal
- **Interaction terms** -- shot rate x opponent weakness, PP time x opponent penalties, xG x opponent xG-against, etc.

All features use a `shift(1)` offset so they only reflect information available before the game starts.

---

## Dashboard tabs

### Opportunities

The main view. Shows today's slate with each player's predicted probability from all three models, the FanDuel line (American odds and implied probability), and the edge for each model. Filter by date, team, or position. The summary bar shows total players on the slate, number of +EV plays, and the best edge available.

### Diagnostics

Model evaluation across seasons. Shows logloss, Brier score, AUC, calibration curves, and lift tables. Includes a train/val/test season selector so you can verify the model isn't just memorizing training data.

### Backtest

Simulates flat-stake betting on historical +EV plays. Adjust the edge threshold to see ROI, cumulative P&L, and performance by edge bucket.

---

## How odds work in this project

The Odds API returns FanDuel's anytime goal scorer prices. Most come as American odds (+310 means bet $100 to win $310), but some arrive as decimal odds (13.0 means a $1 bet returns $13). The `normalize_book_price()` function detects the format automatically:

- Negative numbers: American favorite (e.g., -200)
- 100 or higher: American underdog (e.g., +1100)
- 1.01 to 50: Decimal odds, converted to American
- 51 to 99: American (rare but valid)

The implied probability is `1 / decimal_odds` or the standard American-to-implied formula. FanDuel's implied probabilities include a vig (overround), typically 5-10%, so their lines slightly overstate the true probability of each outcome.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ODDS_API_KEY` | For odds | API key from The Odds API. Put in `.env` at project root. |

---

## Retraining the xG model

The xG model trains on shot-level play-by-play data and should be retrained when you have significantly more data (e.g., a new season). Retraining is quick (under a minute on ~800K shots).

```bash
# 1. Make sure PBP data is up to date (resume-aware, safe to re-run)
python -m scrapers.nhl_api.backfill_pbp --workers 2

# 2. Retrain the xG model
python -m models.xg_model

# 3. Retrain the main models (they'll pick up the updated xG features automatically)
python models/run_training.py --no-tune    # quick
python models/run_training.py              # with Optuna tuning (slower, better)
```

The xG model will print AUC, log loss, and Brier score for train/val/test splits, plus the top 15 most important features.

### xG model performance (900K shots, 7,669 games)

| Split | AUC | LogLoss | Brier | Mean Pred | Actual Rate |
|-------|-----|---------|-------|-----------|-------------|
| Train (600K shots, 2020-24) | 0.843 | 0.164 | 0.045 | 5.22% | 5.39% |
| Val (168K shots, 2024-25) | 0.842 | 0.158 | 0.043 | 5.13% | 5.13% |
| Test (131K shots, 2025-26) | 0.834 | 0.166 | 0.045 | 5.41% | 5.38% |

These are strong numbers for xG -- comparable to public models like MoneyPuck. The near-zero train-val gap (0.843 vs 0.842 AUC) shows the model generalizes well. Calibration is tight: predicted and actual goal rates differ by less than 0.2 percentage points across all splits.

Top features: shot distance, shot type, offensive zone indicator, empty net, distance-angle interaction, rebounds, and shot flurries.

### New season checklist

At the start of a new NHL season:

1. **Update `config/settings.yaml`** -- add the new season to `seasons`, shift `validation_season` and `test_season` forward by one year.

2. **Backfill the new season** once games start being played:

   ```bash
   python -m scrapers.nhl_api.backfill
   python -m scrapers.nhl_api.fetch_missing_boxscores
   python -m scrapers.nhl_stats_api.backfill
   python -m scrapers.nhl_api.backfill_pbp --workers 2
   ```

3. **Retrain all models** (the old season's validation set becomes part of training):

   ```bash
   python -m models.xg_model
   python models/run_training.py
   ```

4. **Refresh rosters** for new acquisitions:

   ```bash
   python -m scrapers.nhl_api.refresh_rosters
   ```

5. Resume daily operations with `python scheduler/daily_job.py`.

---

## Notes

- The database is SQLite with WAL journaling. It lives at `data/nhl_forecasting.db` by default.
- Only regular season (game_type=2) and playoff (game_type=3) games are used. Preseason is filtered out.
- The daily job only generates predictions for today's date to avoid stale rolling features for future games.
- Model pickles include the feature column list, scaler (LR only), and calibrator. If you change the feature engineering code, you must retrain.
- The Stats API enrichment and PBP backfill are both resume-aware. If interrupted, just run them again and they'll pick up where they left off.
- The xG model is a dependency of the main models. If you retrain the xG model, you should also retrain the main models so the xG features are consistent.
