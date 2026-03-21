# NHL Player Goal Probability Model

## Objective

Build a proprietary model that predicts the probability (%) that an NHL player will score a goal in a given future game. Compare model probabilities against sportsbook odds (FanDuel preferred) to identify +EV betting opportunities.

---

## Core System Overview

The system will:

1. Ingest NHL and related data sources
2. Store structured historical + real-time data in a database
3. Engineer predictive features at player, team, and game level
4. Train and validate a probabilistic model
5. Convert sportsbook odds into implied probabilities (if available)
6. Compare model vs. market probabilities to identify value
7. Surface results via a Streamlit dashboard
8. Run on a daily automated schedule

---

## Folder Structure

```
project_root/
│
├── scrapers/
│   ├── nhl_api/
│   ├── fanduel/
│   ├── external/
│
├── database/
│   ├── schema.sql
│   ├── db_client.py
│   ├── ingestion.py
│
├── models/
│   ├── feature_engineering.py
│   ├── training.py
│   ├── inference.py
│   ├── evaluation.py
│
├── streamlit/
│   ├── app.py
│   ├── components/
│
├── scheduler/
│   ├── daily_job.py
│
├── config/
│   ├── settings.yaml
│
└── utils/
```

---

## Data Strategy

This project should not be constrained by a predefined list of data sources or features.

* The NHL API and referenced repositories are a starting point, not a limitation.
* The agent is expected to proactively discover, evaluate, and integrate any additional data sources that may improve predictive performance.
* This includes (but is not limited to): advanced stats, derived metrics, contextual data, and any publicly accessible structured data.

### Guiding Principle

If a variable could plausibly influence a player's probability of scoring, it should be considered.

### Example Feature Categories (Non-Exhaustive)

#### Player Usage & Opportunity

* Time on ice
* Power play usage/share
* Line deployment

#### Shooting & Scoring Signals

* Shots on goal
* Shot attempts
* Shooting percentage
* Expected goals (if derivable)

#### Team Context

* Team offensive strength
* Power play efficiency
* Recent team form

#### Opponent Context

* Defensive strength
* Shots/goals allowed
* Goalie performance
* Penalty kill strength

#### Game Context

* Home vs away
* Back-to-back games
* Schedule density (games in N nights)
* Travel considerations

#### Interaction Effects

* Player vs opponent history
* Teammate quality
* Line combinations

These are examples only. The agent should expand beyond this list wherever beneficial.

---

## Database DesigN

Use PostgreSQL (Supabase preferred).

This should serve only as an example. Modify/adjust based on the demands of available data and data sources.

Design schema flexibly to accommodate evolving features.

### Core Tables (Baseline)

#### players

* player_id
* name
* team

#### games

* game_id
* date
* home_team
* away_team

#### player_game_stats

* player_id
* game_id
* goals
* shots
* shot_attempts
* toi
* powerplay_toi

#### odds (optional / if available)

* player_id
* game_id
* sportsbook
* american_odds
* implied_probability

#### model_outputs

* player_id
* game_id
* predicted_probability

Schema should be extensible as new features are introduced.

---

## Feature Engineering

* Build a flexible pipeline that supports rapid iteration
* Prioritize rolling-window features (e.g., last N games)
* Normalize for opportunity (e.g., per-minute, per-TOI metrics)
* Incorporate interaction terms where meaningful
* Avoid hardcoding feature sets — allow expansion

---

## Modeling Approach

### Problem Type

Binary classification:

* Target = Did player score (1/0)

### Candidate Models

* Logistic Regression (baseline)
* Gradient Boosting (XGBoost / LightGBM preferred)
* Random Forest

### Output

* Probability of scoring (0–1)

Focus on:

* Calibration (critical)
* Stability across datasets

---

## Odds Integration (Secondary Priority)

If accessible:

* Ingest FanDuel (or equivalent) player goal scorer odds

Convert American odds to implied probability:

* Negative odds: p = |odds| / (|odds| + 100)
* Positive odds: p = 100 / (odds + 100)

Use for:

* Comparing model vs. market probabilities

This component is secondary to model quality.

---

## Model Evaluation (Primary Focus)

Evaluation should prioritize **probability quality and segmentation performance**, not betting outcomes.

### Core Metrics

* Log Loss
* Brier Score
* ROC-AUC

### Calibration

* Reliability curves
* Predicted vs. actual probability alignment

### Segmentation Analysis (Critical)

Evaluate how well the model ranks probabilities:

* Bucket predictions (e.g., deciles or quantiles)
* Compare predicted probability vs. actual scoring rate per bucket
* Produce lift curves / gain charts

Goal:

* Higher predicted probability buckets should correspond to higher realized scoring rates
* Clear monotonic relationship

### Train / Validation Split

* Strict separation of training and validation datasets
* No leakage
* Prefer time-based splits if appropriate

---

## Backtesting

Only implement if reliable odds data (current or historical) is available.

Otherwise:

* Deprioritize betting simulation
* Focus on predictive accuracy and calibration

---

## Scheduler

Run daily:

1. Pull latest data
2. Update database
3. (If available) scrape odds
4. Run feature pipeline
5. Generate predictions
6. Store outputs

Use:

* GitHub Actions OR cron

---

## Streamlit Dashboard

### Views

#### 1. Model Output

* Player
* Game
* Predicted probability

#### 2. Value View (if odds available)

* Model probability
* Implied probability
* Edge

#### 3. Model Diagnostics (Critical)

* Calibration curves
* Lift / gain charts
* Bucketed performance tables

---

## Implementation Order

1. Build ingestion pipeline
2. Design database schema
3. Build feature engineering pipeline
4. Train baseline model
5. Implement evaluation + segmentation
6. Build Streamlit UI
7. Add scheduler
8. Add odds integration (optional)
9. Iterate and optimize

---

## Optimization Strategy

* Start simple, iterate quickly
* Continuously expand feature set
* Validate improvements via segmentation metrics
* Avoid overfitting

---

## Key Principles

* Feature quality > model complexity
* Calibration is critical
* Model ranking ability (segmentation) is the priority
* System should be extensible and iterative
* Automate all repeatable processes

---

## Deliverables

* End-to-end pipeline
* Daily probability predictions
* Streamlit dashboard
* Model evaluation with strong segmentation performance

---

## Notes for Agent Execution

* Do not treat initial feature lists as complete
* Continuously search for additional predictive signals
* Write modular, reusable code
* Ensure idempotent ingestion
* Log all steps
* Handle failures gracefully
* Optimize for speed and reliability

---

## Stretch Goals

* Real-time odds tracking
* Line movement analysis
* Ensemble models
* Player clustering

---

End of instructions.
