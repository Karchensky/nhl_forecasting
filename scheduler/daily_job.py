"""Daily job: ingest latest data, generate predictions, fetch odds."""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_client import get_session, init_db
from database.ingestion import (
    upsert_game,
    upsert_goalie_game_stats,
    upsert_player,
    upsert_player_game_stats,
    upsert_team_game_stats,
)
from scrapers.nhl_api.client import NHLApiClient
from scrapers.nhl_api.parsers import parse_boxscore, parse_schedule
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("scheduler.daily_job")


def _ensure_player(session, player_id, team_id, pos=None):
    """Insert player only if they don't already exist in DB."""
    from sqlalchemy import text as sa_text
    exists = session.execute(
        sa_text("SELECT 1 FROM players WHERE player_id = :pid"),
        {"pid": player_id}
    ).fetchone()
    if not exists:
        upsert_player(session, {
            "player_id": player_id,
            "full_name": f"Player_{player_id}",
            "position": pos,
            "current_team_id": team_id,
            "active": True,
        })


def update_recent_games(client: NHLApiClient, days_back: int = 3):
    """Update games and boxscores from the last N days."""
    cfg = load_config()
    teams = cfg["teams"]
    current_season = cfg["seasons"][-1]

    logger.info("Updating recent games (last %d days)...", days_back)

    game_ids_to_update = set()
    for abbrev in teams:
        data = client.get_schedule(abbrev, current_season)
        if not data:
            continue
        games = parse_schedule(data)
        today = date.today()
        for g in games:
            gd = g.get("game_date")
            if gd and (today - timedelta(days=days_back)) <= gd <= today:
                with get_session() as session:
                    upsert_game(session, g)
                if g["game_state"] in ("FINAL", "OFF"):
                    game_ids_to_update.add(g["game_id"])

    logger.info("Found %d completed games to update", len(game_ids_to_update))

    for gid in game_ids_to_update:
        try:
            data = client.get_boxscore(gid)
            if not data:
                continue
            player_stats, goalie_stats, team_stats = parse_boxscore(data)
            with get_session() as session:
                for ps in player_stats:
                    _ensure_player(session, ps["player_id"], ps["team_id"])
                    upsert_player_game_stats(session, ps)
                for gs in goalie_stats:
                    _ensure_player(session, gs["player_id"], gs["team_id"], pos="G")
                    upsert_goalie_game_stats(session, gs)
                for ts in team_stats:
                    upsert_team_game_stats(session, ts)
        except Exception as e:
            logger.error("Failed to update game %s: %s", gid, e)


def run_predictions():
    """Generate predictions for the current season."""
    from models.inference import predict_upcoming, store_predictions

    logger.info("Generating predictions...")
    predictions = predict_upcoming(model_name="lightgbm")
    if not predictions.empty:
        store_predictions(predictions)
        logger.info("Stored %d predictions", len(predictions))
    else:
        logger.warning("No predictions generated.")


def fetch_odds():
    """Attempt to fetch current odds."""
    try:
        from scrapers.external.odds_api import OddsApiClient
        client = OddsApiClient()
        if not client.api_key:
            logger.info("No Odds API key configured, skipping odds fetch.")
            return
        from scrapers.external.odds_api import fetch_and_store_odds
        fetch_and_store_odds()
    except Exception as e:
        logger.error("Odds fetch failed: %s", e)


def run_daily():
    """Full daily pipeline."""
    logger.info("Starting daily job...")
    init_db()
    client = NHLApiClient()

    update_recent_games(client, days_back=3)
    fetch_odds()
    run_predictions()

    logger.info("Daily job complete!")


if __name__ == "__main__":
    run_daily()
