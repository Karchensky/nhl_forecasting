"""Daily job: ingest latest data, enrich with Stats API, generate predictions, fetch odds."""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text as sa_text, update

from database.db_client import get_engine, get_session, init_db
from database.ingestion import (
    upsert_game,
    upsert_goalie_game_stats,
    upsert_player,
    upsert_player_game_stats,
    upsert_team_game_stats,
)
from database.models import PlayerGameStats, TeamGameStats
from scrapers.nhl_api.client import NHLApiClient
from scrapers.nhl_api.parsers import parse_boxscore, parse_schedule
from scrapers.nhl_stats_api.client import NHLStatsApiClient
from scrapers.nhl_stats_api.parsers import SKATER_REPORTS, TEAM_REPORTS
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("scheduler.daily_job")


def _ensure_player(session, player_id, team_id, pos=None):
    """Insert player only if they don't already exist in DB."""
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
    """Update games and boxscores from the last N days, plus upcoming games."""
    cfg = load_config()
    teams = cfg["teams"]
    current_season = cfg["seasons"][-1]
    today = date.today()

    logger.info("Updating recent + upcoming games (last %d days + tomorrow)...", days_back)

    completed_game_ids = set()
    seen_schedules = set()

    for abbrev in teams:
        if abbrev in seen_schedules:
            continue
        seen_schedules.add(abbrev)

        data = client.get_schedule(abbrev, current_season)
        if not data:
            continue
        games = parse_schedule(data)
        for g in games:
            gd = g.get("game_date")
            if not gd:
                continue
            window_start = today - timedelta(days=days_back)
            window_end = today + timedelta(days=2)
            if window_start <= gd <= window_end:
                with get_session() as session:
                    upsert_game(session, g)
                if g["game_state"] in ("FINAL", "OFF"):
                    completed_game_ids.add(g["game_id"])

    logger.info("Found %d completed games to update boxscores", len(completed_game_ids))

    for gid in completed_game_ids:
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

    return completed_game_ids


def enrich_games_with_stats_api(game_ids: set[int]):
    """Enrich recently completed games with advanced stats from the Stats API."""
    if not game_ids:
        return

    logger.info("Enriching %d games with Stats API data...", len(game_ids))
    stats_client = NHLStatsApiClient()

    for gid in game_ids:
        try:
            for report_name, parser_fn in SKATER_REPORTS.items():
                rows = stats_client.fetch_skater_report_by_game(report_name, gid)
                if not rows:
                    continue
                updates = parser_fn(rows)
                if updates:
                    with get_session() as session:
                        for key, vals in updates:
                            clean = {k: v for k, v in vals.items() if v is not None}
                            if not clean:
                                continue
                            session.execute(
                                update(PlayerGameStats)
                                .where(
                                    PlayerGameStats.player_id == key["player_id"],
                                    PlayerGameStats.game_id == key["game_id"],
                                )
                                .values(**clean)
                            )

            for report_name, parser_fn in TEAM_REPORTS.items():
                rows = stats_client.fetch_team_report_by_game(report_name, gid)
                if not rows:
                    continue
                updates = parser_fn(rows)
                if updates:
                    with get_session() as session:
                        for key, vals in updates:
                            clean = {k: v for k, v in vals.items() if v is not None}
                            if not clean:
                                continue
                            session.execute(
                                update(TeamGameStats)
                                .where(
                                    TeamGameStats.team_id == key["team_id"],
                                    TeamGameStats.game_id == key["game_id"],
                                )
                                .values(**clean)
                            )

            logger.info("Enriched game %d", gid)
        except Exception as e:
            logger.error("Failed to enrich game %d: %s", gid, e)


def _get_upcoming_game_ids() -> list[int]:
    """Get game IDs for today and tomorrow that have not been played yet."""
    today = date.today()
    tomorrow = today + timedelta(days=1)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa_text("""
                SELECT game_id FROM games
                WHERE game_date >= :today AND game_date <= :tomorrow
                  AND game_state NOT IN ('FINAL', 'OFF')
            """),
            {"today": str(today), "tomorrow": str(tomorrow)},
        ).fetchall()
    return [r[0] for r in rows]


def _build_game_id_lookup() -> dict:
    """Build mapping from 'Away Team @ Home Team' -> game_id for today/tomorrow."""
    today = date.today()
    tomorrow = today + timedelta(days=1)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa_text("""
                SELECT g.game_id, at.full_name as away_name, ht.full_name as home_name
                FROM games g
                JOIN teams at ON g.away_team_id = at.team_id
                JOIN teams ht ON g.home_team_id = ht.team_id
                WHERE g.game_date >= :today AND g.game_date <= :tomorrow
            """),
            {"today": str(today), "tomorrow": str(tomorrow)},
        ).fetchall()
    lookup = {}
    for game_id, away_name, home_name in rows:
        lookup[f"{away_name} @ {home_name}"] = game_id
    return lookup


def _build_player_id_lookup() -> dict:
    """Build mapping from lowercase player name -> player_id."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa_text("SELECT player_id, full_name FROM players WHERE active = 1")
        ).fetchall()
    return {name.lower().strip(): pid for pid, name in rows}


def run_predictions():
    """Generate predictions for upcoming games using historical features."""
    from models.inference import predict_upcoming, store_predictions

    logger.info("Generating predictions for upcoming games...")
    try:
        preds = predict_upcoming()
        if not preds.empty:
            store_predictions(preds)
            logger.info("Stored %d predictions", len(preds))
        else:
            logger.info("No predictions generated.")
    except Exception as e:
        logger.error("Prediction failed: %s", e)


def fetch_odds():
    """Fetch today's player goal scorer odds from The Odds API."""
    try:
        from scrapers.external.odds_api import (
            OddsApiClient,
            fetch_and_store_odds,
        )

        client = OddsApiClient()
        if not client.api_key:
            logger.info("No Odds API key configured, skipping odds fetch.")
            return

        game_id_lookup = _build_game_id_lookup()
        player_id_lookup = _build_player_id_lookup()

        if not game_id_lookup:
            logger.info("No upcoming games found for odds lookup.")
            return

        logger.info(
            "Fetching odds for %d upcoming games...", len(game_id_lookup)
        )
        fetch_and_store_odds(
            game_id_lookup=game_id_lookup,
            player_id_lookup=player_id_lookup,
        )
    except Exception as e:
        logger.error("Odds fetch failed: %s", e)


def run_daily():
    """Full daily pipeline."""
    logger.info("Starting daily job...")
    init_db()
    client = NHLApiClient()

    completed = update_recent_games(client, days_back=3)
    enrich_games_with_stats_api(completed)
    fetch_odds()
    run_predictions()

    logger.info("Daily job complete!")


if __name__ == "__main__":
    run_daily()
