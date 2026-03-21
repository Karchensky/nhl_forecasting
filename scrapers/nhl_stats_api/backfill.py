"""Backfill historical data from the NHL Stats API into enrichment columns.

Strategy: iterate (season, team) combos per report endpoint, parse rows,
batch-UPDATE the database.  Skips combos that already have data populated.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import text, update

from database.db_client import get_engine, get_session
from database.models import PlayerGameStats, TeamGameStats
from scrapers.nhl_stats_api.client import NHLStatsApiClient
from scrapers.nhl_stats_api.parsers import SKATER_REPORTS, TEAM_REPORTS
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("nhl_stats_api.backfill")

BATCH_SIZE = 500


def _get_team_ids() -> list[int]:
    """Return distinct team_ids that appear in player_game_stats."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT team_id FROM team_game_stats")
        ).fetchall()
    return [r[0] for r in rows]


def _season_team_has_data(
    season_id: int, team_id: int, table: str, check_col: str
) -> bool:
    """Return True if the enrichment column is already populated for this combo."""
    engine = get_engine()
    if table == "player_game_stats":
        sql = text(
            f"""
            SELECT COUNT(*) FROM player_game_stats pgs
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.season = :season AND pgs.team_id = :team
              AND pgs.{check_col} IS NOT NULL AND pgs.{check_col} != 0
            """
        )
    else:
        sql = text(
            f"""
            SELECT COUNT(*) FROM team_game_stats tgs
            JOIN games g ON tgs.game_id = g.game_id
            WHERE g.season = :season AND tgs.team_id = :team
              AND tgs.{check_col} IS NOT NULL
            """
        )
    with engine.connect() as conn:
        count = conn.execute(sql, {"season": season_id, "team": team_id}).scalar()
    return count > 0


def _apply_updates_player(updates: list[tuple[dict, dict]]):
    """Batch-apply column updates to player_game_stats."""
    if not updates:
        return
    with get_session() as session:
        for i in range(0, len(updates), BATCH_SIZE):
            batch = updates[i : i + BATCH_SIZE]
            for key, vals in batch:
                clean = {k: v for k, v in vals.items() if v is not None}
                if not clean:
                    continue
                stmt = (
                    update(PlayerGameStats)
                    .where(
                        PlayerGameStats.player_id == key["player_id"],
                        PlayerGameStats.game_id == key["game_id"],
                    )
                    .values(**clean)
                )
                session.execute(stmt)


def _apply_updates_team(updates: list[tuple[dict, dict]]):
    """Batch-apply column updates to team_game_stats."""
    if not updates:
        return
    with get_session() as session:
        for i in range(0, len(updates), BATCH_SIZE):
            batch = updates[i : i + BATCH_SIZE]
            for key, vals in batch:
                clean = {k: v for k, v in vals.items() if v is not None}
                if not clean:
                    continue
                stmt = (
                    update(TeamGameStats)
                    .where(
                        TeamGameStats.team_id == key["team_id"],
                        TeamGameStats.game_id == key["game_id"],
                    )
                    .values(**clean)
                )
                session.execute(stmt)


def backfill_skater_reports(
    seasons: list[int] | None = None,
    reports: list[str] | None = None,
    resume: bool = True,
):
    """Backfill skater-level Stats API data for all seasons and teams."""
    cfg = load_config()
    if seasons is None:
        seasons = cfg["seasons"]
    team_ids = _get_team_ids()
    if reports is None:
        reports = list(SKATER_REPORTS.keys())

    client = NHLStatsApiClient()

    check_cols = {
        "timeonice": "pp_toi_seconds",
        "summary": "gw_goals",
        "puckPossessions": "oz_start_pct",
        "goalsForAgainst": "es_goals_for",
        "powerplay": "pp_shots",
        "percentages": "pdo",
        "realtime": "total_shot_attempts",
        "faceoffpercentages": "total_faceoffs",
        "summaryshooting": "corsi_for",
    }

    total_combos = len(reports) * len(seasons) * len(team_ids)
    done = 0

    for report_name in reports:
        parser_fn = SKATER_REPORTS[report_name]
        check_col = check_cols.get(report_name)
        logger.info("=== Skater report: %s ===", report_name)

        for season in seasons:
            for team_id in team_ids:
                done += 1

                if resume and check_col:
                    if _season_team_has_data(season, team_id, "player_game_stats", check_col):
                        logger.debug(
                            "[%d/%d] Skip %s s=%d t=%d (already populated)",
                            done, total_combos, report_name, season, team_id,
                        )
                        continue

                rows = client.fetch_skater_report(report_name, season, team_id)
                if not rows:
                    logger.debug(
                        "[%d/%d] %s s=%d t=%d -> 0 rows",
                        done, total_combos, report_name, season, team_id,
                    )
                    continue

                updates = parser_fn(rows)
                _apply_updates_player(updates)

                logger.info(
                    "[%d/%d] %s s=%d t=%d -> %d rows updated",
                    done, total_combos, report_name, season, team_id, len(updates),
                )

    logger.info("Skater backfill complete.")


def backfill_team_reports(
    seasons: list[int] | None = None,
    reports: list[str] | None = None,
    resume: bool = True,
):
    """Backfill team-level Stats API data for all seasons and teams."""
    cfg = load_config()
    if seasons is None:
        seasons = cfg["seasons"]
    team_ids = _get_team_ids()
    if reports is None:
        reports = list(TEAM_REPORTS.keys())

    client = NHLStatsApiClient()

    check_cols = {
        "powerplay": "pp_opportunities_actual",
        "penaltykill": "times_shorthanded",
        "goalsforbystrength": "goals_5v5",
        "goalsagainstbystrength": "goals_against_5v5",
    }

    total_combos = len(reports) * len(seasons) * len(team_ids)
    done = 0

    for report_name in reports:
        parser_fn = TEAM_REPORTS[report_name]
        check_col = check_cols.get(report_name)
        logger.info("=== Team report: %s ===", report_name)

        for season in seasons:
            for team_id in team_ids:
                done += 1

                if resume and check_col:
                    if _season_team_has_data(season, team_id, "team_game_stats", check_col):
                        logger.debug(
                            "[%d/%d] Skip %s s=%d t=%d (already populated)",
                            done, total_combos, report_name, season, team_id,
                        )
                        continue

                rows = client.fetch_team_report(report_name, season, team_id)
                if not rows:
                    logger.debug(
                        "[%d/%d] %s s=%d t=%d -> 0 rows",
                        done, total_combos, report_name, season, team_id,
                    )
                    continue

                updates = parser_fn(rows)
                _apply_updates_team(updates)

                logger.info(
                    "[%d/%d] %s s=%d t=%d -> %d rows updated",
                    done, total_combos, report_name, season, team_id, len(updates),
                )

    logger.info("Team backfill complete.")


def backfill_all(
    seasons: list[int] | None = None,
    resume: bool = True,
):
    """Run full backfill of both skater and team reports."""
    logger.info("Starting full Stats API backfill ...")
    backfill_skater_reports(seasons=seasons, resume=resume)
    backfill_team_reports(seasons=seasons, resume=resume)
    logger.info("Full Stats API backfill complete.")


if __name__ == "__main__":
    import sys

    resume = "--no-resume" not in sys.argv
    backfill_all(resume=resume)
