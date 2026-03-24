"""Backfill play-by-play shot events for historical games.

Resume-aware: skips games that already have shot_events rows.

Usage:
    python -m scrapers.nhl_api.backfill_pbp              # all configured seasons
    python -m scrapers.nhl_api.backfill_pbp --season 20252026
    python -m scrapers.nhl_api.backfill_pbp --no-resume   # re-fetch everything
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from sqlalchemy import text as sa_text

from database.db_client import get_engine, get_session, init_db
from database.ingestion import upsert_shot_event
from scrapers.nhl_api.client import NHLApiClient
from scrapers.nhl_api.parsers import parse_play_by_play
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("nhl_api.backfill_pbp")


def _games_needing_pbp(seasons: list[int] | None, force: bool = False) -> list[int]:
    """Return game_ids of completed regular/playoff games missing shot_events."""
    engine = get_engine()
    with engine.connect() as conn:
        season_filter = ""
        params: dict = {}
        if seasons:
            placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
            season_filter = f"AND g.season IN ({placeholders})"
            params = {f"s{i}": s for i, s in enumerate(seasons)}

        if force:
            rows = conn.execute(sa_text(f"""
                SELECT g.game_id FROM games g
                WHERE g.game_state IN ('FINAL', 'OFF')
                  AND g.game_type IN (2, 3)
                  {season_filter}
                ORDER BY g.game_id
            """), params).fetchall()
        else:
            rows = conn.execute(sa_text(f"""
                SELECT g.game_id FROM games g
                WHERE g.game_state IN ('FINAL', 'OFF')
                  AND g.game_type IN (2, 3)
                  {season_filter}
                  AND g.game_id NOT IN (
                      SELECT DISTINCT game_id FROM shot_events
                  )
                ORDER BY g.game_id
            """), params).fetchall()
    return [int(r[0]) for r in rows]


def _fetch_single_pbp(args: tuple) -> tuple:
    """Fetch and parse PBP for one game (thread-pool worker).

    Retries up to 3 times with exponential backoff on 429/5xx errors.
    """
    import time as _time

    game_id, base_url = args
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(
                f"{base_url}/gamecenter/{game_id}/play-by-play", timeout=30
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    _time.sleep(2 ** attempt)  # 1s, 2s, 4s
                    continue
            resp.raise_for_status()
            data = resp.json()
            records = parse_play_by_play(data, game_id)
            return game_id, records, None
        except Exception as e:
            if attempt < max_retries:
                _time.sleep(2 ** attempt)
                continue
            return game_id, [], str(e)
    return game_id, [], "max retries exceeded"


def backfill_pbp(
    seasons: list[int] | None = None,
    force: bool = False,
    workers: int = 6,
):
    """Fetch play-by-play and store shot events."""
    init_db()
    game_ids = _games_needing_pbp(seasons, force=force)
    if not game_ids:
        logger.info("All games already have shot events — nothing to backfill.")
        return

    logger.info("Backfilling PBP shot events for %d games...", len(game_ids))
    client = NHLApiClient()
    base_url = client.base_url

    args_list = [(gid, base_url) for gid in game_ids]
    failed: list[int] = []
    total_events = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_single_pbp, a): a[0] for a in args_list
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="PBP"):
            gid = futures[future]
            try:
                gid, records, error = future.result()
                if error:
                    logger.debug("PBP fetch failed for %s: %s", gid, error)
                    failed.append(gid)
                    continue
                if records:
                    with get_session() as session:
                        for r in records:
                            upsert_shot_event(session, r)
                    total_events += len(records)
            except Exception as e:
                logger.error("PBP processing error for %s: %s", gid, e)
                failed.append(gid)

    logger.info(
        "PBP backfill complete: %d games, %d shot events stored, %d failures",
        len(game_ids) - len(failed), total_events, len(failed),
    )
    if failed:
        logger.warning("Failed game_ids (first 20): %s", failed[:20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill play-by-play shot events")
    parser.add_argument("--season", type=int, default=None, help="Single season to backfill")
    parser.add_argument("--no-resume", action="store_true", help="Re-fetch all games")
    parser.add_argument("--workers", type=int, default=6, help="Thread pool size")
    args = parser.parse_args()

    seasons = [args.season] if args.season else None
    backfill_pbp(seasons=seasons, force=args.no_resume, workers=args.workers)
