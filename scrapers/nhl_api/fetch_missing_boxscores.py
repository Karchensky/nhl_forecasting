"""Fetch missing boxscores: concurrent API calls, sequential DB writes."""
import sys
import time
import queue
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
from sqlalchemy import text

from database.db_client import get_engine, get_session, init_db
from database.ingestion import (
    upsert_goalie_game_stats,
    upsert_player,
    upsert_player_game_stats,
    upsert_team_game_stats,
)
from scrapers.nhl_api.parsers import parse_boxscore
from utils.logger import get_logger

logger = get_logger("fetch_missing")
BASE_URL = "https://api-web.nhle.com/v1"


def get_missing_game_ids() -> list[int]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT game_id FROM games "
            "WHERE game_state IN ('FINAL','OFF') "
            "AND game_id NOT IN (SELECT DISTINCT game_id FROM team_game_stats) "
            "ORDER BY game_id"
        )).fetchall()
    return [r[0] for r in rows]


def fetch_one(gid: int):
    """Fetch a single boxscore from NHL API. Returns parsed data or error."""
    try:
        resp = requests.get(f"{BASE_URL}/gamecenter/{gid}/boxscore", timeout=30)
        if resp.status_code == 429:
            time.sleep(5)
            resp = requests.get(f"{BASE_URL}/gamecenter/{gid}/boxscore", timeout=30)
        if resp.status_code == 404:
            return gid, None, "404"
        resp.raise_for_status()
        data = resp.json()
        parsed = parse_boxscore(data)
        return gid, parsed, None
    except Exception as e:
        return gid, None, str(e)


def _ensure_player(session, player_id, team_id, pos=None):
    """Insert player only if they don't already exist."""
    from sqlalchemy import text
    exists = session.execute(
        text("SELECT 1 FROM players WHERE player_id = :pid"),
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


def store_batch(results: list):
    """Write a batch of parsed boxscores to DB sequentially."""
    with get_session() as session:
        for gid, (player_stats, goalie_stats, team_stats) in results:
            for ps in player_stats:
                _ensure_player(session, ps["player_id"], ps["team_id"])
                upsert_player_game_stats(session, ps)
            for gs in goalie_stats:
                _ensure_player(session, gs["player_id"], gs["team_id"], pos="G")
                upsert_goalie_game_stats(session, gs)
            for ts in team_stats:
                upsert_team_game_stats(session, ts)


def main():
    init_db()
    missing = get_missing_game_ids()
    logger.info("Missing boxscores: %d", len(missing))
    if not missing:
        logger.info("All boxscores present!")
        return

    failed = []
    batch = []
    BATCH_SIZE = 50

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_one, gid): gid for gid in missing}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Boxscores"):
            gid = futures[future]
            try:
                gid, parsed, error = future.result()
                if error or parsed is None:
                    failed.append(gid)
                    continue
                batch.append((gid, parsed))
                if len(batch) >= BATCH_SIZE:
                    store_batch(batch)
                    batch = []
            except Exception as e:
                failed.append(gid)

    if batch:
        store_batch(batch)

    logger.info("Done. Succeeded: %d, Failed: %d", len(missing) - len(failed), len(failed))
    if failed:
        logger.info("Failed count by type - rerun to retry")


if __name__ == "__main__":
    main()
