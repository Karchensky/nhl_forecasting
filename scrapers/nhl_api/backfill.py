"""Orchestrates full historical data backfill from the NHL API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tqdm import tqdm

from database.db_client import get_session, init_db
from database.ingestion import (
    upsert_game,
    upsert_goalie_game_stats,
    upsert_player,
    upsert_player_game_stats,
    upsert_team,
    upsert_team_game_stats,
)
from scrapers.nhl_api.client import NHLApiClient
from scrapers.nhl_api.parsers import parse_boxscore, parse_roster, parse_schedule
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("nhl_api.backfill")

TEAM_ID_MAP = {
    "ANA": 24, "BOS": 6, "BUF": 7, "CGY": 20, "CAR": 12,
    "CHI": 16, "COL": 21, "CBJ": 29, "DAL": 25, "DET": 17,
    "EDM": 22, "FLA": 13, "LAK": 26, "MIN": 30, "MTL": 8,
    "NSH": 18, "NJD": 1, "NYI": 2, "NYR": 3, "OTT": 9,
    "PHI": 4, "PIT": 5, "SJS": 28, "SEA": 55, "STL": 19,
    "TBL": 14, "TOR": 10, "VAN": 23, "VGK": 54, "WPG": 52,
    "WSH": 15, "ARI": 53, "UTA": 59,
}

SEASON_TEAM_ID_OVERRIDES = {
    (20252026, "UTA"): 68,
}

SEASON_TEAM_EXCLUDE = {
    20202021: {"UTA", "SEA"},
    20212022: {"UTA"},
    20222023: {"UTA"},
    20232024: {"UTA"},
    20242025: {"ARI"},
    20252026: {"ARI"},
}


def _teams_for_season(all_teams: list[str], season: int) -> list[str]:
    excluded = SEASON_TEAM_EXCLUDE.get(season, set())
    return [t for t in all_teams if t not in excluded]


def _team_id_for_season(abbrev: str, season: int) -> int | None:
    override = SEASON_TEAM_ID_OVERRIDES.get((season, abbrev))
    if override:
        return override
    return TEAM_ID_MAP.get(abbrev)


def backfill_rosters(client: NHLApiClient, seasons: list[int], teams: list[str]):
    logger.info("Backfilling rosters...")
    for season in seasons:
        active_teams = _teams_for_season(teams, season)
        for abbrev in tqdm(active_teams, desc=f"Rosters {season}"):
            team_id = _team_id_for_season(abbrev, season)
            if not team_id:
                continue
            data = client.get_roster(abbrev, season)
            if not data:
                logger.warning("No roster for %s/%s", abbrev, season)
                continue
            team_data, players = parse_roster(data, abbrev, team_id)
            with get_session() as session:
                upsert_team(session, team_data)
                for p in players:
                    upsert_player(session, p)


def backfill_schedules(client: NHLApiClient, seasons: list[int], teams: list[str]) -> set[int]:
    logger.info("Backfilling schedules...")
    all_game_ids = set()
    for season in seasons:
        active_teams = _teams_for_season(teams, season)
        season_game_ids = set()
        for abbrev in tqdm(active_teams, desc=f"Schedule {season}"):
            data = client.get_schedule(abbrev, season)
            if not data:
                continue
            games = parse_schedule(data)
            with get_session() as session:
                for g in games:
                    _update_team_names_from_game(session, g, data)
                    upsert_game(session, g)
                    if g["game_state"] in ("FINAL", "OFF"):
                        season_game_ids.add(g["game_id"])
        logger.info("Season %s: %d completed games", season, len(season_game_ids))
        all_game_ids.update(season_game_ids)
    return all_game_ids


def _update_team_names_from_game(session, game_dict: dict, schedule_data: dict):
    """Extract full team names from schedule game entries and upsert teams."""
    for raw_game in schedule_data.get("games", []):
        if raw_game["id"] != game_dict["game_id"]:
            continue
        for side in ("homeTeam", "awayTeam"):
            t = raw_game.get(side, {})
            tid = t.get("id")
            abbrev = t.get("abbrev", "")
            place = t.get("placeName", {})
            common = t.get("commonName", {})
            place_str = place.get("default", "") if isinstance(place, dict) else ""
            common_str = common.get("default", "") if isinstance(common, dict) else ""
            full = f"{place_str} {common_str}".strip() if place_str else common_str
            if tid and abbrev:
                upsert_team(session, {
                    "team_id": tid,
                    "abbreviation": abbrev,
                    "full_name": full or abbrev,
                })
        break


def _ensure_player_exists(session, player_id, team_id, pos=None):
    """Insert player with placeholder name only if they don't exist yet."""
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


def _fetch_single_boxscore(args):
    """Fetch and parse a single boxscore (used by thread pool)."""
    gid, base_url = args
    import requests
    import time
    try:
        resp = requests.get(f"{base_url}/gamecenter/{gid}/boxscore", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        player_stats, goalie_stats, team_stats = parse_boxscore(data)
        return gid, player_stats, goalie_stats, team_stats, None
    except Exception as e:
        return gid, [], [], [], str(e)


def backfill_boxscores(client: NHLApiClient, game_ids: set[int]):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("Backfilling boxscores for %d games...", len(game_ids))
    sorted_ids = sorted(game_ids)
    failed = []
    base_url = client.base_url

    args_list = [(gid, base_url) for gid in sorted_ids]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_single_boxscore, args): args[0]
                   for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Boxscores"):
            gid = futures[future]
            try:
                gid, player_stats, goalie_stats, team_stats, error = future.result()
                if error:
                    logger.debug("Failed boxscore %s: %s", gid, error)
                    failed.append(gid)
                    continue

                with get_session() as session:
                    for ps in player_stats:
                        _ensure_player_exists(session, ps["player_id"], ps["team_id"])
                        upsert_player_game_stats(session, ps)
                    for gs in goalie_stats:
                        _ensure_player_exists(session, gs["player_id"], gs["team_id"], pos="G")
                        upsert_goalie_game_stats(session, gs)
                    for ts in team_stats:
                        upsert_team_game_stats(session, ts)
            except Exception as e:
                logger.error("Failed processing boxscore %s: %s", gid, e)
                failed.append(gid)

    if failed:
        logger.warning("%d boxscores failed: %s", len(failed), failed[:20])


def run_backfill():
    cfg = load_config()
    seasons = cfg["seasons"]
    teams = cfg["teams"]

    init_db()
    client = NHLApiClient()

    backfill_rosters(client, seasons, teams)
    game_ids = backfill_schedules(client, seasons, teams)
    backfill_boxscores(client, game_ids)

    logger.info("Backfill complete!")


if __name__ == "__main__":
    run_backfill()
