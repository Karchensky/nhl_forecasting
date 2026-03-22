"""Client for The Odds API to fetch FanDuel player goal scorer odds."""

import os
from datetime import datetime

import requests

from database.db_client import get_session
from database.ingestion import upsert_odds
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("scrapers.odds_api")


class OddsApiClient:
    def __init__(self):
        cfg = load_config()["odds_api"]
        self.base_url = cfg["base_url"]
        self.api_key = cfg.get("api_key") or os.environ.get("ODDS_API_KEY", "")
        self.sport = cfg["sport"]
        self.regions = cfg["regions"]
        self.bookmakers = cfg["bookmakers"]
        self.session = requests.Session()

    def _check_api_key(self):
        if not self.api_key:
            logger.warning(
                "No Odds API key configured. Set ODDS_API_KEY env var or "
                "add to config/settings.yaml."
            )
            return False
        return True

    def get_events(self) -> list[dict]:
        """Get upcoming NHL events/games."""
        if not self._check_api_key():
            return []

        url = f"{self.base_url}/sports/{self.sport}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": "h2h",
            "bookmakers": self.bookmakers,
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info("Odds API requests remaining: %s", remaining)
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch events: %s", e)
            return []

    def get_player_goal_odds(self, event_id: str) -> dict | list:
        """Event odds payload: usually a dict; some responses wrap a list."""
        if not self._check_api_key():
            return {}

        url = f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": "player_goal_scorer_anytime",
            "bookmakers": self.bookmakers,
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info("Odds API requests remaining: %s", remaining)
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch player odds for %s: %s", event_id, e)
            return {}


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def _event_payload(data) -> dict:
    """Odds API may return a dict or a one-element list for event odds."""
    if isinstance(data, list):
        return data[0] if data else {}
    return data if isinstance(data, dict) else {}


def parse_player_goal_odds(
    event_data,
    game_id: int,
    player_id_lookup: dict,
    player_lookup_keys: list[str] | None = None,
) -> list[dict]:
    """Parse player goal odds from The Odds API response into DB records."""
    from scrapers.external.odds_matching import resolve_player_id

    event_data = _event_payload(event_data)
    records = []
    bookmakers = event_data.get("bookmakers", [])
    keys = player_lookup_keys or sorted(player_id_lookup.keys())

    for book in bookmakers:
        book_key = book.get("key", "unknown")
        for market in book.get("markets", []):
            if market.get("key") != "player_goal_scorer_anytime":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("description", outcome.get("name", ""))
                price = outcome.get("price", 0)
                if not name or price == 0:
                    continue

                player_id = resolve_player_id(
                    name, player_id_lookup, keys, fuzzy_cutoff=0.91
                )

                if player_id and price != 0:
                    records.append({
                        "player_id": player_id,
                        "game_id": game_id,
                        "sportsbook": book_key,
                        "market": "anytime_goal_scorer",
                        "american_odds": int(price),
                        "implied_probability": american_to_implied(int(price)),
                        "retrieved_at": datetime.utcnow(),
                    })

    return records


def fetch_and_store_odds(
    game_id_lookup: dict = None,
    player_id_lookup: dict = None,
    player_lookup_keys: list[str] | None = None,
):
    """Fetch today's odds and store them.

    game_id_lookup: maps 'away @ home' style key -> game_id (multiple variants per game)
    player_id_lookup: normalized / lowercase player name -> player_id
    player_lookup_keys: sorted keys for difflib fuzzy matching (from daily_job)
    """
    import time as _time

    from scrapers.external.odds_matching import resolve_game_id

    client = OddsApiClient()
    events = client.get_events()

    if not events:
        logger.info("No events found or API key not set.")
        return

    logger.info("Found %d upcoming events", len(events))

    if not player_id_lookup or not game_id_lookup:
        logger.warning("No game_id_lookup or player_id_lookup provided; cannot store odds.")
        return

    pkeys = player_lookup_keys or sorted(player_id_lookup.keys())
    total_stored = 0
    skipped_events = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        home = event.get("home_team", "") or ""
        away = event.get("away_team", "") or ""
        game_id = resolve_game_id(away, home, game_id_lookup)

        if not game_id:
            skipped_events.append(f"{away} @ {home}")
            continue

        _time.sleep(1.5)
        odds_data = client.get_player_goal_odds(event_id)
        if not odds_data:
            logger.warning("Empty odds payload for event %s (%s @ %s)", event_id, away, home)
            continue

        records = parse_player_goal_odds(
            odds_data, game_id, player_id_lookup, pkeys
        )
        slate_key = f"{away} @ {home}"
        if records:
            with get_session() as session:
                for r in records:
                    upsert_odds(session, r)
            total_stored += len(records)
            logger.info("Stored %d odds for game %s (%s)", len(records), game_id, slate_key)
        else:
            logger.warning(
                "No player-name matches for game %s (%s); check Odds API names vs. "
                "players.full_name in DB.",
                game_id,
                slate_key,
            )

    if total_stored == 0:
        logger.warning(
            "Odds fetch finished with 0 stored rows — check team strings vs DB "
            "(Montréal/Montreal, Utah names) or player names vs players.full_name."
        )
        if skipped_events:
            logger.warning(
                "Events with no DB game match (first 5): %s",
                skipped_events[:5],
            )
    else:
        logger.info("Total odds rows upserted this run: %d", total_stored)
