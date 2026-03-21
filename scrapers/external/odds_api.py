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

    def get_player_goal_odds(self, event_id: str) -> list[dict]:
        """Get player anytime goal scorer odds for a specific event."""
        if not self._check_api_key():
            return []

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
            return []


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def parse_player_goal_odds(event_data: dict, game_id: int,
                           player_id_lookup: dict) -> list[dict]:
    """Parse player goal odds from The Odds API response into DB records.

    player_id_lookup: maps lowercase player name -> player_id
    """
    records = []
    bookmakers = event_data.get("bookmakers", [])

    for book in bookmakers:
        book_key = book.get("key", "unknown")
        for market in book.get("markets", []):
            if market.get("key") != "player_goal_scorer_anytime":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("description", outcome.get("name", ""))
                price = outcome.get("price", 0)

                name_lower = name.lower().strip()
                player_id = player_id_lookup.get(name_lower)

                if player_id and price != 0:
                    records.append({
                        "player_id": player_id,
                        "game_id": game_id,
                        "sportsbook": book_key,
                        "market": "anytime_goal_scorer",
                        "american_odds": price,
                        "implied_probability": american_to_implied(price),
                        "retrieved_at": datetime.utcnow(),
                    })

    return records


def fetch_and_store_odds(game_id_lookup: dict = None, player_id_lookup: dict = None):
    """Fetch today's odds and store them.

    game_id_lookup: maps 'away @ home' style key -> game_id
    player_id_lookup: maps lowercase player name -> player_id
    """
    client = OddsApiClient()
    events = client.get_events()

    if not events:
        logger.info("No events found or API key not set.")
        return

    logger.info("Found %d upcoming events", len(events))

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        odds_data = client.get_player_goal_odds(event_id)
        if not odds_data:
            continue

        if player_id_lookup and game_id_lookup:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            key = f"{away} @ {home}"
            game_id = game_id_lookup.get(key)

            if game_id:
                records = parse_player_goal_odds(odds_data, game_id, player_id_lookup)
                if records:
                    with get_session() as session:
                        for r in records:
                            upsert_odds(session, r)
                    logger.info("Stored %d odds for game %s", len(records), game_id)
