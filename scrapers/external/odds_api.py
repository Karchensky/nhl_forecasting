"""Client for The Odds API to fetch FanDuel player goal scorer odds."""

import os
from datetime import date, datetime

import requests
from sqlalchemy import text as sa_text

from database.db_client import get_engine, get_session
from database.ingestion import upsert_odds
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("scrapers.odds_api")

# Do not refresh stored odds once a game is in progress or finished (avoids overwriting
# pregame prices with live / post markets the model is not built for).
BLOCK_ODDS_REFRESH_GAME_STATES = frozenset(
    {"LIVE", "CRIT", "OFF", "FINAL"},
)


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
    """Convert American odds to implied probability (before vig removal)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def implied_to_american(implied: float) -> int:
    """Map implied probability (0,1) to American odds (integer)."""
    p = float(implied)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    if p >= 0.5:
        return -int(round(100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def probability_to_american(p: float) -> int:
    """Fair-money American line for a win probability (same math as implied)."""
    return implied_to_american(p)


def format_american_line(odds: int) -> str:
    """Format for display, e.g. -140, +250."""
    if odds > 0:
        return f"+{odds}"
    return str(odds)


def normalize_book_price(price) -> tuple[int, float]:
    """Return (american_odds, implied_probability) from an API ``price`` field.

    The Odds API usually sends **American** odds (e.g. -200, +1200). Some payloads
    send **decimal** odds (e.g. 1.91, **13** for a longshot ≈ +1200).

    If a small positive number is mis-read as American, you get absurd results:
    decimal **13** (≈7.7% implied, +1200-ish) wrongly as **+13** American → ~89%
    implied (looks like a **−769** favorite when re-converted).

    Rules:

    - **Negative** → American favorite.
    - **≥ 100** → American underdog (+1200, +150, …).
    - **1.01 … 50** → **decimal** odds (``implied = 1/decimal``), covering EU-style
      prices and integer **13** / **8.5** style longshots.
    - **51 … 99** → American (+75, +85, …) — uncommon on props but valid.
    """
    if price is None or price == 0:
        raise ValueError("invalid book price")
    pf = float(price)

    if pf < 0:
        am = int(round(pf))
        return am, american_to_implied(am)

    if pf >= 100:
        am = int(round(pf))
        return am, american_to_implied(am)

    if 1.01 <= pf <= 50.0:
        implied = 1.0 / pf
        return implied_to_american(implied), implied

    am = int(round(pf))
    return am, american_to_implied(am)


def fetch_roster_rows_for_game(game_id: int) -> list[tuple[int, str, str | None]]:
    """Skaters on home/away teams for this game (excludes goalies). Uses current_team_id."""
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT home_team_id, away_team_id FROM games WHERE game_id = :g"),
            {"g": game_id},
        ).fetchone()
        if not row:
            return []
        h, a = int(row[0]), int(row[1])
        rows = conn.execute(
            text("""
                SELECT player_id, full_name, position FROM players
                WHERE (active IS NULL OR active = 1)
                  AND COALESCE(UPPER(TRIM(position)), '') != 'G'
                  AND current_team_id IN (:h, :a)
            """),
            {"h": h, "a": a},
        ).fetchall()
    return [(int(r[0]), r[1], r[2]) for r in rows]


def _event_payload(data) -> dict:
    """Odds API may return a dict or a one-element list for event odds."""
    if isinstance(data, list):
        return data[0] if data else {}
    return data if isinstance(data, dict) else {}


def parse_player_goal_odds(
    event_data,
    game_id: int,
    player_id_lookup: dict | None = None,
    player_lookup_keys: list[str] | None = None,
    roster: list[tuple[int, str, str | None]] | None = None,
) -> list[dict]:
    """Parse player goal odds from The Odds API response into DB records."""
    from scrapers.external.odds_matching import resolve_player_id, resolve_player_in_roster

    event_data = _event_payload(event_data)
    records = []
    bookmakers = event_data.get("bookmakers", [])
    keys = (
        player_lookup_keys
        if player_lookup_keys is not None
        else sorted((player_id_lookup or {}).keys())
    )

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

                position_hint = (
                    outcome.get("position")
                    or outcome.get("participant_position")
                    or outcome.get("player_position")
                )
                if roster:
                    player_id = resolve_player_in_roster(
                        name, roster, position_hint=position_hint
                    )
                elif player_id_lookup:
                    player_id = resolve_player_id(
                        name, player_id_lookup, keys, fuzzy_cutoff=0.91
                    )
                else:
                    player_id = None

                if player_id and price != 0:
                    try:
                        am, imp = normalize_book_price(price)
                    except ValueError:
                        continue
                    records.append({
                        "player_id": player_id,
                        "game_id": game_id,
                        "sportsbook": book_key,
                        "market": "anytime_goal_scorer",
                        "american_odds": am,
                        "implied_probability": imp,
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

    if not game_id_lookup:
        logger.warning("No game_id_lookup provided; cannot store odds.")
        return

    pkeys = player_lookup_keys or sorted((player_id_lookup or {}).keys())
    total_stored = 0
    skipped_events = []

    today = date.today()
    today_s = str(today)
    engine = get_engine()

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

        with engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT game_date, game_state FROM games WHERE game_id = :g"
                ),
                {"g": game_id},
            ).fetchone()
        if not row:
            logger.warning("No games row for game_id=%s; skipping odds fetch", game_id)
            continue
        gd = row[0]
        gstate = (row[1] or "").strip().upper() if row[1] else ""

        if gd is not None and str(gd) != today_s:
            logger.debug(
                "Skipping Odds API event %s (%s @ %s): game_date=%s (today=%s)",
                event_id,
                away,
                home,
                gd,
                today_s,
            )
            continue

        if gstate in BLOCK_ODDS_REFRESH_GAME_STATES:
            logger.info(
                "Skipping odds refresh for game_id=%s (%s @ %s): game_state=%s "
                "(live or final — keep existing pregame lines)",
                game_id,
                away,
                home,
                gstate,
            )
            continue

        _time.sleep(1.5)
        odds_data = client.get_player_goal_odds(event_id)
        if not odds_data:
            logger.warning("Empty odds payload for event %s (%s @ %s)", event_id, away, home)
            continue

        roster = fetch_roster_rows_for_game(game_id)
        if roster:
            records = parse_player_goal_odds(odds_data, game_id, roster=roster)
        elif player_id_lookup:
            logger.warning(
                "Game %s: empty roster-scoped skater list; using global player map "
                "(may miss matches if current_team_id is stale)",
                game_id,
            )
            records = parse_player_goal_odds(
                odds_data, game_id, player_id_lookup, pkeys
            )
        else:
            logger.warning("Game %s: no roster rows and no global player lookup", game_id)
            records = []

        if not records and roster and player_id_lookup:
            records = parse_player_goal_odds(
                odds_data, game_id, player_id_lookup, pkeys
            )
            if records:
                logger.info(
                    "Game %s: roster-scoped match failed; stored %d rows via global map",
                    game_id,
                    len(records),
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
