"""Base HTTP client for the NHL Web API with rate limiting and retries."""

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("nhl_api.client")


class NHLApiClient:
    def __init__(self):
        cfg = load_config()["nhl_api"]
        self.base_url = cfg["base_url"]
        self.delay = cfg.get("request_delay", 0.5)
        self.max_retries = cfg.get("max_retries", 3)
        self.backoff = cfg.get("retry_backoff", 2.0)
        self._last_request_time = 0.0

        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def get(self, endpoint: str, params: dict = None) -> dict | None:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = self.backoff ** (attempt + 2)
                    logger.warning("429 rate limited on %s, waiting %.0fs...", url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.warning("404 Not Found: %s", url)
                    return None
                logger.error("HTTP error for %s: %s", url, e)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff ** attempt)
            except requests.exceptions.RequestException as e:
                logger.error("Request failed for %s: %s", url, e)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff ** attempt)
        return None

    def get_roster(self, team_abbrev: str, season: int) -> dict | None:
        return self.get(f"roster/{team_abbrev}/{season}")

    def get_schedule(self, team_abbrev: str, season: int) -> dict | None:
        return self.get(f"club-schedule-season/{team_abbrev}/{season}")

    def get_boxscore(self, game_id: int) -> dict | None:
        return self.get(f"gamecenter/{game_id}/boxscore")

    def get_standings(self, date_str: str = "now") -> dict | None:
        return self.get(f"standings/{date_str}")

    def get_schedule_date(self, date_str: str) -> dict | None:
        return self.get(f"schedule/{date_str}")

    def get_play_by_play(self, game_id: int) -> dict | None:
        return self.get(f"gamecenter/{game_id}/play-by-play")
