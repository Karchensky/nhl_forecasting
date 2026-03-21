"""HTTP client for the NHL Stats API (api.nhle.com/stats/rest/en/).

Provides per-game, per-player advanced stats across multiple report types.
Handles pagination (max 100 rows/page), rate limiting, and retries.
"""

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("nhl_stats_api.client")

PAGE_SIZE = 100
TOTAL_CAP = 10_000


class NHLStatsApiClient:
    def __init__(self):
        cfg = load_config()["nhl_api"]
        self.base_url = cfg.get("stats_url", "https://api.nhle.com/stats/rest/en")
        self.delay = cfg.get("request_delay", 1.0)
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

    def _get_page(self, url: str, params: dict) -> dict | None:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = self.backoff ** (attempt + 2)
                    logger.warning("429 rate limited, waiting %.0fs ...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.warning("404 Not Found: %s", url)
                    return None
                logger.error("HTTP error %s: %s", url, e)
                if attempt == self.max_retries:
                    return None
                time.sleep(self.backoff ** attempt)
            except requests.exceptions.RequestException as e:
                logger.error("Request failed %s: %s", url, e)
                if attempt == self.max_retries:
                    return None
                time.sleep(self.backoff ** attempt)
        return None

    def fetch_report(
        self,
        entity: str,
        report_type: str,
        cayenne_exp: str,
    ) -> list[dict]:
        """Fetch all rows for a report, auto-paginating through results.

        Args:
            entity: 'skater', 'goalie', or 'team'
            report_type: e.g. 'timeonice', 'powerplay', 'summary'
            cayenne_exp: Filter expression, e.g. 'seasonId=20252026 and teamId=22'

        Returns:
            List of row dicts from the API.
        """
        url = f"{self.base_url}/{entity}/{report_type}"
        all_rows: list[dict] = []
        start = 0

        while True:
            params = {
                "isAggregate": "false",
                "isGame": "true",
                "limit": PAGE_SIZE,
                "start": start,
                "cayenneExp": cayenne_exp,
            }
            data = self._get_page(url, params)
            if data is None:
                logger.error(
                    "Failed to fetch %s/%s start=%d filter=%s",
                    entity, report_type, start, cayenne_exp,
                )
                break

            rows = data.get("data", [])
            if not rows:
                break

            all_rows.extend(rows)
            total = data.get("total", 0)

            if start + PAGE_SIZE >= total or start + PAGE_SIZE >= TOTAL_CAP:
                break
            start += PAGE_SIZE

        return all_rows

    def fetch_skater_report(
        self, report_type: str, season_id: int, team_id: int
    ) -> list[dict]:
        cayenne = f"seasonId={season_id} and teamId={team_id}"
        return self.fetch_report("skater", report_type, cayenne)

    def fetch_team_report(
        self, report_type: str, season_id: int, team_id: int
    ) -> list[dict]:
        cayenne = f"seasonId={season_id} and teamId={team_id}"
        return self.fetch_report("team", report_type, cayenne)

    def fetch_skater_report_by_game(
        self, report_type: str, game_id: int
    ) -> list[dict]:
        cayenne = f"gameId={game_id}"
        return self.fetch_report("skater", report_type, cayenne)

    def fetch_team_report_by_game(
        self, report_type: str, game_id: int
    ) -> list[dict]:
        cayenne = f"gameId={game_id}"
        return self.fetch_report("team", report_type, cayenne)
