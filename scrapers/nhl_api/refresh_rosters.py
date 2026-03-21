"""Quick roster refresh to restore proper player names."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scrapers.nhl_api.client import NHLApiClient
from scrapers.nhl_api.parsers import parse_roster
from scrapers.nhl_api.backfill import TEAM_ID_MAP, SEASON_TEAM_ID_OVERRIDES, _teams_for_season, _team_id_for_season
from database.db_client import get_session, init_db
from database.ingestion import upsert_player, upsert_team
from utils.config import load_config
from utils.logger import get_logger
from tqdm import tqdm

logger = get_logger("refresh_rosters")

def main():
    init_db()
    cfg = load_config()
    client = NHLApiClient()

    for season in cfg["seasons"]:
        teams = _teams_for_season(cfg["teams"], season)
        for abbrev in tqdm(teams, desc=f"Rosters {season}"):
            team_id = _team_id_for_season(abbrev, season)
            if not team_id:
                continue
            data = client.get_roster(abbrev, season)
            if not data:
                continue
            team_data, players = parse_roster(data, abbrev, team_id)
            with get_session() as session:
                upsert_team(session, team_data)
                for p in players:
                    upsert_player(session, p)

    logger.info("Roster refresh complete!")

if __name__ == "__main__":
    main()
