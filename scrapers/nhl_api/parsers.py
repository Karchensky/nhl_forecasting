"""Parsers that convert NHL API JSON responses into database-ready dicts."""

from datetime import date, datetime


def parse_toi(toi_str: str) -> int:
    """Convert 'MM:SS' time-on-ice string to total seconds."""
    if not toi_str:
        return 0
    parts = toi_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def _localized(obj) -> str:
    """Extract the default localized string from NHL API name objects."""
    if isinstance(obj, dict):
        return obj.get("default", "")
    return str(obj) if obj else ""


def parse_roster(data: dict, team_abbrev: str, team_id: int) -> tuple[dict, list[dict]]:
    """Parse roster response into team dict and list of player dicts."""
    team_data = {
        "team_id": team_id,
        "abbreviation": team_abbrev,
        "full_name": team_abbrev,
    }

    players = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            pos_code = p.get("positionCode", "")
            if group == "goalies":
                pos_code = "G"
            raw_bd = p.get("birthDate")
            bd = None
            if raw_bd:
                try:
                    bd = datetime.strptime(raw_bd, "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    bd = None

            players.append({
                "player_id": p["id"],
                "full_name": f"{_localized(p.get('firstName', ''))} {_localized(p.get('lastName', ''))}".strip(),
                "position": pos_code,
                "birth_date": bd,
                "shoots": p.get("shootsCatches"),
                "current_team_id": team_id,
                "height_inches": p.get("heightInInches"),
                "weight_lbs": p.get("weightInPounds"),
                "active": True,
            })

    return team_data, players


def parse_schedule(data: dict) -> list[dict]:
    """Parse club-schedule-season response into list of game dicts."""
    games = []
    for g in data.get("games", []):
        game_type = g.get("gameType", 0)
        if game_type not in (2, 3):
            continue

        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        state = g.get("gameState", "")

        raw_date = g.get("gameDate")
        game_date = None
        if raw_date:
            try:
                game_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                game_date = None

        games.append({
            "game_id": g["id"],
            "season": g.get("season", 0),
            "game_type": game_type,
            "game_date": game_date,
            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "game_state": state,
            "venue": _localized(g.get("venue", "")),
        })

    return games


def parse_boxscore(data: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse boxscore into player_game_stats, goalie_game_stats, team_game_stats lists."""
    game_id = data["id"]
    home_team = data.get("homeTeam", {})
    away_team = data.get("awayTeam", {})
    pbgs = data.get("playerByGameStats", {})

    player_stats = []
    goalie_stats = []
    team_stats = []

    for side, team_info, is_home in [
        ("homeTeam", home_team, True),
        ("awayTeam", away_team, False),
    ]:
        team_id = team_info.get("id")
        side_data = pbgs.get(side, {})

        p_goals = 0
        p_shots = 0
        p_pim = 0
        p_pp_goals = 0
        p_hits = 0
        p_blocks = 0
        p_takeaways = 0
        p_giveaways = 0

        for group in ("forwards", "defense"):
            for p in side_data.get(group, []):
                toi_sec = parse_toi(p.get("toi", ""))
                goals = p.get("goals", 0)
                shots = p.get("sog", 0)
                pim = p.get("pim", 0)
                pp_g = p.get("powerPlayGoals", 0)
                hits = p.get("hits", 0)
                blocks = p.get("blockedShots", 0)
                ta = p.get("takeaways", 0)
                ga = p.get("giveaways", 0)

                p_goals += goals
                p_shots += shots
                p_pim += pim
                p_pp_goals += pp_g
                p_hits += hits
                p_blocks += blocks
                p_takeaways += ta
                p_giveaways += ga

                faceoff_pct = p.get("faceoffWinningPctg", 0.0)
                shifts = p.get("shifts", 0)

                player_stats.append({
                    "player_id": p["playerId"],
                    "game_id": game_id,
                    "team_id": team_id,
                    "goals": goals,
                    "assists": p.get("assists", 0),
                    "points": p.get("points", 0),
                    "shots": shots,
                    "hits": hits,
                    "blocked_shots": blocks,
                    "pim": pim,
                    "plus_minus": p.get("plusMinus", 0),
                    "toi_seconds": toi_sec,
                    "pp_toi_seconds": 0,
                    "sh_toi_seconds": 0,
                    "ev_toi_seconds": 0,
                    "pp_goals": pp_g,
                    "sh_goals": 0,
                    "gw_goals": 0,
                    "ot_goals": 0,
                    "faceoff_wins": 0,
                    "faceoff_losses": 0,
                    "takeaways": ta,
                    "giveaways": ga,
                })

        for g in side_data.get("goalies", []):
            toi_sec = parse_toi(g.get("toi", ""))
            sa = g.get("shotsAgainst", 0)
            sv = g.get("saves", 0)
            ga_count = g.get("goalsAgainst", 0)

            es_str = g.get("evenStrengthShotsAgainst", "0/0")
            pp_str = g.get("powerPlayShotsAgainst", "0/0")
            sh_str = g.get("shorthandedShotsAgainst", "0/0")

            def _parse_saves_str(s):
                parts = s.split("/")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
                return 0, 0

            es_saves, es_sa = _parse_saves_str(es_str)
            pp_saves, pp_sa = _parse_saves_str(pp_str)
            sh_saves, sh_sa = _parse_saves_str(sh_str)

            goalie_stats.append({
                "player_id": g["playerId"],
                "game_id": game_id,
                "team_id": team_id,
                "decision": g.get("decision"),
                "saves": sv,
                "shots_against": sa,
                "goals_against": ga_count,
                "save_pct": g.get("savePctg"),
                "toi_seconds": toi_sec,
                "pp_saves": pp_saves,
                "sh_saves": sh_saves,
                "ev_saves": es_saves,
                "started": g.get("starter", False),
            })

        score = team_info.get("score", 0)
        sog = team_info.get("sog", 0)
        opp_pim = 0
        opp_side = "awayTeam" if is_home else "homeTeam"
        for group in ("forwards", "defense"):
            for p in pbgs.get(opp_side, {}).get(group, []):
                opp_pim += p.get("pim", 0)
        pp_opps = max(1, opp_pim // 2) if opp_pim > 0 else 0

        opp_score = (away_team if is_home else home_team).get("score", 0)
        won = score > opp_score if score is not None and opp_score is not None else None

        team_stats.append({
            "team_id": team_id,
            "game_id": game_id,
            "goals": score,
            "shots": sog if sog else p_shots,
            "pim": p_pim,
            "pp_goals": p_pp_goals,
            "pp_opportunities": pp_opps,
            "faceoff_win_pct": None,
            "blocked_shots": p_blocks,
            "hits": p_hits,
            "takeaways": p_takeaways,
            "giveaways": p_giveaways,
            "is_home": is_home,
            "won": won,
        })

    return player_stats, goalie_stats, team_stats
