"""Parsers that map NHL Stats API report rows to DB column updates.

Each parser function takes a list of raw API rows and returns a list of
(key_dict, update_dict) tuples where key_dict identifies the row to update
and update_dict contains the column values to set.
"""


def _safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Skater report parsers
# ---------------------------------------------------------------------------

def parse_timeonice(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/timeonice: PP/SH/EV/OT TOI, shifts, TOI per shift."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "pp_toi_seconds": _safe_int(r.get("ppTimeOnIce")),
                "sh_toi_seconds": _safe_int(r.get("shTimeOnIce")),
                "ev_toi_seconds": _safe_int(r.get("evTimeOnIce")),
                "ot_toi_seconds": _safe_int(r.get("otTimeOnIce")),
                "shifts": _safe_int(r.get("shifts")),
                "time_on_ice_per_shift": _safe_float(r.get("timeOnIcePerShift")),
            },
        ))
    return results


def parse_summary(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/summary: SH/GW/OT goals, faceoff win %, EV goals."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "sh_goals": _safe_int(r.get("shGoals")),
                "gw_goals": _safe_int(r.get("gameWinningGoals")),
                "ot_goals": _safe_int(r.get("otGoals")),
                "faceoff_wins": None,
                "faceoff_losses": None,
            },
        ))
    return results


def parse_puck_possessions(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/puckPossessions: Corsi/Fenwick %, zone starts, shot rates."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "oz_start_pct": _safe_float(r.get("offensiveZoneStartPct")),
                "dz_start_pct": _safe_float(r.get("defensiveZoneStartPct")),
                "individual_shots_for_per60": _safe_float(
                    r.get("individualShotsForPer60")
                ),
                "on_ice_shooting_pct": _safe_float(r.get("onIceShootingPct")),
            },
        ))
    return results


def parse_goals_for_against(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/goalsForAgainst: on-ice EV goals for/against."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "es_goals_for": _safe_int(r.get("evenStrengthGoalsFor")),
                "es_goals_against": _safe_int(r.get("evenStrengthGoalsAgainst")),
            },
        ))
    return results


def parse_powerplay(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/powerplay: PP shots, shot attempts, TOI share."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "pp_shots": _safe_int(r.get("ppShots")),
                "pp_individual_sat_for": _safe_int(r.get("ppIndividualSatFor")),
                "pp_toi_pct_per_game": _safe_float(r.get("ppTimeOnIcePctPerGame")),
            },
        ))
    return results


def parse_percentages(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/percentages: PDO (shooting + save %), on-ice save %."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "pdo": _safe_float(r.get("skaterShootingPlusSavePct5v5")),
                "on_ice_save_pct": _safe_float(r.get("skaterSavePct5v5")),
            },
        ))
    return results


def parse_realtime(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/realtime: missed shots, total attempts, first/EN goals."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "missed_shots": _safe_int(r.get("missedShots")),
                "total_shot_attempts": _safe_int(r.get("totalShotAttempts")),
                "first_goals": _safe_int(r.get("firstGoals")),
                "empty_net_goals": _safe_int(r.get("emptyNetGoals")),
            },
        ))
    return results


def parse_faceoff_percentages(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/faceoffpercentages: zone faceoffs, EV faceoff %."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        total = _safe_int(r.get("totalFaceoffs")) or 0
        win_pct = _safe_float(r.get("faceoffWinPct"))
        faceoff_wins = None
        faceoff_losses = None
        if total > 0 and win_pct is not None:
            faceoff_wins = round(total * win_pct)
            faceoff_losses = total - faceoff_wins
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "oz_faceoffs": _safe_int(r.get("offensiveZoneFaceoffs")),
                "dz_faceoffs": _safe_int(r.get("defensiveZoneFaceoffs")),
                "nz_faceoffs": _safe_int(r.get("neutralZoneFaceoffs")),
                "total_faceoffs": _safe_int(r.get("totalFaceoffs")),
                "ev_faceoff_pct": _safe_float(r.get("evFaceoffPct")),
                "faceoff_wins": faceoff_wins,
                "faceoff_losses": faceoff_losses,
            },
        ))
    return results


# We also need corsi_for/against from summaryshooting
def parse_summaryshooting(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /skater/summaryshooting: Corsi for/against, Fenwick for/against."""
    results = []
    for r in rows:
        pid = r.get("playerId")
        gid = r.get("gameId")
        if not pid or not gid:
            continue
        results.append((
            {"player_id": pid, "game_id": gid},
            {
                "corsi_for": _safe_int(r.get("satFor")),
                "corsi_against": _safe_int(r.get("satAgainst")),
                "fenwick_for": _safe_int(r.get("usatFor")),
                "fenwick_against": _safe_int(r.get("usatAgainst")),
                "individual_corsi_for": _safe_int(r.get("satTotal")),
            },
        ))
    return results


# ---------------------------------------------------------------------------
# Team report parsers
# ---------------------------------------------------------------------------

def parse_team_powerplay(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /team/powerplay: actual PP opportunities, PP %, PP TOI."""
    results = []
    for r in rows:
        tid = r.get("teamId")
        gid = r.get("gameId")
        if not tid or not gid:
            continue
        results.append((
            {"team_id": tid, "game_id": gid},
            {
                "pp_opportunities_actual": _safe_int(r.get("ppOpportunities")),
                "pp_pct": _safe_float(r.get("powerPlayPct")),
                "pp_toi_seconds": _safe_int(r.get("ppTimeOnIcePerGame")),
            },
        ))
    return results


def parse_team_penaltykill(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /team/penaltykill: times shorthanded, PK %."""
    results = []
    for r in rows:
        tid = r.get("teamId")
        gid = r.get("gameId")
        if not tid or not gid:
            continue
        results.append((
            {"team_id": tid, "game_id": gid},
            {
                "times_shorthanded": _safe_int(r.get("timesShorthanded")),
                "pk_pct": _safe_float(r.get("penaltyKillPct")),
            },
        ))
    return results


def parse_team_goals_by_strength(rows: list[dict]) -> list[tuple[dict, dict]]:
    """From /team/goalsforbystrength: goals at each game state."""
    results = []
    for r in rows:
        tid = r.get("teamId")
        gid = r.get("gameId")
        if not tid or not gid:
            continue
        results.append((
            {"team_id": tid, "game_id": gid},
            {
                "goals_5v5": _safe_int(r.get("goalsFor5On5")),
                "goals_5v4": _safe_int(r.get("goalsFor5On4")),
            },
        ))
    return results


# The /team/goalsagainstbystrength fills the "against" side
def parse_team_goals_against_by_strength(
    rows: list[dict],
) -> list[tuple[dict, dict]]:
    results = []
    for r in rows:
        tid = r.get("teamId")
        gid = r.get("gameId")
        if not tid or not gid:
            continue
        results.append((
            {"team_id": tid, "game_id": gid},
            {
                "goals_against_5v5": _safe_int(r.get("goalsAgainst5On5")),
                "goals_against_5v4": _safe_int(r.get("goalsAgainst5On4")),
            },
        ))
    return results


# ---------------------------------------------------------------------------
# Registry mapping report names to parser functions and target table
# ---------------------------------------------------------------------------

SKATER_REPORTS = {
    "timeonice": parse_timeonice,
    "summary": parse_summary,
    "puckPossessions": parse_puck_possessions,
    "goalsForAgainst": parse_goals_for_against,
    "powerplay": parse_powerplay,
    "percentages": parse_percentages,
    "realtime": parse_realtime,
    "faceoffpercentages": parse_faceoff_percentages,
    "summaryshooting": parse_summaryshooting,
}

TEAM_REPORTS = {
    "powerplay": parse_team_powerplay,
    "penaltykill": parse_team_penaltykill,
    "goalsforbystrength": parse_team_goals_by_strength,
    "goalsagainstbystrength": parse_team_goals_against_by_strength,
}
