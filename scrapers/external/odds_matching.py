"""Normalize Odds API strings vs. NHL DB names (teams + players)."""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

from utils.logger import get_logger

logger = get_logger("scrapers.odds_matching")


def normalize_team_label(s: str) -> str:
    """ASCII-ish lowercase team string for fuzzy matchup keys."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_player_name(s: str) -> str:
    """Normalize player name from Odds API / NHL."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    for suffix in (
        " (c)", " (a)", " (c.)", " (a.)",
        " (lw)", " (rw)", " (d)", " (g)",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


def build_game_id_lookup_from_rows(rows: list[tuple]) -> dict[str, int]:
    """Map many string variants -> game_id.

    Each row: (game_id, away_full, home_full, away_abbr, home_abbr)
    """
    lookup: dict[str, int] = {}
    for game_id, away_fn, home_fn, away_ab, home_ab in rows:
        pairs = [
            (away_fn, home_fn),
            (away_ab, home_ab),
        ]
        for away, home in pairs:
            if not away or not home:
                continue
            keys = [
                f"{away} @ {home}",
                f"{normalize_team_label(away)} @ {normalize_team_label(home)}",
            ]
            for k in keys:
                if k and k not in lookup:
                    lookup[k] = game_id
                elif k and lookup.get(k) != game_id:
                    logger.debug("Duplicate game key %r (game_ids %s vs %s)", k, lookup[k], game_id)
    return lookup


def resolve_game_id(
    away_team: str,
    home_team: str,
    lookup: dict[str, int],
) -> int | None:
    """Resolve Odds API home_team / away_team to our game_id."""
    candidates = [
        f"{away_team} @ {home_team}",
        f"{normalize_team_label(away_team)} @ {normalize_team_label(home_team)}",
    ]
    for c in candidates:
        gid = lookup.get(c)
        if gid is not None:
            return gid
    # Last resort: normalized lookup scan (handles Montréal vs Montreal spacing)
    na, nh = normalize_team_label(away_team), normalize_team_label(home_team)
    nk = f"{na} @ {nh}"
    gid = lookup.get(nk)
    if gid is not None:
        return gid
    # Fuzzy on full "away @ home" keys only
    if len(lookup) < 500:
        pool = [k for k in lookup if " @ " in k]
        match = get_close_matches(nk, pool, n=1, cutoff=0.92)
        if match:
            logger.info("Fuzzy-matched event to game_id via %r -> %r", nk, match[0])
            return lookup[match[0]]
    return None


def build_player_id_lookup(
    rows: list[tuple[int, str]],
    fuzzy_cutoff: float = 0.91,
) -> tuple[dict[str, int], list[str]]:
    """Exact + normalized keys; return (lookup, sorted_keys_for_fuzzy)."""
    lookup: dict[str, int] = {}
    for pid, full in rows:
        if not full:
            continue
        keys = {full.lower().strip(), normalize_player_name(full)}
        for k in keys:
            if not k:
                continue
            if k in lookup and lookup[k] != pid:
                logger.warning(
                    "Player lookup collision on %r: ids %s vs %s (skipping duplicate key)",
                    k,
                    lookup[k],
                    pid,
                )
                continue
            lookup[k] = pid
    sorted_keys = sorted(lookup.keys())
    return lookup, sorted_keys


def resolve_player_id(
    odds_name: str,
    lookup: dict[str, int],
    sorted_keys: list[str],
    fuzzy_cutoff: float = 0.91,
) -> int | None:
    n = normalize_player_name(odds_name)
    if not n:
        return None
    pid = lookup.get(n)
    if pid is not None:
        return pid
    # Odds API sometimes uses "First Last" vs DB "F. Last" — try last token match? risky
    matches = get_close_matches(n, sorted_keys, n=1, cutoff=fuzzy_cutoff)
    if matches:
        m = matches[0]
        pid = lookup[m]
        logger.debug("Fuzzy player match %r -> %r (id=%s)", odds_name, m, pid)
        return pid
    return None
