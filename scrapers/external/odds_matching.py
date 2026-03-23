"""Normalize Odds API strings vs. NHL DB names (teams + players)."""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

import yaml

from utils.logger import get_logger

logger = get_logger("scrapers.odds_matching")

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_DEFAULT_TEAM_MAP = _CONFIG_DIR / "odds_team_map.yaml"


@lru_cache(maxsize=1)
def load_odds_team_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load (abbrev_upper -> display_name, normalized_alias -> display_name)."""
    abbrev: dict[str, str] = {}
    aliases: dict[str, str] = {}
    path = _DEFAULT_TEAM_MAP
    if path.is_file():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for k, v in (raw.get("abbrev_to_display_name") or {}).items():
                if k and v:
                    abbrev[str(k).strip().upper()] = str(v).strip()
            for k, v in (raw.get("odds_label_aliases") or {}).items():
                if k and v:
                    aliases[normalize_team_label(str(k))] = str(v).strip()
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)
    return abbrev, aliases


def display_name_for_db_team(
    full_name: str | None,
    abbreviation: str | None,
    abbrev_map: dict[str, str],
) -> str:
    """Prefer NHL 3-letter abbrev -> Odds-style display name; else DB full_name."""
    ab = (abbreviation or "").strip().upper()
    if ab and ab in abbrev_map:
        return abbrev_map[ab]
    fn = (full_name or "").strip()
    return fn or ab or ""


def repair_team_typo_label(s: str) -> str:
    """Fix known Odds API typos before matching DB team names."""
    if not s:
        return s
    low = s.lower()
    # Whole-label or substring repairs (case-insensitive)
    if "dallaander" in low:
        return re.sub(r"(?i)dallaanders?", "Dallas Stars", s)
    return s


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

    DB often stores teams.full_name as a 3-letter abbrev (roster parse); Odds API uses
    full city names. We add keys using config/odds_team_map.yaml abbrev_to_display_name.
    """
    abbrev_map, _aliases = load_odds_team_maps()
    lookup: dict[str, int] = {}

    def _add_keys(game_id: int, away: str, home: str) -> None:
        if not away or not home:
            return
        variants = [
            f"{away} @ {home}",
            f"{normalize_team_label(away)} @ {normalize_team_label(home)}",
        ]
        for k in variants:
            if not k:
                continue
            if k not in lookup:
                lookup[k] = game_id
            elif lookup.get(k) != game_id:
                logger.debug("Duplicate game key %r (game_ids %s vs %s)", k, lookup[k], game_id)

    for game_id, away_fn, home_fn, away_ab, home_ab in rows:
        away_db = (away_fn or "").strip()
        home_db = (home_fn or "").strip()
        away_disp = display_name_for_db_team(away_db, away_ab, abbrev_map)
        home_disp = display_name_for_db_team(home_db, home_ab, abbrev_map)
        # Raw DB pair (e.g. CAR @ PIT) and Odds-style pair (Carolina @ Pittsburgh)
        _add_keys(game_id, away_db, home_db)
        _add_keys(game_id, away_disp, home_disp)
        # Abbrev @ abbrev (normalized lowercase)
        aa = (away_ab or "").strip().upper()
        ha = (home_ab or "").strip().upper()
        if aa and ha:
            _add_keys(game_id, aa, ha)

    return lookup


def canonicalize_odds_team_label(raw: str) -> str:
    """Map Odds API team string toward config/display names before key lookup."""
    raw = repair_team_typo_label(raw or "")
    if not raw.strip():
        return raw
    _abbrev_map, alias_map = load_odds_team_maps()
    n = normalize_team_label(raw)
    if n in alias_map:
        return alias_map[n]
    tok = raw.strip().upper()
    if len(tok) <= 3 and tok in _abbrev_map:
        return _abbrev_map[tok]
    return raw.strip()


def _resolve_game_id_once(
    away_team: str,
    home_team: str,
    lookup: dict[str, int],
) -> int | None:
    """Single orientation: away_team @ home_team vs DB."""
    away_team = canonicalize_odds_team_label(away_team)
    home_team = canonicalize_odds_team_label(home_team)
    candidates = [
        f"{away_team} @ {home_team}",
        f"{normalize_team_label(away_team)} @ {normalize_team_label(home_team)}",
    ]
    for c in candidates:
        gid = lookup.get(c)
        if gid is not None:
            return gid
    na, nh = normalize_team_label(away_team), normalize_team_label(home_team)
    nk = f"{na} @ {nh}"
    gid = lookup.get(nk)
    if gid is not None:
        return gid
    if len(lookup) < 500:
        pool = [k for k in lookup if " @ " in k]
        match = get_close_matches(nk, pool, n=1, cutoff=0.92)
        if match:
            logger.info("Fuzzy-matched event to game_id via %r -> %r", nk, match[0])
            return lookup[match[0]]
    return None


def resolve_game_id(
    away_team: str,
    home_team: str,
    lookup: dict[str, int],
    try_swapped: bool = True,
) -> int | None:
    """Resolve Odds API away_team / home_team to our game_id.

    Tries typo repair, normalized keys, fuzzy match, then swapped away/home in case
    the book lists teams in the opposite order vs NHL.
    """
    gid = _resolve_game_id_once(away_team, home_team, lookup)
    if gid is not None:
        return gid
    if try_swapped:
        gid = _resolve_game_id_once(home_team, away_team, lookup)
        if gid is not None:
            logger.info(
                "Matched Odds API event using swapped away/home: %r @ %r",
                home_team,
                away_team,
            )
    return gid


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


def resolve_player_in_roster(
    odds_name: str,
    roster: list[tuple[int, str, str | None]],
    position_hint: str | None = None,
    fuzzy_cutoff: float = 0.88,
) -> int | None:
    """Resolve a book player string to player_id using only home/away roster rows.

    Disambiguates homonyms (e.g. two Sebastian Ahos) by team context. Optional
    ``position_hint`` (from the API outcome, if present) breaks rare remaining ties.
    """
    n = normalize_player_name(odds_name)
    if not n:
        return None
    exact: list[tuple[int, str | None]] = []
    for pid, full, pos in roster:
        if not full:
            continue
        keys = {normalize_player_name(full), full.lower().strip()}
        if n in keys:
            exact.append((pid, pos))
    if len(exact) == 1:
        return exact[0][0]
    if len(exact) > 1:
        if position_hint:
            ph = (position_hint or "").strip().upper()
            for pid, pos in exact:
                p = (pos or "").strip().upper()
                if ph and p and (p.startswith(ph) or ph.startswith(p) or p == ph):
                    return pid
        chosen = min(pid for pid, _ in exact)
        logger.debug(
            "Multiple roster exact matches for %r; using player_id %s",
            odds_name,
            chosen,
        )
        return chosen
    pool: list[tuple[str, int]] = []
    for pid, full, _pos in roster:
        if not full:
            continue
        nn = normalize_player_name(full)
        if nn:
            pool.append((nn, pid))
    if not pool:
        return None
    names = [x[0] for x in pool]
    hits = get_close_matches(n, names, n=3, cutoff=fuzzy_cutoff)
    if not hits:
        return None
    hit = hits[0]
    pids = [pid for nn, pid in pool if nn == hit]
    if len(pids) == 1:
        logger.debug("Fuzzy roster match %r -> %r (id=%s)", odds_name, hit, pids[0])
        return pids[0]
    return min(pids)
