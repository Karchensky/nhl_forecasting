"""Upsert helpers for idempotent data ingestion."""

from sqlalchemy import inspect
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from database.models import (
    Game,
    GoalieGameStats,
    ModelOutput,
    Odds,
    Player,
    PlayerGameStats,
    Team,
    TeamGameStats,
)


def upsert_team(session: Session, data: dict):
    stmt = sqlite_insert(Team.__table__).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["team_id"],
        set_={k: v for k, v in data.items() if k != "team_id"},
    )
    session.execute(stmt)


def upsert_player(session: Session, data: dict):
    stmt = sqlite_insert(Player.__table__).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id"],
        set_={k: v for k, v in data.items() if k != "player_id"},
    )
    session.execute(stmt)


def upsert_game(session: Session, data: dict):
    stmt = sqlite_insert(Game.__table__).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["game_id"],
        set_={k: v for k, v in data.items() if k != "game_id"},
    )
    session.execute(stmt)


def upsert_player_game_stats(session: Session, data: dict):
    stmt = sqlite_insert(PlayerGameStats.__table__).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "game_id"],
        set_={k: v for k, v in data.items() if k not in ("player_id", "game_id")},
    )
    session.execute(stmt)


def upsert_goalie_game_stats(session: Session, data: dict):
    stmt = sqlite_insert(GoalieGameStats.__table__).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "game_id"],
        set_={k: v for k, v in data.items() if k not in ("player_id", "game_id")},
    )
    session.execute(stmt)


def upsert_team_game_stats(session: Session, data: dict):
    stmt = sqlite_insert(TeamGameStats.__table__).values(**data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["team_id", "game_id"],
        set_={k: v for k, v in data.items() if k not in ("team_id", "game_id")},
    )
    session.execute(stmt)


def upsert_odds(session: Session, data: dict):
    """Insert or update by (player_id, game_id, sportsbook, market).

    Must match ``Odds`` unique constraint — **not** ``id`` (autoincrement),
    or every insert is treated as new and repeats violate the unique index.
    """
    clean = {k: v for k, v in data.items() if k != "id"}
    stmt = sqlite_insert(Odds.__table__).values(**clean)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "game_id", "sportsbook", "market"],
        set_={
            "american_odds": clean["american_odds"],
            "implied_probability": clean["implied_probability"],
            "retrieved_at": clean["retrieved_at"],
        },
    )
    session.execute(stmt)


def upsert_model_output(session: Session, data: dict):
    clean = {k: v for k, v in data.items() if k != "id"}
    stmt = sqlite_insert(ModelOutput.__table__).values(**clean)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "game_id", "model_version"],
        set_={"predicted_probability": clean["predicted_probability"]},
    )
    session.execute(stmt)
