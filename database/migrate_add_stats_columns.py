"""One-time migration: add Stats API enrichment columns to existing tables."""

from sqlalchemy import text
from database.db_client import get_engine

PLAYER_GAME_STATS_COLUMNS = [
    ("shifts", "INTEGER"),
    ("ot_toi_seconds", "INTEGER"),
    ("time_on_ice_per_shift", "REAL"),
    ("corsi_for", "INTEGER"),
    ("corsi_against", "INTEGER"),
    ("fenwick_for", "INTEGER"),
    ("fenwick_against", "INTEGER"),
    ("oz_start_pct", "REAL"),
    ("dz_start_pct", "REAL"),
    ("individual_corsi_for", "INTEGER"),
    ("individual_shots_for_per60", "REAL"),
    ("on_ice_shooting_pct", "REAL"),
    ("on_ice_save_pct", "REAL"),
    ("pdo", "REAL"),
    ("pp_shots", "INTEGER"),
    ("pp_individual_sat_for", "INTEGER"),
    ("pp_toi_pct_per_game", "REAL"),
    ("missed_shots", "INTEGER"),
    ("total_shot_attempts", "INTEGER"),
    ("first_goals", "INTEGER"),
    ("empty_net_goals", "INTEGER"),
    ("es_goals_for", "INTEGER"),
    ("es_goals_against", "INTEGER"),
    ("oz_faceoffs", "INTEGER"),
    ("dz_faceoffs", "INTEGER"),
    ("nz_faceoffs", "INTEGER"),
    ("total_faceoffs", "INTEGER"),
    ("ev_faceoff_pct", "REAL"),
]

TEAM_GAME_STATS_COLUMNS = [
    ("pp_opportunities_actual", "INTEGER"),
    ("pp_pct", "REAL"),
    ("pp_toi_seconds", "INTEGER"),
    ("times_shorthanded", "INTEGER"),
    ("pk_pct", "REAL"),
    ("goals_5v5", "INTEGER"),
    ("goals_5v4", "INTEGER"),
    ("goals_against_5v5", "INTEGER"),
    ("goals_against_5v4", "INTEGER"),
]


def _get_existing_columns(conn, table_name: str) -> set[str]:
    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
    return {row[1] for row in result.fetchall()}


def migrate():
    engine = get_engine()
    with engine.connect() as conn:
        for table, columns in [
            ("player_game_stats", PLAYER_GAME_STATS_COLUMNS),
            ("team_game_stats", TEAM_GAME_STATS_COLUMNS),
        ]:
            existing = _get_existing_columns(conn, table)
            added = 0
            for col_name, col_type in columns:
                if col_name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                    ))
                    added += 1
            conn.commit()
            print(f"  {table}: added {added} columns, skipped {len(columns) - added}")
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
