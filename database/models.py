from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_id = Column(Integer, primary_key=True)
    abbreviation = Column(String(3), nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    conference = Column(String(50))
    division = Column(String(50))

    players = relationship("Player", back_populates="team")


class Player(Base):
    __tablename__ = "players"

    player_id = Column(Integer, primary_key=True)
    full_name = Column(String(100), nullable=False)
    position = Column(String(5))
    birth_date = Column(Date)
    shoots = Column(String(1))
    current_team_id = Column(Integer, ForeignKey("teams.team_id"))
    height_inches = Column(Integer)
    weight_lbs = Column(Integer)
    active = Column(Boolean, default=True)

    team = relationship("Team", back_populates="players")


class Game(Base):
    __tablename__ = "games"

    game_id = Column(Integer, primary_key=True)
    season = Column(Integer, nullable=False, index=True)
    game_type = Column(Integer, nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    game_state = Column(String(10))
    venue = Column(String(100))

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"

    player_id = Column(Integer, ForeignKey("players.player_id"), primary_key=True)
    game_id = Column(Integer, ForeignKey("games.game_id"), primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    points = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    hits = Column(Integer, default=0)
    blocked_shots = Column(Integer, default=0)
    pim = Column(Integer, default=0)
    plus_minus = Column(Integer, default=0)
    toi_seconds = Column(Integer, default=0)
    pp_toi_seconds = Column(Integer, default=0)
    sh_toi_seconds = Column(Integer, default=0)
    ev_toi_seconds = Column(Integer, default=0)
    pp_goals = Column(Integer, default=0)
    sh_goals = Column(Integer, default=0)
    gw_goals = Column(Integer, default=0)
    ot_goals = Column(Integer, default=0)
    faceoff_wins = Column(Integer, default=0)
    faceoff_losses = Column(Integer, default=0)
    takeaways = Column(Integer, default=0)
    giveaways = Column(Integer, default=0)

    # --- Stats API enrichment columns (Phase 0) ---
    shifts = Column(Integer)
    ot_toi_seconds = Column(Integer)
    time_on_ice_per_shift = Column(Float)
    corsi_for = Column(Integer)
    corsi_against = Column(Integer)
    fenwick_for = Column(Integer)
    fenwick_against = Column(Integer)
    oz_start_pct = Column(Float)
    dz_start_pct = Column(Float)
    individual_corsi_for = Column(Integer)
    individual_shots_for_per60 = Column(Float)
    on_ice_shooting_pct = Column(Float)
    on_ice_save_pct = Column(Float)
    pdo = Column(Float)
    pp_shots = Column(Integer)
    pp_individual_sat_for = Column(Integer)
    pp_toi_pct_per_game = Column(Float)
    missed_shots = Column(Integer)
    total_shot_attempts = Column(Integer)
    first_goals = Column(Integer)
    empty_net_goals = Column(Integer)
    es_goals_for = Column(Integer)
    es_goals_against = Column(Integer)
    oz_faceoffs = Column(Integer)
    dz_faceoffs = Column(Integer)
    nz_faceoffs = Column(Integer)
    total_faceoffs = Column(Integer)
    ev_faceoff_pct = Column(Float)

    player = relationship("Player")
    game = relationship("Game")
    team = relationship("Team")


class GoalieGameStats(Base):
    __tablename__ = "goalie_game_stats"

    player_id = Column(Integer, ForeignKey("players.player_id"), primary_key=True)
    game_id = Column(Integer, ForeignKey("games.game_id"), primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    decision = Column(String(5))
    saves = Column(Integer, default=0)
    shots_against = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    save_pct = Column(Float)
    toi_seconds = Column(Integer, default=0)
    pp_saves = Column(Integer, default=0)
    sh_saves = Column(Integer, default=0)
    ev_saves = Column(Integer, default=0)
    started = Column(Boolean, default=False)

    player = relationship("Player")
    game = relationship("Game")
    team = relationship("Team")


class TeamGameStats(Base):
    __tablename__ = "team_game_stats"

    team_id = Column(Integer, ForeignKey("teams.team_id"), primary_key=True)
    game_id = Column(Integer, ForeignKey("games.game_id"), primary_key=True)
    goals = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    pim = Column(Integer, default=0)
    pp_goals = Column(Integer, default=0)
    pp_opportunities = Column(Integer, default=0)
    faceoff_win_pct = Column(Float)
    blocked_shots = Column(Integer, default=0)
    hits = Column(Integer, default=0)
    takeaways = Column(Integer, default=0)
    giveaways = Column(Integer, default=0)
    is_home = Column(Boolean, nullable=False)
    won = Column(Boolean)

    # --- Stats API enrichment columns (Phase 0) ---
    pp_opportunities_actual = Column(Integer)
    pp_pct = Column(Float)
    pp_toi_seconds = Column(Integer)
    times_shorthanded = Column(Integer)
    pk_pct = Column(Float)
    goals_5v5 = Column(Integer)
    goals_5v4 = Column(Integer)
    goals_against_5v5 = Column(Integer)
    goals_against_5v4 = Column(Integer)

    team = relationship("Team")
    game = relationship("Game")


class Odds(Base):
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.game_id"), nullable=False)
    sportsbook = Column(String(50), nullable=False)
    market = Column(String(50), nullable=False)
    american_odds = Column(Integer, nullable=False)
    implied_probability = Column(Float, nullable=False)
    retrieved_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "game_id", "sportsbook", "market",
                         name="uq_odds_player_game_book_market"),
    )

    player = relationship("Player")
    game = relationship("Game")


class ShotEvent(Base):
    """Individual shot-level events from play-by-play data for xG modeling."""
    __tablename__ = "shot_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.game_id"), nullable=False, index=True)
    event_id = Column(Integer, nullable=False)
    period = Column(Integer, nullable=False)
    period_type = Column(String(5), nullable=False)        # REG, OT
    time_in_period_seconds = Column(Integer, nullable=False)
    event_type = Column(String(20), nullable=False)        # goal, shot-on-goal, missed-shot, blocked-shot
    x_coord = Column(Float)
    y_coord = Column(Float)
    zone_code = Column(String(1))                          # O, D, N
    shot_type = Column(String(20))                         # wrist, snap, slap, backhand, tip-in, deflected, wrap-around, poke
    shooter_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    goalie_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)  # NULL = empty net
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    situation_code = Column(String(4))                     # e.g. 1551 = 5v5
    is_goal = Column(Boolean, nullable=False, default=False)
    distance = Column(Float)                               # feet from net center
    angle = Column(Float)                                  # degrees from center line

    __table_args__ = (
        UniqueConstraint("game_id", "event_id", name="uq_shot_event_game_event"),
    )

    game = relationship("Game")
    shooter = relationship("Player", foreign_keys=[shooter_id])
    goalie = relationship("Player", foreign_keys=[goalie_id])
    team = relationship("Team")


class ModelOutput(Base):
    __tablename__ = "model_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.game_id"), nullable=False)
    model_version = Column(String(50), nullable=False)
    predicted_probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "game_id", "model_version",
                         name="uq_model_output_player_game_version"),
    )

    player = relationship("Player")
    game = relationship("Game")
