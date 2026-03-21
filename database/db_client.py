from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base
from utils.config import load_config

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = load_config()
        db_path = Path(cfg["database"]["sqlite_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        _engine = create_engine(
            url,
            echo=cfg["database"].get("echo", False),
            connect_args={"timeout": 60},
        )

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session() -> Session:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def reset_db():
    """Drop and recreate all tables. USE WITH CAUTION."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
