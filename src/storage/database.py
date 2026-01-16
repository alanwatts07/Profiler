"""Database connection and session management."""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLSession

from .models import Base
from ..config import config


class Database:
    """Database connection manager."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection.

        Args:
            database_url: SQLAlchemy database URL. Uses config if not provided.
        """
        self.database_url = database_url or config.DATABASE_URL
        self.engine = create_engine(
            self.database_url,
            echo=False,
            future=True,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables. Use with caution!"""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[SQLSession, None, None]:
        """Get a database session as a context manager.

        Usage:
            with db.get_session() as session:
                session.query(Speaker).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_new_session(self) -> SQLSession:
        """Get a new session (caller responsible for closing)."""
        return self.SessionLocal()


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_db(database_url: Optional[str] = None) -> Database:
    """Initialize and return the database, creating tables if needed."""
    global _db
    _db = Database(database_url)
    _db.create_tables()
    return _db
