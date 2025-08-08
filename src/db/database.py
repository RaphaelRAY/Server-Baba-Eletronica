"""Database module with variable and MySQL backends."""

import datetime
from typing import List, Dict, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Event(Base):
    """Tabela de eventos para o banco SQL."""

    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    type = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Database:
    """Banco com suporte a memória ou MySQL."""

    SERVER_MEMORY = 0
    SERVER_MYSQL = 1

    def __init__(self, server: int = SERVER_MEMORY, url: Optional[str] = None):
        self.server = server
        if server == self.SERVER_MEMORY:
            self._events: List[Dict[str, str]] = []
        elif server == self.SERVER_MYSQL:
            if url is None:
                raise ValueError("URL required for MySQL server")
            self.engine = create_engine(url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        else:
            raise ValueError("Invalid server option")

    def save_event(self, data: dict) -> None:
        """Save a new event according to the server type."""

        if self.server == self.SERVER_MEMORY:
            event = {
                "type": data.get("type"),
                "confidence": data.get("confidence"),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
            self._events.append(event)
        else:
            session = self.Session()
            event = Event(type=data.get("type"), confidence=data.get("confidence"))
            session.add(event)
            session.commit()
            session.close()

    def get_recent_events(self, limit: int = 50):
        """Return recent events from the chosen backend."""

        if self.server == self.SERVER_MEMORY:
            return list(reversed(self._events[-limit:]))
        session = self.Session()
        events = (
            session.query(Event)
            .order_by(Event.timestamp.desc())
            .limit(limit)
            .all()
        )
        session.close()
        return [
            {
                "type": e.type,
                "confidence": e.confidence,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ]
