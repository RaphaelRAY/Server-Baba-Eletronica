"""Database module with variable and MySQL backends."""

import datetime
from typing import List, Dict, Optional

import os
import base64
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    create_engine = None  # type: ignore
    Column = Integer = String = Float = DateTime = None  # type: ignore
    def declarative_base():  # type: ignore
        class _Base:  # minimal placeholder
            pass
        return _Base
    sessionmaker = None  # type: ignore
    SQLALCHEMY_AVAILABLE = False

Base = declarative_base()

# Lista global de eventos quando o backend está em memória
memory_events: List[Dict[str, str]] = []


if SQLALCHEMY_AVAILABLE:
    class Event(Base):
        """Tabela de eventos para o banco SQL."""

        __tablename__ = "events"
        id = Column(Integer, primary_key=True)
        type = Column(String)
        confidence = Column(Float)
        timestamp = Column(DateTime, default=datetime.datetime.utcnow)
        # Campos opcionais para alinhamento com armazenamento em memória
        level = Column(String, nullable=True)
        image_path = Column(String, nullable=True)
else:
    # Placeholder so type hints don't break; never used without SQLAlchemy
    class Event:  # pragma: no cover - used only when SQLAlchemy missing
        pass


class Database:
    """Banco com suporte a memória ou MySQL."""

    SERVER_MEMORY = 0
    SERVER_MYSQL = 1

    def __init__(self, server: int = SERVER_MEMORY, url: Optional[str] = None):
        """Initialize the database backend."""
        self.server = server
        if server == self.SERVER_MEMORY:
            self._events = memory_events
        elif server == self.SERVER_MYSQL:
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError("SQLAlchemy is required for MySQL/SQL storage")
            if url is None:
                raise ValueError("URL required for MySQL server")
            self.engine = create_engine(url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        else:
            raise ValueError("Invalid server option")

    def save_event(self, data: dict) -> None:
        """Save a new event and optionally store an image.

        Supported input keys in data:
        - type: str – event type
        - confidence: float – confidence score
        - level: str – optional level label
        - image_bytes: bytes – optional JPEG (or raw) bytes to persist
        """

        # Prepare optional image save
        image_path: str | None = None
        image_bytes: bytes | None = data.get("image_bytes")
        if image_bytes:
            try:
                image_path = self._store_image(image_bytes, suffix=data.get("type"))
            except Exception:
                # Do not break event saving if image fails
                image_path = None

        if self.server == self.SERVER_MEMORY:
            # Inline base64 for convenience when using in-memory backend
            img_b64 = (
                base64.b64encode(image_bytes).decode("ascii") if image_bytes else None
            )
            event = {
                "type": data.get("type"),
                "confidence": data.get("confidence"),
                "level": data.get("level", "info"),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "image_path": image_path,
                "image_b64": img_b64,
            }
            self._events.append(event)
        else:
            session = self.Session()
            try:
                event = Event(
                    type=data.get("type"),
                    confidence=data.get("confidence"),
                    level=data.get("level"),
                    image_path=image_path,
                )
                session.add(event)
                session.commit()
            finally:
                session.close()

    def get_recent_events(self, offset: int = 0, limit: int = 50):
        """Return recent events with optional offset and limit."""

        if self.server == self.SERVER_MEMORY:
            events = list(reversed(self._events))
            return events[offset : offset + limit]

        session = self.Session()
        events = (
            session.query(Event)
            .order_by(Event.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        session.close()
        return [self._event_to_dict(e) for e in events]

    def get_all_events(self):
        """Return all events ordered by newest first."""

        if self.server == self.SERVER_MEMORY:
            return list(reversed(self._events))

        session = self.Session()
        events = session.query(Event).order_by(Event.timestamp.desc()).all()
        session.close()
        return [self._event_to_dict(e) for e in events]

    def _event_to_dict(self, e: Event) -> dict:
        """Serialize SQL event row to dict including base64 if available."""
        out = {
            "type": e.type,
            "confidence": e.confidence,
            "timestamp": e.timestamp.isoformat(),
        }
        if getattr(e, "level", None) is not None:
            out["level"] = e.level
        img_path = getattr(e, "image_path", None)
        if img_path:
            out["image_path"] = img_path
            try:
                with open(img_path, "rb") as fh:
                    out["image_b64"] = base64.b64encode(fh.read()).decode("ascii")
            except Exception:
                # If reading fails, omit base64 but keep path for debugging
                pass
        return out

    def _store_image(self, content: bytes, *, suffix: str | None = None) -> str:
        """Persist bytes to a JPEG file and return its path.

        The directory is taken from env var EVENTS_DIR or defaults to data/events.
        Suffix may be appended to the filename to indicate type.
        """
        base_dir = os.getenv("EVENTS_DIR", os.path.join("data", "events"))
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{ts}{('_' + suffix) if suffix else ''}.jpg"
        path = os.path.join(base_dir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path
