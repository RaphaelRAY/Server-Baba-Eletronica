"""Database module with in-memory, SQLAlchemy and MongoDB backends."""

import datetime
from typing import List, Dict, Optional, Callable

import os
import base64
import logging
logger = logging.getLogger(__name__)
import time
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

try:  # optional dependency for MongoDB backend
    import pymongo
    from pymongo import MongoClient
    from pymongo.collection import Collection
    PYMONGO_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    MongoClient = None  # type: ignore
    Collection = None  # type: ignore
    PYMONGO_AVAILABLE = False

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
    SERVER_MONGO = 2

    def __init__(self, server: int = SERVER_MEMORY, url: Optional[str] = None):
        """Initialize the database backend."""
        self.server = server
        # Optional sink to publish events when saved (e.g., SSE broadcast)
        self._event_sink: Optional[Callable[[dict], None]] = None
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
        elif server == self.SERVER_MONGO:
            if not PYMONGO_AVAILABLE:
                raise ImportError("pymongo is required for MongoDB storage")
            if url is None:
                raise ValueError("URL required for MongoDB server")
            self._mongo_client = MongoClient(url)
            # Determine database from URL, env or default
            try:
                db = self._mongo_client.get_default_database()
            except Exception:
                db = None
            if db is None:
                db_name = os.getenv("MONGO_DB", "baby_monitor")
                db = self._mongo_client[db_name]
            self._mongo_db = db
            col_name = os.getenv("MONGO_COLLECTION", "events")
            self._mongo_col: Collection = db[col_name]
            try:
                self._mongo_col.create_index([("timestamp", pymongo.DESCENDING)])
            except Exception:
                pass
        else:
            raise ValueError("Invalid server option")

        # Cooldown for repeated events by type (in seconds)
        try:
            self._cooldown_default = float(os.getenv("EVENT_COOLDOWN_SECS", "0") or 0)
        except Exception:
            self._cooldown_default = 0.0
        # Load per-type overrides from env: EVENT_COOLDOWN_<TYPE>
        self._cooldown_by_type: Dict[str, float] = {}
        for key, val in os.environ.items():
            if key.startswith("EVENT_COOLDOWN_") and key != "EVENT_COOLDOWN_SECS":
                typ = key[len("EVENT_COOLDOWN_"):].lower()
                try:
                    self._cooldown_by_type[typ] = float(val)
                except Exception:
                    continue
        # Track last saved timestamps by type
        self._last_saved_ts: Dict[str, float] = {}

    def set_event_sink(self, sink: Callable[[dict], None] | None) -> None:
        """Set a callback to receive sanitized event dicts after save."""
        self._event_sink = sink

    def _publish_event(self, event: dict) -> None:
        """Publish an event to the sink without heavy image payloads."""
        if not self._event_sink:
            return
        try:
            # Strip large/binary fields if present
            sanitized = {k: v for k, v in event.items() if k not in ("image_b64", "image_bytes")}
            self._event_sink(sanitized)
        except Exception:
            # Never break main flow on publish errors
            pass

    def save_event(self, data: dict) -> None:
        """Save a new event and optionally store an image.

        Supported input keys in data:
        - type: str – event type
        - confidence: float – confidence score
        - level: str – optional level label
        - image_bytes: bytes – optional JPEG (or raw) bytes to persist
        """

        # Global de-duplication by type with cooldown BEFORE any I/O
        typ = (data.get("type") or "").lower() or "unknown"
        now = time.time()
        cooldown = self._cooldown_by_type.get(typ, self._cooldown_default)
        last_ts = self._last_saved_ts.get(typ)
        if cooldown and last_ts is not None and (now - last_ts) < cooldown:
            try:
                logger.info(
                    "Evento suprimido por cooldown: type=%s restante=%.1fs",
                    typ,
                    cooldown - (now - last_ts),
                )
            except Exception:
                pass
            return

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
            self._last_saved_ts[typ] = now
            try:
                logger.info(
                    "Evento salvo (memory): type=%s level=%s conf=%s image=%s",
                    event.get("type"),
                    event.get("level"),
                    event.get("confidence"),
                    "yes" if image_path or img_b64 else "no",
                )
            except Exception:
                pass
            # Publish event to sink
            try:
                self._publish_event(event)
            except Exception:
                pass
        elif self.server == self.SERVER_MYSQL:
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
                self._last_saved_ts[typ] = now
                # Publish via event dict serialization
                try:
                    self._publish_event(self._event_to_dict(event))
                except Exception:
                    pass
                try:
                    logger.info(
                        "Evento salvo (sql): id=%s type=%s level=%s conf=%s image=%s",
                        getattr(event, "id", None),
                        getattr(event, "type", None),
                        getattr(event, "level", None),
                        getattr(event, "confidence", None),
                        "yes" if image_path else "no",
                    )
                except Exception:
                    pass
            finally:
                session.close()
        elif self.server == self.SERVER_MONGO:
            try:
                doc = {
                    "type": data.get("type"),
                    "confidence": data.get("confidence"),
                    "level": data.get("level", "info"),
                    "timestamp": datetime.datetime.utcnow(),
                    "image_path": image_path,
                }
                result = self._mongo_col.insert_one(doc)
                self._last_saved_ts[typ] = now
                # Publish using a lightweight dict
                try:
                    self._publish_event(self._mongo_doc_to_dict(doc))
                except Exception:
                    pass
                try:
                    logger.info(
                        "Evento salvo (mongo): id=%s type=%s level=%s conf=%s image=%s",
                        str(getattr(result, "inserted_id", None)),
                        doc.get("type"),
                        doc.get("level"),
                        doc.get("confidence"),
                        "yes" if image_path else "no",
                    )
                except Exception:
                    pass
            except Exception as e:  # pragma: no cover - I/O path
                logger.error("Erro ao salvar evento no MongoDB: %s", e)
        else:
            # Should not happen
            raise RuntimeError("Invalid database server configuration")

    def get_recent_events(self, offset: int = 0, limit: int = 50):
        """Return recent events with optional offset and limit."""

        if self.server == self.SERVER_MEMORY:
            events = list(reversed(self._events))
            return events[offset : offset + limit]

        if self.server == self.SERVER_MYSQL:
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
        elif self.server == self.SERVER_MONGO:
            cursor = (
                self._mongo_col.find()
                .sort("timestamp", -1)
                .skip(int(offset))
                .limit(int(limit))
            )
            return [self._mongo_doc_to_dict(d) for d in cursor]
        else:
            raise RuntimeError("Invalid database server configuration")

    def get_all_events(self):
        """Return all events ordered by newest first."""

        if self.server == self.SERVER_MEMORY:
            return list(reversed(self._events))

        if self.server == self.SERVER_MYSQL:
            session = self.Session()
            events = session.query(Event).order_by(Event.timestamp.desc()).all()
            session.close()
            return [self._event_to_dict(e) for e in events]
        elif self.server == self.SERVER_MONGO:
            cursor = self._mongo_col.find().sort("timestamp", -1)
            return [self._mongo_doc_to_dict(d) for d in cursor]
        else:
            raise RuntimeError("Invalid database server configuration")

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

    def _mongo_doc_to_dict(self, d: dict) -> dict:
        """Serialize MongoDB document to public dict including base64 if available."""
        out = {
            "type": d.get("type"),
            "confidence": d.get("confidence"),
            "timestamp": (
                d.get("timestamp").isoformat() if d.get("timestamp") else None
            ),
        }
        if d.get("level") is not None:
            out["level"] = d.get("level")
        img_path = d.get("image_path")
        if img_path:
            out["image_path"] = img_path
            try:
                with open(img_path, "rb") as fh:
                    out["image_b64"] = base64.b64encode(fh.read()).decode("ascii")
            except Exception:
                pass
        return out
