from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()


class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    type = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_event(self, data: dict) -> None:
        session = self.Session()
        event = Event(type=data.get('type'), confidence=data.get('confidence'))
        session.add(event)
        session.commit()
        session.close()

    def get_recent_events(self, limit: int = 50):
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
                'type': e.type,
                'confidence': e.confidence,
                'timestamp': e.timestamp.isoformat(),
            }
            for e in events
        ]
