from pymongo import MongoClient
from pymongo.collection import Collection
from config import settings


class MongoDB:
    """Singleton MongoDB connection for backend_service."""
    _client = None
    _db = None

    @classmethod
    def get_db(cls):
        if cls._client is None:
            cls._client = MongoClient(settings.MONGO_URI)
            cls._db = cls._client[settings.DATABASE_NAME]
        return cls._db

    @classmethod
    def get_collection(cls, name: str) -> Collection:
        return cls.get_db()[name]
