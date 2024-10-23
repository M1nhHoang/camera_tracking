import redis
import json
import uuid

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from typing import Dict, List, Any

# get from env
uri = "mongodb://mongo_db:27017/"
database_name = "camera_traking"


class MongoDBManager:
    def __init__(
        self,
        uri: str = uri,
        database_name: str = database_name,
        collection_name: str = "",
    ):
        self.client = MongoClient(uri)
        self.database_name = database_name
        self.db = self.client[database_name]
        self.collection_name = collection_name

        if not self.collection_name:
            raise ValueError("collection_name can not empty")

    def get_collection(self) -> Collection:
        return self.db[self.collection_name]

    def insert_one(self, document: dict):
        collection = self.get_collection()
        return collection.insert_one(document)

    def find_one(self, query: dict):
        collection = self.get_collection()
        return collection.find_one(query)

    def find_all(self, query: dict = {}):
        collection = self.get_collection()
        return list(collection.find(query))

    def update_one(self, query: dict, update: dict):
        collection = self.get_collection()
        return collection.update_one(query, {"$set": update})

    def delete_one(self, query: dict):
        collection = self.get_collection()
        return collection.delete_one(query)

    def close_connection(self):
        self.client.close()


class RedisManager:
    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        document_id = str(uuid.uuid4())
        document["_id"] = document_id

        json_data = json.dumps(document)

        self.client.set(f"{collection}:{document_id}", json_data)

        return document_id

    def find_one(self, collection: str, query: Dict[str, Any]):
        for key in self.client.scan_iter(f"{collection}:*"):
            data = self.client.get(key)
            document = json.loads(data)

            if all(document.get(k) == v for k, v in query.items()):
                return document

        return None

    def find_all(
        self, collection: str, query: Dict[str, Any] = {}
    ) -> List[Dict[str, Any]]:
        results = []
        for key in self.client.scan_iter(f"{collection}:*"):
            data = self.client.get(key)
            document = json.loads(data)

            if all(document.get(k) == v for k, v in query.items()):
                results.append(document)

        return results

    def update_one(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any]
    ) -> bool:
        for key in self.client.scan_iter(f"{collection}:*"):
            data = self.client.get(key)
            document = json.loads(data)

            if all(document.get(k) == v for k, v in query.items()):
                document.update(update)
                self.client.set(key, json.dumps(document))
                return True

        return False

    def delete_one(self, collection: str, query: Dict[str, Any]) -> bool:
        for key in self.client.scan_iter(f"{collection}:*"):
            data = self.client.get(key)
            document = json.loads(data)

            if all(document.get(k) == v for k, v in query.items()):
                self.client.delete(key)
                return True

        return False

    def bulk_insert(
        self, collection: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        pipe = self.client.pipeline()
        ids = []

        for document in documents:
            document_id = str(uuid.uuid4())
            document["_id"] = document_id
            ids.append(document_id)

            json_data = json.dumps(document)
            pipe.set(f"{collection}:{document_id}", json_data)

        pipe.execute()
        return ids

    def get_by_id(self, collection: str, document_id: str):
        data = self.client.get(f"{collection}:{document_id}")
        return json.loads(data) if data else None

    def close_connection(self):
        self.client.close()
