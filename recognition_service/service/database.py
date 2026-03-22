from pymongo import MongoClient
from pymongo.collection import Collection


class MongoDBManager:
    """Lightweight MongoDB wrapper."""

    def __init__(self, uri: str, database_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[database_name]
        self.collection_name = collection_name

    def get_collection(self) -> Collection:
        return self.db[self.collection_name]

    def insert_one(self, document: dict):
        return self.get_collection().insert_one(document)

    def find_one(self, query: dict):
        return self.get_collection().find_one(query)

    def find_all(self, query: dict = {}):
        return list(self.get_collection().find(query))

    def update_one(self, query: dict, update: dict):
        return self.get_collection().update_one(query, {"$set": update})

    def delete_one(self, query: dict):
        return self.get_collection().delete_one(query)
