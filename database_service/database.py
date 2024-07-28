from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

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
