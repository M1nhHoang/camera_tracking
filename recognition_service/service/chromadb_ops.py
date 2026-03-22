import logging
from uuid import uuid4
import chromadb


class ChromaDBClient:
    """Wrapper for ChromaDB face embedding operations."""

    def __init__(self, host: str, port: int, collection_name: str = "face_tracking"):
        client = chromadb.HttpClient(host=host, port=port)
        try:
            self.collection = client.create_collection(collection_name)
            logging.info("ChromaDB collection created")
        except Exception:
            self.collection = client.get_collection(collection_name)
            logging.info("ChromaDB collection already exists")

    def insert(self, metadata: dict, embedding: list):
        """Insert a new face embedding."""
        self.collection.add(
            ids=[str(uuid4())],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def search(self, embedding: list, n_results: int = 1) -> dict:
        """Search for similar face embeddings."""
        return self.collection.query(
            query_embeddings=[embedding], n_results=n_results
        )

    def get_by_user(self, user_id: str, include=None) -> dict:
        """Get all embeddings for a user."""
        include = include or ["metadatas"]
        return self.collection.get(where={"user_id": user_id}, include=include)

    def delete_by_user(self, user_id: str) -> bool:
        """Delete all embeddings for a user."""
        results = self.collection.get(where={"user_id": user_id})
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return True
        return False

    def update_user_metadata(self, user_id: str, new_metadata: dict) -> bool:
        """Update metadata for all embeddings of a user."""
        results = self.collection.get(where={"user_id": user_id})
        if not results or not results["ids"]:
            return False
        for idx, id in enumerate(results["ids"]):
            updated = {**results["metadatas"][idx], **new_metadata}
            self.collection.update(ids=[id], metadatas=[updated])
        return True

    def count_by_user(self, user_id: str) -> int:
        """Count embeddings for a user."""
        results = self.collection.get(where={"user_id": user_id})
        return len(results["ids"]) if results else 0

    def delete_by_ids(self, ids: list):
        """Delete embeddings by IDs."""
        self.collection.delete(ids=ids)

    def get(self, **kwargs) -> dict:
        """Passthrough to collection.get()."""
        return self.collection.get(**kwargs)
