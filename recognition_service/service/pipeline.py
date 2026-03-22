import logging
import threading
import time
from queue import Queue, Empty

from service.face_utils import FaceProcessor
from service.embedding_client import EmbeddingClient
from service.chromadb_ops import ChromaDBClient
from service.tracking_store import TrackingStore

# Batch configuration
BATCH_INTERVAL = 1.0   # Accumulate items for 1 second
BATCH_MAX_SIZE = 16     # Max items per batch (GPU memory limit)


class IdentificationPipeline:
    """
    Batch identification pipeline.
    Accumulates detections over BATCH_INTERVAL seconds,
    then processes them as a single batch for GPU efficiency.
    """

    def __init__(
        self,
        face_processor: FaceProcessor,
        embedding_client: EmbeddingClient,
        chromadb_client: ChromaDBClient,
        tracking_store: TrackingStore,
        detect_threshold: float = 0.40,
    ):
        self.face_processor = face_processor
        self.embedding_client = embedding_client
        self.chromadb = chromadb_client
        self.tracking_store = tracking_store
        self.detect_threshold = detect_threshold

        self._queue = Queue()
        self._thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._thread.start()

    def enqueue(
        self, detect_id, origin_image, detect_image,
        detect_threshold=None, camera_id=None, face_image=None,
    ):
        """Add detection to processing queue."""
        threshold = detect_threshold or self.detect_threshold
        self._queue.put((
            detect_id, origin_image, detect_image,
            threshold, camera_id, face_image,
        ))

    def _batch_loop(self):
        """Main batch loop — accumulate items, then process as batch."""
        while True:
            batch = self._collect_batch()
            if not batch:
                continue

            try:
                self._process_batch(batch)
            except Exception as e:
                logging.error(f"Batch pipeline error: {e}")

    def _collect_batch(self):
        """Collect items from queue for up to BATCH_INTERVAL seconds."""
        batch = []
        deadline = time.time() + BATCH_INTERVAL

        # Block on first item (don't spin when idle)
        try:
            item = self._queue.get(timeout=BATCH_INTERVAL)
            batch.append(item)
        except Empty:
            return []

        # Collect remaining items until deadline or max size
        while len(batch) < BATCH_MAX_SIZE and time.time() < deadline:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except Empty:
                break

        return batch

    def _process_batch(self, batch):
        """
        Process a batch of detections:
        1. Face detection for items without pre-cropped face
        2. Batch embedding (single GPU forward pass)
        3. Batch ChromaDB search
        4. Cache results + send to database
        """
        # Step 1: Prepare face images
        valid_items = []
        face_images = []

        for item in batch:
            (detect_id, origin_image, detect_image,
             detect_threshold, camera_id, face_image) = item

            # Use pre-cropped face or detect locally
            if face_image is None:
                face_image = self.face_processor.detect(detect_image)
            if face_image is None or (isinstance(face_image, list) and len(face_image) == 0):
                self._queue.task_done()
                continue

            valid_items.append(item)
            face_images.append(face_image)

        if not valid_items:
            return

        # Step 2: Batch embedding (1 gRPC call → 1 GPU forward pass)
        embeddings = self.embedding_client.get_embedding_batch(face_images)

        # Step 3: Process each result
        for i, item in enumerate(valid_items):
            try:
                (detect_id, origin_image, detect_image,
                 detect_threshold, camera_id, face_image) = item

                embedding = embeddings[i]
                if embedding is None:
                    self._queue.task_done()
                    continue

                # Vector search
                search_results = self.chromadb.search(embedding)
                distances = search_results["distances"][0][0]
                metadata = search_results["metadatas"][0][0]

                # Determine match
                is_matched = distances < detect_threshold
                matched_user_id = metadata["user_id"] if is_matched else None
                matched_user_name = metadata.get("user_name", "Unknown") if is_matched else "Unknown"

                # Cache result for gRPC GetTrackingInfo
                from grpc_server import tracking_cache
                tracking_cache.set(detect_id, matched_user_name, not is_matched)

                # Save tracking log directly to MongoDB
                self.tracking_store.save_tracking(
                    detect_id=detect_id,
                    origin_image=self.face_processor.to_base64(origin_image),
                    detect_image=self.face_processor.to_base64(detect_image),
                    face_image=self.face_processor.to_base64(face_images[i]),
                    truth_image_path=metadata["truth_image_path"],
                    distance=distances,
                    camera_id=camera_id,
                    user_id=matched_user_id,
                )

            except Exception as e:
                logging.error(f"Error processing batch item {i}: {e}")
            finally:
                self._queue.task_done()
