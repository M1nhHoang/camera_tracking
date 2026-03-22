# Camera Tracking System

Real-time multi-camera surveillance system with person detection, face identification, and cross-camera tracking.

## Architecture

```
Camera Streams
      |
      v
+-------------------+     gRPC (50051)     +----------------------+     gRPC (50052)     +------------------------+
| detection_service |--------------------->| recognition_service  |--------------------->| face_embedding_service |
| (GPU)             |                      | (CPU)                |                      | (GPU)                  |
|                   |                      |                      |                      |                        |
| - YOLO26n detect  |                      | - Batch pipeline     |                      | - Facenet 128-dim      |
| - ByteTrack       |                      | - ChromaDB search    |                      | - DeepFace (modified)  |
| - Best-shot cache |                      | - Tracking logs      |                      |                        |
| - TTL gating      |                      | - MongoDB direct     |                      |                        |
+-------------------+                      +----------------------+                      +------------------------+
      |                                           |
      |  HTTP (startup)                           |  pymongo
      v                                           v
+-------------------+                      +------------+     +------------+
| backend_service   |---pymongo----------->|  MongoDB   |     |  ChromaDB  |
| (CPU)             |                      +------------+     +------------+
|                   |
| - Web UI (Jinja2) |
| - REST API        |
| - User management |
| - Camera CRUD     |
| - Detection stats |
+-------------------+
```

## Services

| Service | Port | Protocol | GPU | Description |
|---------|------|----------|-----|-------------|
| **detection_service** | 5000 | HTTP + gRPC client | Yes | Camera stream capture, YOLO26n person+face detection, ByteTrack multi-object tracking, best-shot selection with TTL cache |
| **recognition_service** | 8002, 50051 | HTTP + gRPC server | No | Identification pipeline orchestrator. Receives detections via gRPC, generates embeddings via face_embedding_service, searches ChromaDB for identity match, caches results, saves tracking logs to MongoDB |
| **face_embedding_service** | 8001, 50052 | HTTP + gRPC server | Yes | Generates 128-dimensional face embeddings using Facenet model (modified DeepFace). Supports batch inference |
| **backend_service** | 80 | HTTP | No | Web UI dashboard, REST API for cameras/users/detections, serves static files. Connects to MongoDB directly |
| **MongoDB** | 27017 | - | No | Stores users, cameras, detection logs |
| **ChromaDB** | 8000 | - | No | Vector database for face embeddings |

## Detection Pipeline

```
1. Camera stream -> OpenCV VideoCapture
2. YOLO26n unified detection -> person + face bounding boxes (single forward pass)
3. Spatial association -> match face bbox to person bbox
4. ByteTrack -> persistent track IDs across frames
5. Best-shot selection -> keep highest quality face per track (Laplacian variance)
6. TTL gating -> send to recognition every 3s (unknown) / 5min (identified)
7. gRPC fire-and-forget -> recognition_service (non-blocking, ~6ms)
8. Image compression -> face 160x160 Q=70, log images Q=80
9. Background encoding -> origin + detect images encoded in separate thread
```

## Recognition Pipeline

```
1. Receive face images via gRPC (from detection_service)
2. Batch accumulation -> collect items for 1 second (max 16)
3. Batch embedding -> single gRPC call to face_embedding_service -> single GPU forward pass
4. ChromaDB vector search -> find closest matching identity
5. Cache result -> gRPC GetTrackingInfo (detection reads label from here)
6. Save tracking log -> JPEG bytes direct to disk + metadata to MongoDB
7. Override logic -> replace if better image quality OR lower distance
```

## Models

| Model | Architecture | Size | Performance |
|-------|-------------|------|-------------|
| **Person + Face Detection** | YOLO26n (custom trained, 2 classes) | ~15 MB | mAP50: 0.957, mAP50-95: 0.811 |
| **Face Embedding** | Facenet (InceptionResNetV1) | ~88 MB | 128-dim vectors, LFW accuracy: 97.4% |
| **Tracking** | ByteTrack | - | Kalman filter + IoU matching |

## Project Structure

```
camera_tracking/
├── detection_service/
│   ├── main.py                     # Startup, model init, gRPC client
│   ├── router/cameras.py           # Camera CRUD API endpoints
│   ├── service/
│   │   ├── config.py               # CameraStatus, BYTETrackerArgs, DetectionConfig
│   │   ├── detection_model.py      # SharedDetectionModel (singleton YOLO)
│   │   ├── association.py          # Face-to-person spatial matching
│   │   ├── camera.py               # DetectionService (stream, detect, track)
│   │   ├── track_cache.py          # Best-shot selection + TTL cache
│   │   ├── image_utils.py          # JPEG compression (Q=70/Q=80)
│   │   └── grpc_client.py          # RecognitionGrpcClient
│   ├── generated/                  # gRPC stubs
│   ├── weights/                    # YOLO26n model weights
│   └── ByteTrack/                  # Multi-object tracker
│
├── recognition_service/
│   ├── main.py                     # Startup + gRPC server launch
│   ├── grpc_server.py              # gRPC handlers + TrackingInfoCache
│   ├── service/
│   │   ├── recognition.py          # RecognitionService orchestrator
│   │   ├── pipeline.py             # Batch identification pipeline
│   │   ├── tracking_store.py       # MongoDB + disk image storage
│   │   ├── chromadb_ops.py         # ChromaDB CRUD wrapper
│   │   ├── embedding_client.py     # gRPC client for embedding service
│   │   └── database.py             # MongoDBManager
│   └── generated/                  # gRPC stubs
│
├── face_embedding_service/
│   ├── main.py                     # FastAPI + gRPC server
│   ├── grpc_server.py              # Embedding gRPC handler (single + batch)
│   ├── deepface/                   # Modified DeepFace (Facenet only)
│   │   ├── DeepFace.py             # Entry point
│   │   ├── basemodels/Facenet.py   # InceptionResNetV1 architecture
│   │   └── modules/                # representation, preprocessing, modeling
│   └── generated/                  # gRPC stubs
│
├── backend_service/
│   ├── main.py                     # FastAPI app
│   ├── config.py                   # Settings (MongoDB, detection URL)
│   ├── database.py                 # MongoDB singleton connection
│   ├── routers/                    # Web UI + API routes
│   ├── services/                   # Camera, user, detection business logic
│   ├── templates/                  # Jinja2 HTML templates
│   └── static/                     # CSS, JS assets
│
├── protos/
│   ├── recognition.proto           # IdentifyFace, GetTrackingInfo RPCs
│   ├── embedding.proto             # GetEmbedding, GetEmbeddingBatch RPCs
│   └── generate.sh                 # Proto code generation script
│
├── static_files/                   # Shared volume for detection images
├── docker-compose.yml
├── Dockerfile.camera_track_env     # Base GPU image (CUDA 12.1 + Python)
├── Dockerfile.detection            # GPU service
├── Dockerfile.recognition          # CPU service (python:3.10-slim)
├── Dockerfile.face_embedding       # GPU service
├── Dockerfile.backend              # CPU service (python:3.10-slim)
└── TRAIN_YOLO26N_PERSON_FACE.md    # Training guide for custom YOLO model
```

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- Docker Compose

### Setup

```bash
# Create Docker network
docker network create camera_track_network

# Build and start all services
docker-compose build --no-cache
docker-compose up
```

### Ports

| Service | URL |
|---------|-----|
| Web UI | http://localhost:80 |
| Detection API | http://localhost:5000 |
| Recognition gRPC | localhost:50051 |
| Embedding gRPC | localhost:50052 |
| MongoDB | localhost:27017 |
| ChromaDB | localhost:8000 |

## Communication Protocols

### gRPC (realtime AI pipeline)

- **detection -> recognition** (port 50051): `IdentifyFace` (fire-and-forget), `GetTrackingInfo` (cached label)
- **recognition -> embedding** (port 50052): `GetEmbedding` (single), `GetEmbeddingBatch` (batch)

### HTTP (admin/management)

- **backend -> detection** (port 5000): Camera CRUD, start/stop/stream
- **backend -> MongoDB** (port 27017): Direct pymongo connection
- **backend -> static_files**: Serve detection images via shared Docker volume

## Key Design Decisions

- **Unified YOLO model**: Single YOLO26n detects both person and face in one forward pass
- **gRPC over HTTP**: Binary protobuf for realtime image data, reducing serialization overhead and enabling fire-and-forget
- **Batch inference**: Accumulate detections for 1 second, process as single GPU batch (up to 16 items)
- **Best-shot selection**: Only send highest quality face per track, with TTL-based retry (3s unknown, 5min identified)
- **Direct MongoDB**: Services connect to MongoDB directly instead of routing through a database microservice
- **No base64**: Detection images saved as raw JPEG bytes to disk, no encode/decode overhead
- **CPU recognition**: Recognition service runs on CPU (no AI model loaded), only orchestrates pipeline via gRPC
- **Shared volume**: Static files stored on shared Docker volume (future: MinIO/S3 for multi-node)

## TODO

- [ ] User face upload flow: backend -> detection (face validation) -> recognition (embedding) -> ChromaDB
- [ ] Migrate static file storage to MinIO/S3 for multi-node scaling
- [ ] Separate user registration to dedicated business logic service
- [ ] Dashboard with real MongoDB data (currently mock)
