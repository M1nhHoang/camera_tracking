📦 Root Directory
│
├── 📂 camera_inference_service/      # Service xử lý camera và human detection
│   ├── 📜 main.py                   # FastAPI app chính
│   ├── 📜 service.py                # Camera inference service logic
│   ├── 📜 requirements.txt
│   └── 📂 weights/
│       └── 📜 yolo8n_human_detect.pt
│
├── 📂 database_service/             # Service quản lý database
│   ├── 📜 main.py                   # FastAPI app chính
│   ├── 📜 database.py               # MongoDB connection manager
│   ├── 📜 requirements.txt
│   ├── 📂 service/
│   │   ├── 📜 users.py
│   │   ├── 📜 cameras.py
│   │   └── 📜 detected.py
│   └── 📂 router/
│       ├── 📜 users.py
│       ├── 📜 cameras.py
│       └── 📜 detected.py
│
├── 📂 face_embedding_service/       # Service tạo face embeddings
│   ├── 📜 main.py
│   ├── 📜 requirements.txt
│   └── 📜 DeepFace.py
│
├── 📂 face_identify_service/        # Service nhận dạng khuôn mặt
│   ├── 📜 main.py
│   ├── 📜 service.py
│   ├── 📜 requirements.txt
│   └── 📂 weights/
│       └── 📜 yolo8n_face_detect.pt
│
├── 📂 gateway_service/              # API Gateway service
│   ├── 📜 main.py
│   ├── 📜 config.py
│   ├── 📜 requirements.txt
│   ├── 📂 services/
│   │   ├── 📜 camera_service.py
│   │   ├── 📜 user_service.py
│   │   └── 📜 detect_service.py
│   ├── 📂 routers/
│   │   ├── 📜 web_router.py
│   │   ├── 📜 user_router.py
│   │   ├── 📜 detect_router.py
│   │   └── 📜 camera_router.py
│   ├── 📂 static/
│   │   ├── 📂 css/
│   │   │   └── 📜 style.css
│   │   └── 📂 js/
│   │       └── 📜 main.js
│   └── 📂 templates/
│       ├── 📜 base.html
│       ├── 📜 cameras.html
│       ├── 📜 detect.html
│       └── 📜 users.html
│
├── 📜 docker-compose.yml           # Docker compose configuration
├── 📜 Dockerfile.camera_inference
├── 📜 Dockerfile.camera_track_env
├── 📜 Dockerfile.database
├── 📜 Dockerfile.face_embedding
├── 📜 Dockerfile.face_identify
└── 📜 Dockerfile.gateway