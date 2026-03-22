# Training Custom YOLO26n for Person + Face Detection

## Problem Statement

Our camera tracking system currently runs **two separate YOLO models** sequentially:

1. **YOLO8n for person detection** (`detection_service`) — detects full body bounding boxes
2. **YOLO8n for face detection** (`recognition_service`) — detects faces within cropped person regions

This causes:
- **Double inference cost** — two forward passes per frame, heavy on GPU/CPU
- **Network overhead** — person crops sent over HTTP between services
- **Latency** — sequential pipeline (detect person → crop → send → detect face)
- **Cannot scale efficiently** — each model loads separately into GPU memory

### Why not just use face detection alone?

- **Tracking requires person bounding boxes** — faces are not always visible (person turns away, looks down, is occluded). Full body bounding boxes are larger and more stable for ByteTrack multi-object tracking.
- **Security logging needs full body crops** — origin image, person crop, and face crop are all required for investigation purposes.

### Solution

Train a **single unified YOLO26n model** that detects **both `person` (class 0) and `face` (class 1)** in **one forward pass**. This gives us:

- 1 inference instead of 2
- Both person bbox (for tracking) and face bbox (for embedding/identification)
- Spatial association: face bbox inside person bbox → link face to tracked person
- ~5MB model, ~39ms on CPU, ~1.7ms on GPU (TensorRT)

---

## Model: YOLO26n

| Spec | Value |
|------|-------|
| Architecture | YOLO26 nano (Ultralytics) |
| Parameters | 2.4M |
| FLOPs | 5.4B |
| Model size | ~5 MB |
| CPU ONNX speed | ~39 ms |
| GPU TensorRT speed | ~1.7 ms |
| NMS | Native end-to-end (NMS-free) |
| Package | `ultralytics>=8.4.14` |

---

## Dataset Preparation

### Required: 2-class dataset

```yaml
# data.yaml
path: /workspace/dataset
train: images/train
val: images/val

nc: 2
names:
  0: person
  1: face
```

### Directory structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_00001.jpg
│   │   └── ...
│   └── val/
│       ├── img_10001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_00001.txt
    │   └── ...
    └── val/
        ├── img_10001.txt
        └── ...
```

### Label format (YOLO format)

Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```
All coordinates are **normalized (0.0 - 1.0)** relative to image dimensions.

Example `img_00001.txt`:
```
0 0.512 0.480 0.320 0.750   # person
1 0.520 0.180 0.080 0.095   # face (inside the person bbox above)
0 0.150 0.500 0.200 0.600   # another person
```

### Dataset sources (if building from scratch)

| Source | What it provides | Notes |
|--------|-----------------|-------|
| **COCO** (person class) | Person bbox annotations | Extract class 0 (person), relabel as class 0 |
| **WiderFace** | Face bbox annotations | Convert to YOLO format, label as class 1 |
| **CrowdHuman** | Person + head annotations | Can adapt head → face |

**Important: Cross-annotation problem**
- COCO images have person annotations but **no face annotations**
- WiderFace images have face annotations but **no person annotations**
- For best results, use a pre-trained detector to auto-annotate the missing class in each dataset before merging

---

## Training with Docker

### Step 1: Pull the official Ultralytics Docker image

```bash
docker pull ultralytics/ultralytics:latest
```

This image includes:
- PyTorch 2.10.0
- CUDA 12.8 + cuDNN 9
- Ultralytics package (with YOLO26 support)

### Step 2: Run the training container

```bash
docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --gpus all \
  -v /path/to/your/dataset:/workspace/dataset \
  -v /path/to/output:/workspace/runs \
  ultralytics/ultralytics:latest
```

| Flag | Purpose |
|------|---------|
| `--ipc=host` | Shared memory for DataLoader workers |
| `--runtime=nvidia --gpus all` | Enable GPU access |
| `-v .../dataset:/workspace/dataset` | Mount your dataset into the container |
| `-v .../output:/workspace/runs` | Persist training outputs (weights, logs) |

### Step 3: Create data.yaml inside the container

```bash
cat > /workspace/data.yaml << 'EOF'
path: /workspace/dataset
train: images/train
val: images/val

nc: 2
names:
  0: person
  1: face
EOF
```

### Step 4: Start training

```bash
yolo detect train \
  model=yolo26n.pt \
  data=/workspace/data.yaml \
  epochs=300 \
  imgsz=640 \
  batch=16 \
  workers=8 \
  project=/workspace/runs \
  name=yolo26n_person_face
```

Or in Python:
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # loads pretrained COCO weights as starting point

results = model.train(
    data="/workspace/data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    workers=8,
    project="/workspace/runs",
    name="yolo26n_person_face",
)
```

### Key training parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `model` | `yolo26n.pt` | Start from COCO pretrained (transfer learning) |
| `epochs` | 300 | Standard for custom training |
| `imgsz` | 640 | Default YOLO input size |
| `batch` | 16-32 | Adjust based on GPU VRAM (16 for ~8GB, 32 for ~16GB) |
| `workers` | 8 | DataLoader workers |
| `patience` | 50 | Early stopping patience (default) |
| `optimizer` | MuSGD | YOLO26 default (hybrid SGD + Muon), no need to change |
| `lr0` | 0.01 | Default initial learning rate |
| `cos_lr` | True | Cosine learning rate scheduler |

### Training time estimate

| GPU | Estimated time (300 epochs) |
|-----|---------------------------|
| RTX 3090 / 4090 | 8-15 hours |
| RTX 3060 / 4060 | 15-24 hours |
| T4 (cloud) | 12-20 hours |

---

## Step 5: Validate the trained model

```bash
yolo detect val \
  model=/workspace/runs/yolo26n_person_face/weights/best.pt \
  data=/workspace/data.yaml
```

Check per-class mAP:
- `person` mAP@50 should be > 0.70
- `face` mAP@50 should be > 0.65

---

## Step 6: Export for deployment

### ONNX (CPU deployment)
```bash
yolo export model=/workspace/runs/yolo26n_person_face/weights/best.pt format=onnx
```

### TensorRT (GPU deployment)
```bash
yolo export model=/workspace/runs/yolo26n_person_face/weights/best.pt format=engine
```

### OpenVINO (Intel CPU optimization)
```bash
yolo export model=/workspace/runs/yolo26n_person_face/weights/best.pt format=openvino
```

---

## Step 7: Integrate into the camera tracking system

Copy the trained weights to:
```
detection_service/weights/yolo_person_face.pt
```

The `detection_service` is already configured to use a unified model:

```python
# detection_service/main.py
model_config = DetectionConfig(
    model_path="weights/yolo_person_face.pt",
    person_conf_threshold=0.5,
    face_conf_threshold=0.5,
    person_class_id=0,    # matches training class 0
    face_class_id=1,      # matches training class 1
)
```

No code changes needed — just drop in the weights file.

---

## Troubleshooting

### Low face detection accuracy
- Face bboxes are small relative to person bboxes → try `imgsz=1280` for higher resolution training
- Add more face-heavy augmentation: `mosaic=1.0`, `mixup=0.1`
- Check class balance: ensure roughly equal number of person and face annotations

### GPU out of memory
- Reduce `batch` size (16 → 8 → 4)
- Reduce `imgsz` (640 → 480)
- Use `amp=True` (automatic mixed precision, enabled by default)

### Training not converging
- Verify label files match image files (same names, `.txt` extension)
- Check normalized coordinates are within 0.0-1.0
- Visualize some labels: `yolo detect val model=yolo26n.pt data=data.yaml plots=True`

---

## References

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train/)
- [Ultralytics Docker Quickstart](https://docs.ultralytics.com/guides/docker-quickstart/)
- [Dataset Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)
- [YOLO26 Paper (arXiv)](https://arxiv.org/abs/2510.09653)
