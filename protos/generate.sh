#!/bin/bash
# Generate gRPC stubs from proto files
# Run from project root: bash protos/generate.sh

set -e

PROTO_DIR="protos"
CAMERA_OUT="camera_inference_service/generated"
RECOGNITION_OUT="recognition_service/generated"

# Create output directories
mkdir -p "$CAMERA_OUT" "$RECOGNITION_OUT"

# Generate Python gRPC stubs
python3 -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$CAMERA_OUT" \
  --grpc_python_out="$CAMERA_OUT" \
  "$PROTO_DIR/recognition.proto"

# Copy to recognition_service
cp "$CAMERA_OUT/recognition_pb2.py" "$RECOGNITION_OUT/"
cp "$CAMERA_OUT/recognition_pb2_grpc.py" "$RECOGNITION_OUT/"

# Create __init__.py files
touch "$CAMERA_OUT/__init__.py"
touch "$RECOGNITION_OUT/__init__.py"

echo "Proto generation complete."
echo "  -> $CAMERA_OUT/"
echo "  -> $RECOGNITION_OUT/"
