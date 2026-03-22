from typing import Any, Dict, List, Union

import numpy as np

from deepface.commons import image_utils
from deepface.modules import modeling, preprocessing
from deepface.models.FacialRecognition import FacialRecognition


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "Facenet",
    normalization: str = "Facenet",
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """Generate embedding for a single image."""
    model: FacialRecognition = modeling.build_model(model_name)
    target_size = model.input_shape

    img, _ = image_utils.load_image(img_path)
    if len(img.shape) != 3:
        raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

    # RGB to BGR
    img = img[:, :, ::-1]

    # Resize and normalize
    img = preprocessing.resize_image(img=img, target_size=(target_size[1], target_size[0]))
    img = preprocessing.normalize_input(img=img, normalization=normalization)

    embedding = model.forward(img)

    return [{"embedding": embedding, "facial_area": {}, "face_confidence": 0}]


def represent_batch(
    images: List[np.ndarray],
    model_name: str = "Facenet",
    normalization: str = "Facenet",
) -> List[List[float]]:
    """
    Generate embeddings for a batch of images in a single forward pass.

    Args:
        images: list of numpy arrays (RGB/BGR)
        model_name: model to use
        normalization: normalization technique

    Returns:
        list of embedding vectors (list of floats)
    """
    if not images:
        return []

    model: FacialRecognition = modeling.build_model(model_name)
    target_size = model.input_shape

    # Preprocess all images
    processed = []
    for img in images:
        if len(img.shape) != 3:
            raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

        # RGB to BGR
        img = img[:, :, ::-1]

        # Resize and normalize → shape (1, H, W, 3)
        img = preprocessing.resize_image(img=img, target_size=(target_size[1], target_size[0]))
        img = preprocessing.normalize_input(img=img, normalization=normalization)
        processed.append(img[0])  # Remove batch dim → (H, W, 3)

    # Stack into batch tensor → (N, H, W, 3)
    batch = np.stack(processed, axis=0)

    # Single forward pass → (N, 128)
    embeddings = model.model(batch, training=False).numpy()

    return [row.tolist() for row in embeddings]
