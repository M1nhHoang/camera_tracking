import os
import warnings
import logging
from typing import Any, Dict, List, Union

# Must be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from deepface.commons import package_utils, folder_utils
from deepface.modules import representation, modeling

# Validate TF/Keras compatibility
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if package_utils.get_tf_major_version() == 2:
    tf.get_logger().setLevel(logging.ERROR)

# Create weights directory
folder_utils.initialize_folder()


def build_model(model_name: str) -> Any:
    """Build and cache a face recognition model."""
    return modeling.build_model(model_name=model_name)


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "Facenet",
    normalization: str = "Facenet",
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate face embedding from image.

    Args:
        img_path: numpy array (RGB/BGR), file path, or base64 string
        model_name: "Facenet" for 128-dim embeddings
        normalization: "Facenet" for Facenet-specific normalization

    Returns:
        List of dicts with "embedding", "facial_area", "face_confidence"
    """
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
    )
