from typing import Tuple

import numpy as np
import cv2

from deepface.commons import package_utils

tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image


def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image for Facenet model."""
    if normalization == "base":
        return img

    # Restore input to [0, 255] scale (was normalized to [0, 1] in resize_image)
    img *= 255

    if normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")

    return img


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to expected size of a ml model with adding black pixels."""
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # Make it 4-dimensional how ML models expect
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img
