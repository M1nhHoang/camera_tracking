import cv2
import base64
import numpy as np
from PIL import Image
import io
import os
import random


def genarate_id():
    char_list = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
    )
    detect_id = ""
    for _ in range(24):
        detect_id += random.choice(char_list)
    return detect_id


def convert_image_to_base64(image):
    # Check if image is a valid base64 string
    if isinstance(image, str):
        return image
    elif isinstance(image, np.ndarray):
        # If image is a numpy array, convert it to bytes
        img_byte_arr = cv2.imencode(".jpg", image)[1].tobytes()
    elif isinstance(image, bytes):
        # If image is bytes, process it normally
        img_data = cv2.imdecode(
            np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        img_byte_arr = cv2.imencode(".jpg", img_data)[1].tobytes()
    else:
        raise ValueError("Unsupported image format")

    # Convert bytes to base64 string
    image_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
    return f"{image_base64}"


def base64_to_image(image_data):
    if isinstance(image_data, np.ndarray):
        # If image is a numpy array (cv2)
        return image_data
    elif isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
        image_np = np.array(Image.open(io.BytesIO(image_bytes)))
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image_data, bytes):
        # If image is image_bytes
        image_np = np.array(Image.open(io.BytesIO(image_data)))
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unsupported image format")


def is_better_quality(new_img, old_img):
    new_img = base64_to_image(new_img)
    old_img = base64_to_image(old_img)

    # Convert images to grayscale and compare their sizes
    gray1 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    if gray1.shape != gray2.shape:
        gray1 = cv2.resize(gray1, gray2.shape[:2][::-1])
        gray2 = cv2.resize(gray2, gray1.shape[:2][::-1])

    laplacian_var1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
    laplacian_var2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

    return laplacian_var1 > laplacian_var2


def check_folder_exists(folder_name):
    folder_path = os.path.join("..", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def save_image_to_folder(image, folder_name, path=None):
    image = base64_to_image(image)
    folder_path = check_folder_exists(folder_name)

    if path:
        image_path = os.path.join(folder_path, path)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_name = genarate_id() + ".jpg"
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return os.path.basename(image_path) if path else image_name
