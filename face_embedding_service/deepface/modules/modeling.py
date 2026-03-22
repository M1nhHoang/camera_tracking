from typing import Any

from deepface.basemodels import Facenet


def build_model(model_name: str) -> Any:
    """Build and cache a deepface model (singleton pattern)."""
    global model_obj

    models = {
        "Facenet": Facenet.FaceNet128dClient,
    }

    if "model_obj" not in globals():
        model_obj = {}

    if model_name not in model_obj:
        model = models.get(model_name)
        if model:
            model_obj[model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]
