import os
from pathlib import Path

from ultralytics import YOLO


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Location of the trained model
MODEL_PATH = Path(
    os.getenv(
        "WASTE_MODEL_PATH",
        str(PROJECT_ROOT / "model" / "best.pt")
    )
)


# Six categories used in the project
WASTE_CATEGORIES = [
    "Plastic",
    "Metal",
    "Glass",
    "Paper/Cardboard",
    "Food Waste",
    "Foam"
]


# Store the loaded model so it is not loaded for every image
_model = None


def is_model_loaded():
    """Return True if the model is already loaded."""
    return _model is not None


def model_file_exists():
    """Check whether best.pt exists."""
    return MODEL_PATH.is_file()


def normalize_category_name(name):
    """Normalize category names for checking."""

    name = str(name).strip().lower()

    aliases = {
        "paper and cardboard": "paper/cardboard",
        "paper_cardboard": "paper/cardboard",
        "food_waste": "food waste"
    }

    return aliases.get(name, name)


def load_model():
    """
    Load the trained YOLO segmentation model.
    The model must contain exactly six waste categories.
    """

    global _model

    # If already loaded, reuse it
    if _model is not None:
        return _model

    # Check whether best.pt exists
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Please place best.pt inside the model folder."
        )

    # Load trained YOLO segmentation model
    candidate = YOLO(str(MODEL_PATH))

    # Get class names from the model
    names = candidate.names

    if isinstance(names, dict):
        class_names = [
            str(names[index])
            for index in sorted(names.keys())
        ]
    else:
        class_names = [
            str(name)
            for name in names
        ]

    # Expected categories
    expected = {
        normalize_category_name(name)
        for name in WASTE_CATEGORIES
    }

    # Categories found in model
    actual = {
        normalize_category_name(name)
        for name in class_names
    }

    # Make sure this is our six-class model
    if len(class_names) != 6 or actual != expected:
        raise RuntimeError(
            "The model must be trained with exactly these six "
            "categories: Plastic, Metal, Glass, Paper/Cardboard, "
            "Food Waste and Foam. "
            f"Model categories found: {class_names}"
        )

    # Save loaded model
    _model = candidate

    print("Waste segmentation model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Categories:", class_names)

    return _model


def predict_image(image):
    """
    Run waste detection and segmentation.

    Input:
        image: OpenCV image in BGR format.

    Output:
        YOLO prediction results.
    """

    model = load_model()

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.05,
        iou=0.45,
        verbose=False
    )

    return results