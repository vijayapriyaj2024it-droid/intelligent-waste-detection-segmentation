import cv2
import numpy as np


# Maximum allowed image size: 25 megapixels
MAX_IMAGE_PIXELS = 25_000_000


def preprocess_image(image_bytes):
    """
    Decode and validate an uploaded image.

    Input:
        image_bytes: Raw bytes from the uploaded image.

    Output:
        image: Image represented as a NumPy array in BGR format.
    """

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    # Convert the image bytes into a NumPy array.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image using OpenCV.
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "The uploaded file is not a valid image."
        )

    height, width = image.shape[:2]

    # Prevent excessively large images from consuming too much memory.
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Image is too large. Please upload a smaller image."
        )

    # Return the original-resolution image.
    # YOLO performs its own model-specific resizing and preprocessing.
    return image