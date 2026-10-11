import base64
from collections import Counter

import cv2
import numpy as np


RECOMMENDATIONS = {
    "Plastic": (
        "Place recyclable plastic items in the appropriate "
        "plastic recycling stream where accepted."
    ),

    "Metal": (
        "Separate metal cans and containers for metal recycling."
    ),

    "Glass": (
        "Separate glass containers carefully for glass recycling."
    ),

    "Paper/Cardboard": (
        "Keep paper and cardboard clean and dry for recycling."
    ),

    "Food Waste": (
        "Place food waste in the organic waste or composting "
        "stream where available."
    ),

    "Foam": (
        "Check local waste facilities because foam recycling "
        "availability varies."
    )
}


# Colours are in OpenCV's BGR format.
CATEGORY_COLORS = {
    "Plastic": (255, 120, 0),
    "Metal": (0, 200, 255),
    "Glass": (255, 0, 180),
    "Paper/Cardboard": (0, 220, 100),
    "Food Waste": (0, 140, 255),
    "Foam": (200, 100, 255)
}


def image_to_data_url(image):
    """
    Convert an OpenCV image to a base64 PNG data URL.

    This allows the frontend to display the generated image.
    """

    success, encoded_image = cv2.imencode(
        ".png",
        image
    )

    if not success:
        raise RuntimeError(
            "Could not encode the generated image."
        )

    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    return f"data:image/png;base64,{image_base64}"


def create_segmentation_result(original_image, result):
    """
    Create the final visualization using actual model masks.

    original_image:
        Original OpenCV image.

    result:
        YOLO result for the uploaded image.
    """

    height, width = original_image.shape[:2]

    # Create an empty combined binary mask.
    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Start the overlay with the original image.
    overlay = original_image.copy()

    detected_objects = []

    # Retrieve predictions.
    boxes = result.boxes
    masks = result.masks

    # Handle images where the model detects nothing.
    if boxes is None or len(boxes) == 0:

        black_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        return {
            "success": True,
            "category": None,
            "confidence": None,
            "recommendation": (
                "No waste objects were detected. "
                "Try a clearer image with visible waste."
            ),
            "total_items": 0,
            "objects": [],
            "overlay_image": image_to_data_url(
                original_image
            ),
            "mask_image": image_to_data_url(
                black_mask
            ),
            "message": "No waste objects detected."
        }

    # A detection-only model cannot provide the required masks.
    if masks is None:
        raise RuntimeError(
            "The loaded model returned detections without "
            "segmentation masks. Use an instance-segmentation model."
        )

    # Retrieve individual predicted masks.
    predicted_masks = masks.data.cpu().numpy()

    if len(predicted_masks) != len(boxes):
        raise RuntimeError(
            "The number of segmentation masks does not match "
            "the number of detected objects."
        )

    for index, (box, predicted_mask) in enumerate(
        zip(boxes, predicted_masks)
    ):

        # Obtain class ID and confidence from actual inference.
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        category = result.names[class_id]

        if category not in RECOMMENDATIONS:
            raise RuntimeError(
                f"Unexpected model category: {category}"
            )

        # Convert the predicted mask to the original image size.
        resized_mask = cv2.resize(
            predicted_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert the mask into a binary image.
        binary_mask = resized_mask > 0

        # Calculate the actual number of masked pixels.
        area_pixels = int(
            np.count_nonzero(binary_mask)
        )

        # Add this object's mask to the combined mask.
        combined_mask[binary_mask] = 255

        # Apply the category's colour to the predicted region.
        colour = CATEGORY_COLORS[category]

        overlay[binary_mask] = colour

        # Obtain the predicted bounding box.
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # Draw the bounding box around the detected object.
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            colour,
            2
        )

        # Add category and confidence to the image.
        label = (
            f"{category} "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            overlay,
            label,
            (max(0, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA
        )

        detected_objects.append({
            "index": index + 1,
            "category": category,
            "confidence": round(confidence, 4),

            "bounding_box": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            },

            "area_pixels": area_pixels
        })

    # Blend the coloured segmentation with the original image.
    # The original object remains partially visible.
    final_overlay = cv2.addWeighted(
        original_image,
        0.55,
        overlay,
        0.45,
        0
    )

    # Select the highest-confidence detection as the main result.
    primary_object = max(
        detected_objects,
        key=lambda item: item["confidence"]
    )

    primary_category = primary_object["category"]

    # Count how many objects belong to each category.
    category_counts = Counter(
        obj["category"]
        for obj in detected_objects
    )

    # Return the main result and all object detections.
    return {
        "success": True,

        "category": primary_category,

        "confidence": primary_object["confidence"],

        "recommendation": RECOMMENDATIONS[
            primary_category
        ],

        "total_items": len(detected_objects),

        "category_counts": dict(category_counts),

        "objects": detected_objects,

        "overlay_image": image_to_data_url(
            final_overlay
        ),

        "mask_image": image_to_data_url(
            combined_mask
        ),

        "message": "Waste analysis completed successfully."
    }