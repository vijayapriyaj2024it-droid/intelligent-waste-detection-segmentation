import base64
import io
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image as PILImage
from ultralytics import YOLO

# ============================================================
# FLASK & APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# ============================================================
# MODEL SETUP (Clean pre-trained YOLOv8)
# ============================================================

MODEL_NAME = "yolov8n.pt"
print(f"Initializing EcoVision AI Backend with model: {MODEL_NAME}")

model = YOLO(MODEL_NAME)
print("Model loaded successfully!")

# ============================================================
# WASTE CATEGORIES & MAPPING
# ============================================================

WASTE_CATEGORIES = {
    "Plastic": "Place plastic waste in the appropriate plastic recycling stream.",
    "Metal": "Separate metal cans and containers for metal recycling.",
    "Glass": "Separate glass items carefully for glass recycling.",
    "Paper/Cardboard": "Keep paper and cardboard clean and dry for recycling.",
    "Food Waste": "Place food waste in the organic waste or composting stream.",
    "Foam": "Check local facilities for foam recycling options."
}

# Optimized COCO mapping with foam and cardboard proxy mappings
COCO_TO_WASTE_MAP = {
    # Foam (Styrofoam clamshell containers often map to 'toilet' contour shape)
    "toilet": "Foam",
    
    # Metal
    "vase": "Metal",        # Metallic drink cans
    "fork": "Metal",
    "knife": "Metal",
    "spoon": "Metal",
    
    # Plastic & General Containers
    "bottle": "Plastic",
    "cup": "Plastic",
    
    # Glass
    "wine glass": "Glass",
    "bowl": "Glass",
    
    # Paper/Cardboard
    "book": "Paper/Cardboard",
    "bench": "Paper/Cardboard",   # Cardboard stacks often map here
    
    # Food Waste
    "banana": "Food Waste",
    "apple": "Food Waste",
    "orange": "Food Waste",
    "broccoli": "Food Waste",
    "carrot": "Food Waste",
    "sandwich": "Food Waste",
    "hot dog": "Food Waste",
    "pizza": "Food Waste",
    "donut": "Food Waste",
    "cake": "Food Waste"
}

# ============================================================
# IMAGE UTILITIES
# ============================================================

def image_to_base64(image):
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not convert image to base64.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")


def read_uploaded_image(file):
    image_bytes = file.read()
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")
    try:
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError("Invalid image file format.") from e
    
    if image is None:
        raise ValueError("Could not decode image.")
    return image


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def analyze_waste(image):
    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        verbose=False
    )

    result = results[0]
    original = image.copy()
    overlay = image.copy()
    objects = []

    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            confidence = float(box.conf.item())
            class_id = int(box.cls.item())
            coco_name = result.names[class_id]

            # Map to waste category, defaulting to Plastic
            category = COCO_TO_WASTE_MAP.get(coco_name, "Plastic")

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Draw bounding box and label
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{category} ({coco_name}) {confidence * 100:.1f}%"
            cv2.putText(
                overlay, label, (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

            objects.append({
                "index": len(objects) + 1,
                "category": category,
                "confidence": round(confidence, 4),
                "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            })

    if len(objects) == 0:
        return {
            "success": True,
            "category": None,
            "confidence": 0,
            "total_items": 0,
            "objects": [],
            "recommendation": "No recognized items detected. Try uploading an image with a bottle, fruit, cup, or container.",
            "overlay_image": image_to_base64(original),
            "mask_image": image_to_base64(np.zeros_like(original))
        }

    final_overlay = cv2.addWeighted(original, 0.55, overlay, 0.45, 0)
    primary = max(objects, key=lambda x: x["confidence"])

    return {
        "success": True,
        "category": primary["category"],
        "confidence": primary["confidence"],
        "total_items": len(objects),
        "objects": objects,
        "recommendation": WASTE_CATEGORIES[primary["category"]],
        "overlay_image": image_to_base64(final_overlay),
        "mask_image": image_to_base64(np.zeros_like(original)),
        "message": "Waste analysis completed successfully."
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "model": MODEL_NAME})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Please upload an image."}), 400

        uploaded_file = request.files["image"]
        if uploaded_file.filename == "":
            return jsonify({"success": False, "error": "No image selected."}), 400

        image = read_uploaded_image(uploaded_file)
        result = analyze_waste(image)
        return jsonify(result), 200

    except ValueError as err:
        return jsonify({"success": False, "error": str(err)}), 400
    except Exception as err:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": str(err)}), 500


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":
    print("\n==========================================")
    print("      EcoVision AI Backend (Updated)")
    print("==========================================")
    print("API Endpoint: http://127.0.0.1:5000/predict")
    print("==========================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)