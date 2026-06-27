"""
Flask backend for AI-vs-Real image detection.

Serves an EfficientNetV2S transfer-learned model that outputs the probability
that an uploaded image is AI-generated (FAKE). Preprocessing mirrors training
exactly: decode as RGB, resize to img_size, feed pixels in [0, 255] (the model
normalises internally).
"""
import json
import logging
import os

import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_effnetv2s.keras")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # reject uploads larger than 10 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("deepfake-api")

# ---------------------------------------------------------------------------
# Load model + metadata once at startup
# ---------------------------------------------------------------------------
log.info("Loading model from %s", MODEL_PATH)
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as exc:  # pragma: no cover - startup guard
    log.error(
        "Could not load model (%s). If this is a Keras version mismatch, check "
        "the TensorFlow version used for training against requirements.txt.",
        exc,
    )
    raise

with open(META_PATH) as f:
    META = json.load(f)

IMG_SIZE = int(META.get("img_size", 224))
THRESHOLD = float(META.get("decision_threshold", 0.5))
FAKE_LABEL = int(META.get("fake_label", 1))
log.info(
    "Model loaded. img_size=%d  threshold=%.4f  fake_label=%d",
    IMG_SIZE, THRESHOLD, FAKE_LABEL,
)

# Warm up the graph so the first real request isn't slow.
model.predict(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
log.info("Warmup inference complete. Ready to serve.")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)


def preprocess(image_bytes):
    """Decode raw bytes the same way training did: RGB, resized, pixels in [0, 255]."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # float32 in [0, 255]
    return tf.expand_dims(img, 0)  # [1, H, W, 3]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "img_size": IMG_SIZE,
        "threshold": round(THRESHOLD, 4),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request (expected form field 'image')."}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = img_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        batch = preprocess(image_bytes)
    except Exception as exc:
        log.warning("Failed to decode image: %s", exc)
        return jsonify({"error": "Could not decode the uploaded file as an image."}), 400

    try:
        p1 = float(model.predict(batch, verbose=0).ravel()[0])  # P(label == 1)
    except Exception:
        log.exception("Inference failed")
        return jsonify({"error": "Prediction failed."}), 500

    p_fake = p1 if FAKE_LABEL == 1 else 1.0 - p1
    label = int(p_fake >= THRESHOLD)
    verdict = "AI / FAKE" if label == 1 else "REAL"

    log.info("prediction: verdict=%s  p_fake=%.4f", verdict, p_fake)
    return jsonify({
        "verdict": verdict,
        "p_fake": round(p_fake, 4),
        "threshold": round(THRESHOLD, 4),
        "label": label,
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 10 MB)."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
