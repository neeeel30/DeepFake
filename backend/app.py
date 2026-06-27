"""
Flask backend for AI-vs-Real image detection.

Serves an EfficientNetV2S transfer-learned model that outputs the probability
that an uploaded image is AI-generated (FAKE). Preprocessing mirrors training
exactly: decode as RGB, resize to img_size, feed pixels in [0, 255] (the model
normalises internally).

Endpoints:
  GET  /health    readiness + model info
  POST /predict   fast verdict: {verdict, p_fake, threshold, label}
  POST /explain   verdict + Grad-CAM overlay (base64 PNG) of what drove it
"""
import base64
import json
import logging
import os

import numpy as np
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
GRADCAM_LAYER = "top_activation"       # last conv feature map in EfficientNetV2S
GRADCAM_ALPHA = 0.4                    # heatmap opacity in the overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
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
        "the TensorFlow version used for training against requirements.txt.", exc,
    )
    raise

with open(META_PATH) as f:
    META = json.load(f)

IMG_SIZE = int(META.get("img_size", 224))
THRESHOLD = float(META.get("decision_threshold", 0.5))
FAKE_LABEL = int(META.get("fake_label", 1))
log.info("Model loaded. img_size=%d  threshold=%.4f  fake_label=%d",
         IMG_SIZE, THRESHOLD, FAKE_LABEL)

# Grad-CAM gradient model: input -> (last conv feature map, pre-sigmoid logits).
# Built once and reused for every /explain call.
grad_model = tf.keras.Model(
    model.inputs,
    [model.get_layer(GRADCAM_LAYER).output, model.get_layer("logits").output],
)


def _jet(gray):
    """Map a [0,1] heatmap (H,W) to an RGB uint8 'jet' colormap (no matplotlib)."""
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def explain_image(batch):
    """One forward+backward pass -> (p_fake, overlay_png_bytes).

    Grad-CAM is taken w.r.t. the pre-sigmoid logit over the last conv feature map.
    """
    with tf.GradientTape() as tape:
        conv_out, logits = grad_model(batch, training=False)
        tape.watch(conv_out)
        target = logits[:, 0]
    grads = tape.gradient(target, conv_out)

    conv_out = tf.cast(conv_out[0], tf.float32)                  # (h, w, C)
    grads = tf.cast(grads[0], tf.float32)                        # (h, w, C)
    weights = tf.reduce_mean(grads, axis=(0, 1))                 # (C,)
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights, axis=-1)) # (h, w)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    cam = tf.image.resize(cam[..., None], [IMG_SIZE, IMG_SIZE])[..., 0].numpy()
    heat = _jet(cam)                                             # (H, W, 3) uint8
    base_img = tf.cast(batch[0], tf.uint8).numpy()              # what the model saw
    overlay = np.clip((1 - GRADCAM_ALPHA) * base_img + GRADCAM_ALPHA * heat,
                      0, 255).astype(np.uint8)
    png = tf.io.encode_png(tf.constant(overlay)).numpy()

    p1 = float(tf.sigmoid(target)[0])
    p_fake = p1 if FAKE_LABEL == 1 else 1.0 - p1
    return p_fake, png


# Warm up both paths so the first real request isn't slow.
_zeros = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model.predict(_zeros, verbose=0)
explain_image(_zeros)
log.info("Warmup complete (predict + explain). Ready to serve.")

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
    return tf.expand_dims(img, 0)


def _fields(p_fake):
    label = int(p_fake >= THRESHOLD)
    return {
        "verdict": "AI / FAKE" if label == 1 else "REAL",
        "p_fake": round(float(p_fake), 4),
        "threshold": round(THRESHOLD, 4),
        "label": label,
    }


def _read_batch():
    """Validate the request -> (batch, None) or (None, (payload, status))."""
    if "image" not in request.files:
        return None, ({"error": "No image file in request (expected form field 'image')."}, 400)
    img_file = request.files["image"]
    if img_file.filename == "":
        return None, ({"error": "Empty filename."}, 400)
    image_bytes = img_file.read()
    if not image_bytes:
        return None, ({"error": "Uploaded file is empty."}, 400)
    try:
        return preprocess(image_bytes), None
    except Exception as exc:
        log.warning("Failed to decode image: %s", exc)
        return None, ({"error": "Could not decode the uploaded file as an image."}, 400)


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
    batch, err = _read_batch()
    if err:
        return jsonify(err[0]), err[1]
    try:
        p1 = float(model.predict(batch, verbose=0).ravel()[0])
    except Exception:
        log.exception("Inference failed")
        return jsonify({"error": "Prediction failed."}), 500
    p_fake = p1 if FAKE_LABEL == 1 else 1.0 - p1
    fields = _fields(p_fake)
    log.info("predict: verdict=%s p_fake=%.4f", fields["verdict"], fields["p_fake"])
    return jsonify(fields)


@app.route("/explain", methods=["POST"])
def explain():
    batch, err = _read_batch()
    if err:
        return jsonify(err[0]), err[1]
    try:
        p_fake, png = explain_image(batch)
    except Exception:
        log.exception("Explain failed")
        return jsonify({"error": "Explanation failed."}), 500
    fields = _fields(p_fake)
    fields["heatmap_png"] = base64.b64encode(png).decode("ascii")
    log.info("explain: verdict=%s p_fake=%.4f", fields["verdict"], fields["p_fake"])
    return jsonify(fields)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 10 MB)."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
