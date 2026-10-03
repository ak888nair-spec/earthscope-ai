from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import os

import cv2
import numpy as np


# ============================================================
# EARTHSCOPE
# LIGHTWEIGHT RENDER-FREE ANALYSIS ENGINE
#
# This version intentionally does NOT load CLIP/CLIPSeg.
# Render's free 512 MB service cannot reliably host those
# models. This backend uses CPU OpenCV/Pillow analysis so the
# service can start within the free memory limit.
# ============================================================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = {
    "cloud": {
        "name": "Cloud formation",
        "description":
            "A visible concentration of atmospheric clouds or cloud cover.",
        "type": "region",
    },
    "ice": {
        "name": "Ice / snow",
        "description":
            "A surface region visually consistent with snow, glacier ice, or a polar ice sheet.",
        "type": "region",
    },
    "water": {
        "name": "Open water",
        "description":
            "A surface region visually consistent with ocean, sea, lake, or other open water.",
        "type": "region",
    },
    "forest": {
        "name": "Forest / vegetation",
        "description":
            "A surface region containing visually dense vegetation or forest cover.",
        "type": "region",
    },
    "urban": {
        "name": "Urban development",
        "description":
            "A surface region visually consistent with cities, buildings, roads, or dense human development.",
        "type": "region",
    },
    "desert": {
        "name": "Desert / dry terrain",
        "description":
            "A broad surface region visually consistent with dry, sandy, or barren terrain.",
        "type": "region",
    },
    "mountains": {
        "name": "Mountain terrain",
        "description":
            "A surface region containing rugged terrain or mountainous relief.",
        "type": "region",
    },
    "crater": {
        "name": "Impact crater",
        "description":
            "A circular or roughly circular depression that may be visually consistent with an impact crater.",
        "type": "region",
    },
    "fire": {
        "name": "Fire / smoke plume",
        "description":
            "A region that may be visually consistent with active fire or a smoke plume.",
        "type": "phenomenon",
    },
    "storm": {
        "name": "Storm system",
        "description":
            "A large organized atmospheric cloud structure that may be consistent with a storm system.",
        "type": "phenomenon",
    },
}


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def image_to_data_url(image, quality=82):
    """Return a browser-readable JPEG data URL."""
    image = image.convert("RGB").copy()

    # Keep response size reasonable.
    max_side = 1400
    w, h = image.size

    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def resize_for_analysis(image, max_side=900):
    """Reduce analysis resolution to keep CPU/RAM usage low."""
    w, h = image.size

    if max(w, h) <= max_side:
        return image

    scale = max_side / max(w, h)

    return image.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )


def clean_mask(mask, kernel_size=5):
    mask = (mask > 0).astype(np.uint8)

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask.astype(np.float32)


def mask_area_percent(mask):
    if mask is None or mask.size == 0:
        return 0.0

    return float(
        np.mean(mask > 0.5) * 100.0
    )


def mask_overlap(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0

    a = mask_a > 0.5
    b = mask_b > 0.5

    a_area = int(a.sum())
    b_area = int(b.sum())

    if a_area == 0 or b_area == 0:
        return 0.0

    intersection = int(np.logical_and(a, b).sum())

    return float(
        intersection / min(a_area, b_area)
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def check_image_quality(image):
    width, height = image.size

    if min(width, height) < 160:
        return {
            "valid": False,
            "reason": "low_resolution",
            "message":
                "The image resolution is too low for reliable visual analysis.",
        }

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    contrast = float(gray.std())

    if contrast < 8:
        return {
            "valid": False,
            "reason": "low_information",
            "message":
                "The image contains too little visual contrast or information.",
        }

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    sharpness = float(laplacian.var())

    if sharpness < 20:
        return {
            "valid": False,
            "reason": "blurry",
            "message":
                "The observation is too blurry for reliable visual analysis.",
        }

    return {"valid": True}


# ============================================================
# LIGHTWEIGHT VISUAL ANALYSIS
# ============================================================

def build_visual_features(image):
    """
    Create reusable color/texture measurements.

    No large neural network is loaded here. This is deliberately
    lightweight so the backend can run on Render's free 512 MB
    service.
    """

    image = resize_for_analysis(image, 900)

    rgb = np.asarray(image.convert("RGB"))

    # OpenCV uses BGR.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Color masks.
    blue = (
        ((h >= 85) & (h <= 135) & (s > 55) & (v > 45))
        |
        ((h >= 95) & (h <= 135) & (s > 35) & (v > 70))
    )

    green = (
        (h >= 30)
        & (h <= 95)
        & (s > 45)
        & (v > 35)
    )

    tan = (
        (h >= 8)
        & (h <= 30)
        & (s > 35)
        & (v > 65)
    )

    bright_white = (
        (s < 55)
        & (v > 175)
    )

    red = (
        ((h < 12) | (h > 165))
        & (s > 100)
        & (v > 120)
    )

    orange = (
        (h >= 8)
        & (h <= 28)
        & (s > 110)
        & (v > 130)
    )

    dark_blue = (
        (h >= 85)
        & (h <= 140)
        & (s > 30)
        & (v < 150)
    )

    gray_low_sat = (
        (s < 60)
        & (v > 45)
        & (v < 205)
    )

    # Texture / edges.
    edges = cv2.Canny(
        gray,
        60,
        150,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    # Local standard deviation as a simple texture estimate.
    blur = cv2.GaussianBlur(
        gray.astype(np.float32),
        (0, 0),
        5,
    )

    texture = np.abs(
        gray.astype(np.float32) - blur
    )

    texture_score = float(
        np.mean(texture > 12)
    )

    return {
        "image": image,
        "rgb": rgb,
        "bgr": bgr,
        "hsv": hsv,
        "gray": gray,
        "blue": blue,
        "green": green,
        "tan": tan,
        "bright_white": bright_white,
        "red": red,
        "orange": orange,
        "dark_blue": dark_blue,
        "gray_low_sat": gray_low_sat,
        "edges": edges,
        "edge_density": edge_density,
        "texture_score": texture_score,
    }


def region_mask(mask):
    return clean_mask(
        mask.astype(np.uint8),
        kernel_size=5,
    )


# ============================================================
# FEATURE DETECTORS
# ============================================================

def detect_water(f):
    blue = f["blue"]
    dark_blue = f["dark_blue"]

    blue_ratio = float(np.mean(blue))
    dark_ratio = float(np.mean(dark_blue))

    score = min(
        1.0,
        blue_ratio * 1.35
        + dark_ratio * 0.65
    )

    if score < 0.12:
        return None

    mask = region_mask(
        blue | dark_blue
    )

    area = mask_area_percent(mask)

    if area < 2.0:
        return None

    confidence = min(
        96.0,
        52.0 + score * 38.0,
    )

    return {
        "id": "water",
        "name": FEATURES["water"]["name"],
        "description": FEATURES["water"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_forest(f):
    green = f["green"]

    green_ratio = float(np.mean(green))

    # Vegetation needs enough green coverage and some texture.
    if green_ratio < 0.10:
        return None

    if f["texture_score"] < 0.035:
        return None

    score = min(
        1.0,
        green_ratio * 1.8
        + f["texture_score"] * 0.8,
    )

    mask = region_mask(green)
    area = mask_area_percent(mask)

    if area < 3.0:
        return None

    confidence = min(
        95.0,
        48.0 + score * 42.0,
    )

    return {
        "id": "forest",
        "name": FEATURES["forest"]["name"],
        "description": FEATURES["forest"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_desert(f):
    tan = f["tan"]
    tan_ratio = float(np.mean(tan))

    if tan_ratio < 0.12:
        return None

    # Very green/blue images are unlikely to be dominated by dry terrain.
    green_ratio = float(np.mean(f["green"]))
    blue_ratio = float(np.mean(f["blue"]))

    if green_ratio > tan_ratio * 1.25:
        return None

    if blue_ratio > tan_ratio * 1.4:
        return None

    score = min(
        1.0,
        tan_ratio * 1.8
        + f["texture_score"] * 0.6,
    )

    mask = region_mask(tan)
    area = mask_area_percent(mask)

    if area < 4.0:
        return None

    confidence = min(
        94.0,
        47.0 + score * 42.0,
    )

    return {
        "id": "desert",
        "name": FEATURES["desert"]["name"],
        "description": FEATURES["desert"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_ice(f):
    white = f["bright_white"]
    blue = f["blue"]

    white_ratio = float(np.mean(white))
    blue_ratio = float(np.mean(blue))

    # Ice/snow is bright and relatively low-saturation.
    if white_ratio < 0.12:
        return None

    score = min(
        1.0,
        white_ratio * 1.55
        + blue_ratio * 0.25,
    )

    mask = region_mask(white)

    area = mask_area_percent(mask)

    if area < 4.0:
        return None

    # Avoid calling a tiny bright patch a continent-scale ice sheet.
    confidence = min(
        94.0,
        50.0 + score * 40.0,
    )

    return {
        "id": "ice",
        "name": FEATURES["ice"]["name"],
        "description": FEATURES["ice"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_clouds(f, ice_detection=None):
    white = f["bright_white"]
    texture = f["texture_score"]

    # Clouds tend to be bright but less uniformly solid than snow/ice.
    if float(np.mean(white)) < 0.10:
        return None

    # A modest amount of local texture supports cloud-like structure.
    if texture < 0.025:
        return None

    cloud_mask = region_mask(white)
    area = mask_area_percent(cloud_mask)

    if area < 2.0:
        return None

    # Prevent a huge bright ice sheet from also becoming a cloud.
    if ice_detection is not None:
        overlap = mask_overlap(
            cloud_mask,
            ice_detection["mask"],
        )

        if (
            overlap > 0.55
            and
            ice_detection["confidence"] >= 65
            and
            area > 15
        ):
            return None

    cloud_score = min(
        1.0,
        float(np.mean(white)) * 1.7
        + texture * 0.8,
    )

    confidence = min(
        92.0,
        45.0 + cloud_score * 43.0,
    )

    return {
        "id": "cloud",
        "name": FEATURES["cloud"]["name"],
        "description": FEATURES["cloud"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": cloud_mask,
        "type": "region",
    }


def detect_fire(f):
    red = f["red"]
    orange = f["orange"]

    hot = red | orange

    hot_ratio = float(np.mean(hot))

    if hot_ratio < 0.008:
        return None

    # Fire-like colors should be locally concentrated rather than
    # covering most of the image.
    if hot_ratio > 0.35:
        return None

    mask = region_mask(hot)
    area = mask_area_percent(mask)

    if area < 0.2:
        return None

    score = min(
        1.0,
        hot_ratio * 3.0
        + f["texture_score"] * 0.4,
    )

    confidence = min(
        88.0,
        45.0 + score * 40.0,
    )

    return {
        "id": "fire",
        "name": FEATURES["fire"]["name"],
        "description": FEATURES["fire"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "phenomenon",
    }


def detect_craters(f):
    gray = f["gray"]

    # Hough circles are used only as a visual cue. They are not proof
    # that a detected circle is an impact crater.
    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2,
    )

    min_dim = min(gray.shape)

    min_radius = max(
        8,
        int(min_dim * 0.025),
    )

    max_radius = max(
        min_radius + 5,
        int(min_dim * 0.22),
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 12),
        param1=90,
        param2=34,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(
        circles[0]
    ).astype(int)

    if len(circles) == 0:
        return None

    # Strong edge/texture support helps avoid treating random circles
    # in ordinary photographs as craters.
    if f["edge_density"] < 0.025:
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)

    useful = 0

    for x, y, r in circles[:8]:
        if r <= 0:
            continue

        cv2.circle(
            mask,
            (int(x), int(y)),
            int(r),
            1,
            -1,
        )
        useful += 1

    if useful == 0:
        return None

    mask = region_mask(mask)
    area = mask_area_percent(mask)

    confidence = min(
        87.0,
        48.0
        + useful * 5.0
        + min(15.0, f["edge_density"] * 100.0),
    )

    return {
        "id": "crater",
        "name": FEATURES["crater"]["name"],
        "description": FEATURES["crater"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_mountains(f):
    edges = f["edges"]

    edge_density = f["edge_density"]

    if edge_density < 0.055:
        return None

    # Long edges / ridges can support rugged terrain.
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=max(25, min(f["gray"].shape) // 12),
        maxLineGap=12,
    )

    line_count = 0 if lines is None else len(lines)

    if line_count < 5:
        return None

    score = min(
        1.0,
        edge_density * 5.0
        + line_count / 250.0,
    )

    # Use an edge-derived region for visualization.
    mask = (edges > 0).astype(np.float32)

    # Dilate so the overlay is readable.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(np.float32)

    area = mask_area_percent(mask)

    if area > 45:
        return None

    confidence = min(
        88.0,
        43.0 + score * 43.0,
    )

    return {
        "id": "mountains",
        "name": FEATURES["mountains"]["name"],
        "description": FEATURES["mountains"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_urban(f):
    edge_density = f["edge_density"]
    gray_ratio = float(np.mean(f["gray_low_sat"]))

    # Urban areas often have dense edges and neutral/gray pixels.
    if edge_density < 0.075:
        return None

    if gray_ratio < 0.22:
        return None

    score = min(
        1.0,
        edge_density * 4.0
        + gray_ratio * 0.7,
    )

    edge_mask = (f["edges"] > 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.dilate(
        edge_mask,
        kernel,
        iterations=1,
    ).astype(np.float32)

    area = mask_area_percent(mask)

    if area < 1.0 or area > 50:
        return None

    confidence = min(
        86.0,
        40.0 + score * 45.0,
    )

    return {
        "id": "urban",
        "name": FEATURES["urban"]["name"],
        "description": FEATURES["urban"]["description"],
        "confidence": confidence,
        "area_percent": round(area, 1),
        "mask": mask,
        "type": "region",
    }


def detect_storm(f, cloud_detection=None):
    if cloud_detection is None:
        return None

    cloud_area = cloud_detection["area_percent"]

    # A storm needs substantial organized cloud-like coverage.
    if cloud_area < 12:
        return None

    gray = f["gray"]

    # Look for a broad curved/circular structure.
    blurred = cv2.GaussianBlur(
        gray,
        (15, 15),
        0,
    )

    edges = cv2.Canny(
        blurred,
        35,
        100,
    )

    circular_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=max(30, min(gray.shape) // 10),
        maxLineGap=20,
    )

    line_count = 0 if circular_lines is None else len(circular_lines)

    if line_count < 3:
        return None

    storm_strength = min(
        1.0,
        cloud_area / 70.0
        + line_count / 180.0,
    )

    if storm_strength < 0.32:
        return None

    confidence = min(
        90.0,
        50.0 + storm_strength * 37.0,
    )

    return {
        "id": "storm",
        "name": FEATURES["storm"]["name"],
        "description": FEATURES["storm"]["description"],
        "confidence": confidence,
        "area_percent": None,
        "mask": None,
        "type": "phenomenon",
    }


# ============================================================
# EARTH / SPACE RELEVANCE
# ============================================================

def check_relevance(f):
    """
    Lightweight relevance estimate.

    This is intentionally conservative in wording: it estimates
    whether the image has visual characteristics associated with
    Earth/planetary observations. It does not identify the exact
    satellite source or location.
    """

    natural_ratio = max(
        float(np.mean(f["blue"])),
        float(np.mean(f["green"])),
        float(np.mean(f["tan"])),
        float(np.mean(f["bright_white"])),
    )

    structured_ratio = min(
        1.0,
        f["edge_density"] * 4.0
        + f["texture_score"] * 0.8,
    )

    score = (
        natural_ratio * 0.65
        +
        structured_ratio * 0.35
    )

    relevance = int(
        round(
            np.clip(
                25.0 + score * 75.0,
                0,
                100,
            )
        )
    )

    return relevance


# ============================================================
# ALL FEATURE DETECTION
# ============================================================

def detect_features(f):
    detections = []

    ice = detect_ice(f)
    if ice:
        detections.append(ice)

    cloud = detect_clouds(
        f,
        ice,
    )
    if cloud:
        detections.append(cloud)

    water = detect_water(f)
    if water:
        detections.append(water)

    crater = detect_craters(f)
    if crater:
        detections.append(crater)

    storm = detect_storm(
        f,
        cloud,
    )
    if storm:
        detections.append(storm)

    forest = detect_forest(f)
    if forest:
        detections.append(forest)

    desert = detect_desert(f)
    if desert:
        detections.append(desert)

    mountains = detect_mountains(f)
    if mountains:
        detections.append(mountains)

    urban = detect_urban(f)
    if urban:
        detections.append(urban)

    fire = detect_fire(f)
    if fire:
        detections.append(fire)

    # Reduce obvious conflicts.
    if water and ice:
        if water["confidence"] > ice["confidence"] + 18:
            detections = [
                d for d in detections
                if d["id"] != "ice"
            ]

    # Keep only strongest detections.
    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True,
    )

    return detections[:8]


# ============================================================
# ANNOTATION
# ============================================================

COLORS = {
    "ice": (80, 190, 255),
    "cloud": (80, 255, 210),
    "water": (50, 130, 255),
    "forest": (60, 255, 130),
    "urban": (255, 190, 50),
    "desert": (255, 170, 70),
    "mountains": (180, 130, 255),
    "crater": (255, 150, 60),
    "fire": (255, 80, 40),
    "storm": (255, 80, 180),
}


def overlay_features(image, detections):
    base = np.asarray(
        image.convert("RGB")
    ).copy()

    overlay = base.copy()

    for detection in detections:
        mask = detection.get("mask")

        if mask is None:
            continue

        binary = (
            mask > 0.5
        ).astype(np.uint8)

        color = COLORS.get(
            detection["id"],
            (80, 220, 255),
        )

        # Soft transparent fill.
        color_array = np.zeros_like(base)
        color_array[:, :] = color

        alpha = (
            binary[..., None].astype(np.float32)
            * 0.20
        )

        overlay = (
            overlay * (1.0 - alpha)
            +
            color_array * alpha
        ).astype(np.uint8)

        # Contours.
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = [
            c for c in contours
            if cv2.contourArea(c) > 80
        ]

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            2,
        )

    return Image.fromarray(overlay)


# ============================================================
# EXPLANATION
# ============================================================

def build_explanation(detections):
    if not detections:
        return (
            "The image passed the basic quality and relevance "
            "checks, but the visual evidence was not strong "
            "enough to confidently identify a supported feature."
        )

    names = {
        detection["name"]
        for detection in detections
    }

    statements = []

    if "Storm system" in names:
        statements.append(
            "A broad organized atmospheric structure is visible."
        )

    if "Cloud formation" in names:
        statements.append(
            "Bright atmospheric cloud-like regions are visible."
        )

    if "Ice / snow" in names:
        statements.append(
            "A bright low-saturation surface region is visually consistent with snow or ice."
        )

    if "Open water" in names:
        statements.append(
            "A broad blue or dark-blue surface region is visually consistent with open water."
        )

    if "Impact crater" in names:
        statements.append(
            "Circular surface structures were detected and may be consistent with impact craters."
        )

    if "Forest / vegetation" in names:
        statements.append(
            "Green textured surface regions are visible and may be consistent with vegetation."
        )

    if "Urban development" in names:
        statements.append(
            "Dense edge and built-environment-like structure is visible."
        )

    if "Desert / dry terrain" in names:
        statements.append(
            "Dry or sandy-looking terrain is visible."
        )

    if "Mountain terrain" in names:
        statements.append(
            "Rugged, high-edge-density terrain is visible."
        )

    if "Fire / smoke plume" in names:
        statements.append(
            "A concentrated hot-color region may be consistent with fire, "
            "but dedicated thermal/fire data would be needed for confirmation."
        )

    if not statements:
        statements.append(
            "The image contains visually distinctive Earth-surface or atmospheric structures."
        )

    explanation = " ".join(statements)

    explanation += (
        " EarthScope separates visible evidence from interpretation; "
        "this lightweight visual analysis cannot establish precise "
        "meteorological measurements such as wind speed, pressure, "
        "temperature, storm intensity, or an official storm category."
    )

    return explanation


# ============================================================
# API
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "EarthScope analysis engine",
        "engine": "lightweight-cpu",
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({
                "status": "invalid",
                "message": "No image was supplied.",
            }), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({
                "status": "invalid",
                "message": "No image was selected.",
            }), 400

        raw = file.read()

        if not raw:
            return jsonify({
                "status": "invalid",
                "message": "The uploaded image is empty.",
            }), 400

        try:
            image = Image.open(
                io.BytesIO(raw)
            ).convert("RGB")
        except Exception:
            return jsonify({
                "status": "invalid",
                "message": "The uploaded file is not a valid image.",
            }), 400

        quality = check_image_quality(image)

        if not quality["valid"]:
            return jsonify({
                "status": "unsuitable",
                "reason": quality["reason"],
                "message": quality["message"],
                "original_image": image_to_data_url(image),
            })

        features = build_visual_features(image)

        relevance = check_relevance(features)

        print(
            f"[VISION] Relevance estimate: {relevance}%"
        )

        # Keep this gate relatively permissive because this backend
        # intentionally avoids a large neural relevance model.
        if relevance < 28:
            return jsonify({
                "status": "unsuitable",
                "reason": "unrelated",
                "message":
                    "The uploaded image does not contain enough visual evidence of an Earth, planetary, or space observation.",
                "relevance": relevance,
                "original_image": image_to_data_url(image),
            })

        detections = detect_features(features)

        public_detections = []

        for detection in detections:
            public_detections.append({
                "id": detection["id"],
                "name": detection["name"],
                "description": detection["description"],
                "confidence": round(
                    float(detection["confidence"]),
                    1,
                ),
                "area_percent": detection["area_percent"],
                "type": detection["type"],
            })

        annotated = overlay_features(
            features["image"],
            detections,
        )

        if not detections:
            return jsonify({
                "status": "no_features",
                "message":
                    "The image is relevant, but no supported feature reached the confidence threshold.",
                "relevance": relevance,
                "detections": [],
                "explanation": build_explanation([]),
                "original_image": image_to_data_url(image),
                "annotated_image": image_to_data_url(annotated),
                "model_note":
                    "The lightweight CPU engine found an Earth/space-like observation but did not obtain enough visual evidence for a supported feature.",
            })

        return jsonify({
            "status": "success",
            "message": "Observation successfully analyzed.",
            "relevance": relevance,
            "detections": public_detections,
            "explanation": build_explanation(detections),
            "original_image": image_to_data_url(image),
            "annotated_image": image_to_data_url(annotated),
            "model_note":
                "Detections are lightweight visual estimates based on image color, texture, edges, and shape. They are not scientific measurements or official meteorological classifications.",
        })

    except Exception as error:
        print("=" * 60)
        print("EARTHSCOPE SERVER ERROR")
        print("=" * 60)
        print(repr(error))
        print("=" * 60)

        return jsonify({
            "status": "error",
            "message":
                "The observation engine encountered a server-side error.",
        }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
