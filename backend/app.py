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
# IMPROVED LIGHTWEIGHT CPU ANALYSIS ENGINE
#
# Designed for Render Free / low-memory CPU hosting.
# No CLIP/CLIPSeg is loaded.
#
# Main improvement over the previous version:
# - analyzes local regions instead of whole-image color ratios
# - separates sky/cloud/water/land using spatial context
# - uses multiple independent visual cues before reporting a feature
# - suppresses common false positives
# - reports conservative confidence rather than arbitrary high scores
# ============================================================

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

FEATURES = {
    "cloud": {
        "name": "Cloud formation",
        "description": "A visible concentration of atmospheric clouds or cloud cover.",
        "type": "region",
    },
    "ice": {
        "name": "Ice / snow",
        "description": "A surface region visually consistent with snow, glacier ice, or a polar ice sheet.",
        "type": "region",
    },
    "water": {
        "name": "Open water",
        "description": "A surface region visually consistent with ocean, sea, lake, or other open water.",
        "type": "region",
    },
    "forest": {
        "name": "Forest / vegetation",
        "description": "A surface region containing visually dense vegetation or forest cover.",
        "type": "region",
    },
    "urban": {
        "name": "Urban development",
        "description": "A surface region visually consistent with cities, buildings, roads, or dense human development.",
        "type": "region",
    },
    "desert": {
        "name": "Desert / dry terrain",
        "description": "A broad surface region visually consistent with dry, sandy, or barren terrain.",
        "type": "region",
    },
    "mountains": {
        "name": "Mountain terrain",
        "description": "A surface region containing rugged terrain or mountainous relief.",
        "type": "region",
    },
    "crater": {
        "name": "Impact crater",
        "description": "A circular or roughly circular depression that may be visually consistent with an impact crater.",
        "type": "region",
    },
    "fire": {
        "name": "Fire / smoke plume",
        "description": "A region that may be visually consistent with active fire or a smoke plume.",
        "type": "phenomenon",
    },
    "storm": {
        "name": "Storm system",
        "description": "A large organized atmospheric cloud structure that may be consistent with a storm system.",
        "type": "phenomenon",
    },
}


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_data_url(image, quality=84):
    image = image.convert("RGB").copy()
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
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_for_analysis(image, max_side=820):
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )


def clean_mask(mask, open_size=3, close_size=7):
    mask = (mask > 0).astype(np.uint8)
    if mask.size == 0:
        return mask.astype(np.float32)

    k1 = np.ones((open_size, open_size), np.uint8)
    k2 = np.ones((close_size, close_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask.astype(np.float32)


def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8)
    if not np.any(mask):
        return mask.astype(np.float32)

    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)

    holes = (flood == 0).astype(np.uint8)
    result = np.maximum(mask, holes)
    return result.astype(np.float32)


def mask_area_percent(mask):
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.mean(mask > 0.5) * 100.0)


def bbox_from_mask(mask, minimum_area=20):
    ys, xs = np.where(mask > 0.5)
    if len(xs) < minimum_area:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_overlap(a, b):
    if a is None or b is None:
        return 0.0
    aa = a > 0.5
    bb = b > 0.5
    amin = int(aa.sum())
    bmin = int(bb.sum())
    if amin == 0 or bmin == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / min(amin, bmin))


def connected_component_mask(mask, min_area_fraction=0.001, max_components=12):
    """Keep meaningful connected regions and discard tiny speckles."""
    mask = (mask > 0).astype(np.uint8)
    total = mask.size
    min_pixels = max(20, int(total * min_area_fraction))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)

    components = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_pixels:
            components.append((area, i))

    components.sort(reverse=True)
    for _, i in components[:max_components]:
        result[labels == i] = 1

    return result.astype(np.float32)


# ============================================================
# IMAGE QUALITY
# ============================================================

def check_image_quality(image):
    width, height = image.size

    if min(width, height) < 180:
        return {
            "valid": False,
            "reason": "low_resolution",
            "message": "The image resolution is too low for reliable visual analysis.",
        }

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    contrast = float(gray.std())
    if contrast < 10:
        return {
            "valid": False,
            "reason": "low_information",
            "message": "The image contains too little visual contrast or information.",
        }

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < 18:
        return {
            "valid": False,
            "reason": "blurry",
            "message": "The observation is too blurry for reliable visual analysis.",
        }

    return {
        "valid": True,
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 1),
        "contrast": round(contrast, 1),
    }


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_visual_features(image):
    image = resize_for_analysis(image, 820)
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # More selective base masks.
    blue = (
        (h >= 88) & (h <= 135) &
        (s >= 45) & (v >= 55)
    )

    cyan_blue = (
        (h >= 80) & (h < 100) &
        (s >= 35) & (v >= 70)
    )

    green = (
        (h >= 28) & (h <= 92) &
        (s >= 48) & (v >= 35)
    )

    tan = (
        (h >= 7) & (h <= 30) &
        (s >= 38) & (s <= 220) &
        (v >= 65)
    )

    white = (
        (s <= 48) &
        (v >= 178)
    )

    warm = (
        (((h <= 12) | (h >= 168)) & (s >= 125) & (v >= 125)) |
        ((h >= 8) & (h <= 30) & (s >= 145) & (v >= 145))
    )

    neutral = (
        (s <= 65) &
        (v >= 45) &
        (v <= 210)
    )

    dark = v < 85

    edges = cv2.Canny(gray, 55, 145)

    # Local texture at two scales.
    gray_f = gray.astype(np.float32)
    blur5 = cv2.GaussianBlur(gray_f, (0, 0), 3)
    blur11 = cv2.GaussianBlur(gray_f, (0, 0), 7)

    texture_fine = np.abs(gray_f - blur5)
    texture_coarse = np.abs(gray_f - blur11)

    texture_fine_score = float(np.mean(texture_fine > 10))
    texture_coarse_score = float(np.mean(texture_coarse > 14))

    # Local gradient magnitude.
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    return {
        "image": image,
        "rgb": rgb,
        "bgr": bgr,
        "hsv": hsv,
        "lab": lab,
        "gray": gray,
        "h": h,
        "s": s,
        "v": v,
        "blue": blue,
        "cyan_blue": cyan_blue,
        "green": green,
        "tan": tan,
        "white": white,
        "warm": warm,
        "neutral": neutral,
        "dark": dark,
        "edges": edges,
        "gradient": gradient,
        "texture_fine": texture_fine,
        "texture_coarse": texture_coarse,
        "texture_fine_score": texture_fine_score,
        "texture_coarse_score": texture_coarse_score,
        "edge_density": float(np.mean(edges > 0)),
    }


# ============================================================
# SPATIAL CONTEXT
# ============================================================

def make_position_maps(shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy.astype(np.float32) / max(1, h - 1)
    x = xx.astype(np.float32) / max(1, w - 1)

    # Distance from image center.
    center_dist = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

    return y, x, center_dist


def sky_probability(f):
    """
    Estimate sky using image position + blue/bright atmospheric colors.
    It is deliberately a probability rather than a hard classification.
    """
    y, _, _ = make_position_maps(f["gray"].shape)

    blue = f["blue"].astype(np.float32)
    cyan = f["cyan_blue"].astype(np.float32)
    white = f["white"].astype(np.float32)

    upper_bonus = np.clip((0.72 - y) / 0.72, 0, 1)

    probability = (
        blue * 0.60
        + cyan * 0.25
        + white * 0.22
    )

    probability *= (0.45 + 0.55 * upper_bonus)

    return np.clip(probability, 0, 1)


def land_probability(f):
    y, _, _ = make_position_maps(f["gray"].shape)

    green = f["green"].astype(np.float32)
    tan = f["tan"].astype(np.float32)
    neutral = f["neutral"].astype(np.float32)

    lower_bonus = 0.45 + 0.55 * y

    probability = (
        green * 0.48
        + tan * 0.48
        + neutral * 0.12
    ) * lower_bonus

    return np.clip(probability, 0, 1)


# ============================================================
# DETECTOR UTILITIES
# ============================================================

def detection(feature_id, confidence, mask, area=None, evidence=None):
    if area is None:
        area = mask_area_percent(mask)

    return {
        "id": feature_id,
        "name": FEATURES[feature_id]["name"],
        "description": FEATURES[feature_id]["description"],
        "confidence": float(np.clip(confidence, 0, 95)),
        "area_percent": round(float(area), 1) if area is not None else None,
        "mask": mask,
        "type": FEATURES[feature_id]["type"],
        "evidence": evidence or [],
    }


# ============================================================
# WATER
# ============================================================

def detect_water(f):
    y, _, _ = make_position_maps(f["gray"].shape)

    blue = f["blue"] | f["cyan_blue"]

    # Water should normally be smooth at the local scale.
    texture = f["texture_coarse"]
    smooth = texture < 17

    # Sky is a competing explanation. Favor lower/middle blue regions.
    lower_bonus = 0.30 + 0.70 * y

    candidate = blue & smooth & (lower_bonus > 0.48)

    candidate = connected_component_mask(candidate, 0.003, 10)
    area = mask_area_percent(candidate)

    if area < 2.5:
        return None

    blue_strength = float(np.mean(blue[candidate > 0])) if np.any(candidate) else 0
    smooth_strength = float(np.mean(smooth[candidate > 0])) if np.any(candidate) else 0

    # Reject if nearly all blue is confined to the upper sky.
    ys = np.where(candidate > 0)[0]
    if len(ys) and float(np.mean(ys)) < f["gray"].shape[0] * 0.32:
        return None

    score = (
        min(1, area / 45) * 0.45
        + blue_strength * 0.35
        + smooth_strength * 0.20
    )

    confidence = 52 + score * 37

    return detection(
        "water",
        confidence,
        candidate,
        area,
        ["blue/dark-blue surface color", "relatively smooth water-like texture"],
    )


# ============================================================
# VEGETATION
# ============================================================

def detect_forest(f):
    green = f["green"]

    # Vegetation should form connected patches, not isolated green pixels.
    mask = connected_component_mask(green, 0.002, 14)
    area = mask_area_percent(mask)

    if area < 3.0:
        return None

    texture = float(np.mean(f["texture_fine"][mask > 0] > 9)) if np.any(mask) else 0
    green_strength = float(np.mean(green[mask > 0])) if np.any(mask) else 0

    if texture < 0.18 and area < 8:
        return None

    score = (
        min(1, area / 55) * 0.35
        + green_strength * 0.40
        + min(1, texture * 2.2) * 0.25
    )

    confidence = 50 + score * 38

    return detection(
        "forest",
        confidence,
        mask,
        area,
        ["connected green surface regions", "vegetation-like color and texture"],
    )


# ============================================================
# DESERT
# ============================================================

def detect_desert(f):
    tan = f["tan"]
    green_ratio = float(np.mean(f["green"]))
    blue_ratio = float(np.mean(f["blue"]))

    mask = connected_component_mask(tan, 0.003, 12)
    area = mask_area_percent(mask)

    if area < 4.0:
        return None

    if green_ratio > 0.22 and green_ratio > float(np.mean(tan)) * 0.85:
        return None

    if blue_ratio > 0.35:
        return None

    texture = float(np.mean(f["texture_coarse"][mask > 0] > 10)) if np.any(mask) else 0
    tan_strength = float(np.mean(tan[mask > 0])) if np.any(mask) else 0

    score = (
        min(1, area / 60) * 0.35
        + tan_strength * 0.40
        + min(1, texture * 2) * 0.25
    )

    confidence = 48 + score * 40

    return detection(
        "desert",
        confidence,
        mask,
        area,
        ["broad tan/brown terrain", "dry-terrain-like texture"],
    )


# ============================================================
# ICE / SNOW
# ============================================================

def detect_ice(f):
    y, _, _ = make_position_maps(f["gray"].shape)

    white = f["white"]

    # Ice/snow is much more credible when it forms a broad surface region.
    # Avoid tiny white highlights.
    candidate = white.copy()

    # Very upper regions are more likely to be clouds.
    candidate &= ((y > 0.12) | (f["s"] < 35))

    mask = connected_component_mask(candidate, 0.0025, 10)
    mask = fill_holes(mask)
    mask = clean_mask(mask, 3, 9)

    area = mask_area_percent(mask)

    if area < 3.0:
        return None

    mean_sat = float(np.mean(f["s"][mask > 0])) if np.any(mask) else 100
    mean_value = float(np.mean(f["v"][mask > 0])) if np.any(mask) else 0

    # Ice needs low saturation + high brightness.
    if mean_sat > 75 or mean_value < 170:
        return None

    # Broad, contiguous white surfaces get stronger confidence.
    compactness = min(1, area / 35)

    score = (
        compactness * 0.38
        + (1 - min(1, mean_sat / 75)) * 0.34
        + min(1, mean_value / 235) * 0.28
    )

    confidence = 52 + score * 36

    return detection(
        "ice",
        confidence,
        mask,
        area,
        ["bright low-saturation surface", "broad contiguous snow/ice-like region"],
    )


# ============================================================
# CLOUDS
# ============================================================

def detect_clouds(f, ice_detection=None):
    sky = sky_probability(f)

    # Clouds are bright, low/moderate saturation and textured.
    cloud_color = (
        (f["white"] | (f["s"] < 72))
        & (f["v"] > 145)
    )

    textured = (
        (f["texture_fine"] > 7)
        | (f["texture_coarse"] > 11)
    )

    candidate = (
        cloud_color
        & textured
        & (sky > 0.12)
    )

    # Remove very dark pixels and large uniform white regions.
    candidate &= ~f["dark"]

    mask = connected_component_mask(candidate, 0.0015, 18)
    mask = clean_mask(mask, 3, 7)

    area = mask_area_percent(mask)

    if area < 1.5:
        return None

    cloud_pixels = mask > 0
    mean_sat = float(np.mean(f["s"][cloud_pixels])) if np.any(cloud_pixels) else 100
    texture_strength = float(np.mean(f["texture_fine"][cloud_pixels] > 7)) if np.any(cloud_pixels) else 0
    sky_strength = float(np.mean(sky[cloud_pixels])) if np.any(cloud_pixels) else 0

    # Ice/snow is a competing explanation.
    if ice_detection is not None:
        overlap = mask_overlap(mask, ice_detection["mask"])
        if overlap > 0.50:
            # Keep clouds only if they have substantially stronger sky/texture evidence.
            if sky_strength < 0.45 and texture_strength < 0.45:
                return None

    score = (
        min(1, area / 45) * 0.30
        + sky_strength * 0.30
        + texture_strength * 0.25
        + (1 - min(1, mean_sat / 100)) * 0.15
    )

    if score < 0.42:
        return None

    confidence = 50 + score * 37

    return detection(
        "cloud",
        confidence,
        mask,
        area,
        ["bright atmospheric-looking regions", "cloud-like texture", "upper-atmosphere spatial context"],
    )


# ============================================================
# FIRE
# ============================================================

def detect_fire(f):
    warm = f["warm"]

    # Fire should be concentrated, bright, and high-saturation.
    bright_warm = warm & (f["v"] > 145)

    mask = connected_component_mask(bright_warm, 0.00015, 10)
    area = mask_area_percent(mask)

    if area < 0.15 or area > 18:
        return None

    pixels = mask > 0
    saturation = float(np.mean(f["s"][pixels])) if np.any(pixels) else 0
    brightness = float(np.mean(f["v"][pixels])) if np.any(pixels) else 0

    # Fire-like regions should have strong red/orange color.
    if saturation < 130 or brightness < 145:
        return None

    # Fire confidence is intentionally capped because RGB imagery
    # cannot confirm active thermal fire by itself.
    concentration = min(1, area / 6)
    score = (
        concentration * 0.28
        + min(1, saturation / 220) * 0.37
        + min(1, brightness / 240) * 0.35
    )

    confidence = 48 + score * 34

    return detection(
        "fire",
        confidence,
        mask,
        area,
        ["concentrated red/orange region", "high brightness and saturation"],
    )


# ============================================================
# MOUNTAINS
# ============================================================

def detect_mountains(f):
    h, w = f["gray"].shape
    y, _, _ = make_position_maps((h, w))

    # Mountains are normally terrain, so prioritize lower/middle image.
    terrain = y > 0.30

    edges = f["edges"]
    gradient = f["gradient"]

    # Suppress tiny edge noise.
    edge_mask = (edges > 0) & terrain & (gradient > 45)

    # Look for strong connected ridge-like areas.
    dilated = cv2.dilate(
        edge_mask.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    mask = connected_component_mask(dilated, 0.001, 16)
    area = mask_area_percent(mask)

    if area < 1.0 or area > 38:
        return None

    lower_edges = float(np.mean(edges[y > 0.35] > 0))
    texture = f["texture_coarse_score"]

    # Mountain terrain should have stronger relief/edge structure than
    # a smooth desert or ocean.
    score = (
        min(1, lower_edges / 0.16) * 0.40
        + min(1, texture / 0.28) * 0.30
        + min(1, area / 20) * 0.30
    )

    if score < 0.40:
        return None

    confidence = 46 + score * 39

    return detection(
        "mountains",
        confidence,
        mask,
        area,
        ["rugged lower-image terrain", "strong relief/edge structure"],
    )


# ============================================================
# URBAN
# ============================================================

def detect_urban(f):
    h, w = f["gray"].shape
    y, _, _ = make_position_maps((h, w))

    gray_neutral = f["neutral"]

    # Urban areas need dense edges plus neutral/gray built surfaces.
    edge = f["edges"] > 0

    local_edge = cv2.GaussianBlur(
        edge.astype(np.float32),
        (0, 0),
        6,
    )

    dense = local_edge > 0.095
    candidate = dense & gray_neutral & (f["v"] > 55)

    mask = connected_component_mask(candidate, 0.001, 14)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(np.float32)

    area = mask_area_percent(mask)

    if area < 0.7 or area > 28:
        return None

    # Measure straight-line/grid evidence.
    lines = cv2.HoughLinesP(
        f["edges"],
        1,
        np.pi / 180,
        threshold=max(18, min(h, w) // 28),
        minLineLength=max(15, min(h, w) // 16),
        maxLineGap=8,
    )

    line_count = 0 if lines is None else len(lines)

    neutral_strength = float(np.mean(gray_neutral[mask > 0])) if np.any(mask) else 0
    edge_strength = float(np.mean(local_edge[mask > 0])) if np.any(mask) else 0

    score = (
        min(1, edge_strength / 0.28) * 0.42
        + neutral_strength * 0.28
        + min(1, line_count / 70) * 0.30
    )

    if score < 0.43:
        return None

    confidence = 45 + score * 38

    return detection(
        "urban",
        confidence,
        mask,
        area,
        ["dense edge network", "neutral built-surface colors", "repeated linear structure"],
    )


# ============================================================
# CRATERS
# ============================================================

def detect_craters(f):
    gray = f["gray"]
    h, w = gray.shape
    min_dim = min(h, w)

    # Craters are only considered when the image has a relatively
    # terrain-like appearance. This prevents ordinary circular objects
    # from being immediately reported as craters.
    terrain_ratio = max(
        float(np.mean(f["tan"])),
        float(np.mean(f["neutral"])),
    )

    if terrain_ratio < 0.20:
        return None

    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

    min_radius = max(7, int(min_dim * 0.018))
    max_radius = max(min_radius + 4, int(min_dim * 0.16))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(18, min_dim // 10),
        param1=95,
        param2=38,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    valid = []
    mask = np.zeros_like(gray, dtype=np.uint8)

    for x, y, r in circles[:12]:
        if r <= 0:
            continue

        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            continue

        # Compare center intensity with a ring around the circle.
        center = gray[
            max(0, y - r // 3):min(h, y + r // 3 + 1),
            max(0, x - r // 3):min(w, x + r // 3 + 1),
        ]

        ring_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(ring_mask, (x, y), int(r * 1.35), 1, -1)
        cv2.circle(ring_mask, (x, y), max(2, int(r * 0.75)), 0, -1)

        ring = gray[ring_mask > 0]

        if center.size < 10 or ring.size < 10:
            continue

        contrast = abs(float(center.mean()) - float(ring.mean()))

        if contrast >= 10:
            valid.append((x, y, r, contrast))
            cv2.circle(mask, (x, y), r, 1, -1)

    if not valid:
        return None

    mask = clean_mask(mask, 3, 5)
    area = mask_area_percent(mask)

    if area > 20:
        return None

    # One weak circle is not enough. Multiple coherent circles are stronger.
    count_score = min(1, len(valid) / 4)
    contrast_score = min(1, np.mean([v[3] for v in valid]) / 30)
    terrain_score = min(1, terrain_ratio / 0.55)

    score = (
        count_score * 0.35
        + contrast_score * 0.40
        + terrain_score * 0.25
    )

    if score < 0.40:
        return None

    confidence = 45 + score * 38

    return detection(
        "crater",
        confidence,
        mask,
        area,
        ["circular depression-like structures", "center/ring brightness contrast", "terrain-like surface context"],
    )


# ============================================================
# STORM
# ============================================================

def detect_storm(f, cloud_detection=None):
    if cloud_detection is None:
        return None

    cloud_mask = cloud_detection["mask"]
    cloud_area = cloud_detection["area_percent"]

    if cloud_area < 10:
        return None

    # Storms need organization, not just a lot of white pixels.
    gray = f["gray"]
    h, w = gray.shape

    cloud_binary = (cloud_mask > 0).astype(np.uint8)

    # Distance transform gives a crude estimate of broad cloud mass.
    dist = cv2.distanceTransform(cloud_binary, cv2.DIST_L2, 5)
    broad_mass = float(np.mean(dist > max(2.0, min(h, w) * 0.012)))

    # Contours help estimate whether clouds form several organized lobes.
    contours, _ = cv2.findContours(
        cloud_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    large_contours = [
        c for c in contours
        if cv2.contourArea(c) > gray.size * 0.001
    ]

    # Curved/rotational structure cue.
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 30, 95)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, min(h, w) // 28),
        minLineLength=max(20, min(h, w) // 9),
        maxLineGap=18,
    )

    line_count = 0 if lines is None else len(lines)

    organization = min(1, len(large_contours) / 6) * 0.25
    organization += min(1, broad_mass / 0.18) * 0.40
    organization += min(1, line_count / 35) * 0.35

    # Avoid calling an ordinary cloud field a storm.
    if organization < 0.43:
        return None

    confidence = 48 + organization * 37

    return detection(
        "storm",
        confidence,
        None,
        None,
        ["large organized cloud mass", "broad internal structure", "atmospheric pattern consistent with storm organization"],
    )


# ============================================================
# RELEVANCE
# ============================================================

def check_relevance(f):
    h, w = f["gray"].shape
    y, _, _ = make_position_maps((h, w))

    color_evidence = max(
        float(np.mean(f["blue"])),
        float(np.mean(f["green"])),
        float(np.mean(f["tan"])),
        float(np.mean(f["white"])),
    )

    natural_structure = min(
        1.0,
        f["edge_density"] * 3.5
        + f["texture_coarse_score"] * 1.4,
    )

    # Space/Earth observations often contain broad low-frequency regions.
    broad_region = max(
        float(np.mean(f["blue"] | f["green"] | f["tan"] | f["white"])),
        0.0,
    )

    score = (
        min(1, color_evidence * 2.2) * 0.42
        + natural_structure * 0.30
        + min(1, broad_region * 2.0) * 0.28
    )

    return int(np.clip(round(22 + score * 78), 0, 100))


# ============================================================
# DETECTION ORCHESTRATION
# ============================================================

def detect_features(f):
    results = []

    ice = detect_ice(f)
    if ice:
        results.append(ice)

    cloud = detect_clouds(f, ice)
    if cloud:
        results.append(cloud)

    water = detect_water(f)
    if water:
        results.append(water)

    forest = detect_forest(f)
    if forest:
        results.append(forest)

    desert = detect_desert(f)
    if desert:
        results.append(desert)

    mountains = detect_mountains(f)
    if mountains:
        results.append(mountains)

    urban = detect_urban(f)
    if urban:
        results.append(urban)

    crater = detect_craters(f)
    if crater:
        results.append(crater)

    fire = detect_fire(f)
    if fire:
        results.append(fire)

    storm = detect_storm(f, cloud)
    if storm:
        results.append(storm)

    # Strong conflict suppression.
    ids = {r["id"] for r in results}

    if "water" in ids and "forest" in ids:
        water = next(r for r in results if r["id"] == "water")
        forest = next(r for r in results if r["id"] == "forest")

        # They can legitimately coexist, so only suppress one when one
        # overwhelmingly dominates the same image.
        if water["area_percent"] > 55 and forest["area_percent"] < 4:
            results = [r for r in results if r["id"] != "forest"]

    if "cloud" in ids and "ice" in ids:
        cloud = next(r for r in results if r["id"] == "cloud")
        ice = next(r for r in results if r["id"] == "ice")

        if mask_overlap(cloud["mask"], ice["mask"]) > 0.60:
            if ice["confidence"] >= cloud["confidence"]:
                results = [r for r in results if r["id"] != "cloud"]
            else:
                results = [r for r in results if r["id"] != "ice"]

    # Sort by evidence strength and only expose genuinely useful results.
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return [r for r in results if r["confidence"] >= 55][:7]


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
    base = np.asarray(image.convert("RGB")).copy()
    output = base.copy()

    for d in detections:
        mask = d.get("mask")
        if mask is None:
            continue

        binary = (mask > 0.5).astype(np.uint8)
        color = COLORS.get(d["id"], (80, 220, 255))

        color_array = np.zeros_like(base)
        color_array[:, :] = color

        alpha = binary[..., None].astype(np.float32) * 0.24

        output = (
            output * (1 - alpha) +
            color_array * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = [
            c for c in contours
            if cv2.contourArea(c) > 60
        ]

        cv2.drawContours(
            output,
            contours,
            -1,
            color,
            2,
        )

    return Image.fromarray(output)


# ============================================================
# EXPLANATION
# ============================================================

def build_explanation(detections):
    if not detections:
        return (
            "The image passed the basic quality and Earth-observation "
            "relevance checks, but the available visual evidence was "
            "not strong enough to identify a supported feature confidently."
        )

    names = {d["name"] for d in detections}
    statements = []

    if "Open water" in names:
        statements.append(
            "A broad blue or dark-blue surface region is visually consistent with open water."
        )

    if "Forest / vegetation" in names:
        statements.append(
            "Connected green textured regions are visually consistent with vegetation."
        )

    if "Ice / snow" in names:
        statements.append(
            "A broad bright, low-saturation surface region is visually consistent with snow or ice."
        )

    if "Cloud formation" in names:
        statements.append(
            "Bright textured atmospheric-looking regions are consistent with cloud cover."
        )

    if "Storm system" in names:
        statements.append(
            "The cloud field contains broad organized structure that may be consistent with a storm system."
        )

    if "Desert / dry terrain" in names:
        statements.append(
            "A broad tan or dry-looking surface region is consistent with arid terrain."
        )

    if "Mountain terrain" in names:
        statements.append(
            "Strong lower-image relief and edge structure are consistent with rugged or mountainous terrain."
        )

    if "Urban development" in names:
        statements.append(
            "A dense network of edges and repeated linear structures is visually consistent with built development."
        )

    if "Impact crater" in names:
        statements.append(
            "Circular surface structures with center-to-ring contrast were detected and may be consistent with impact craters."
        )

    if "Fire / smoke plume" in names:
        statements.append(
            "A concentrated bright red/orange region may be consistent with fire; RGB imagery alone cannot confirm active thermal fire."
        )

    explanation = " ".join(statements)

    explanation += (
        " EarthScope reports visual evidence rather than claiming a confirmed "
        "scientific classification. This CPU-only engine cannot establish exact "
        "location, wind speed, pressure, temperature, fire temperature, or official "
        "storm intensity/category."
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
        "engine": "improved-lightweight-cpu",
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
            image = Image.open(io.BytesIO(raw)).convert("RGB")
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

        print(f"[VISION] Relevance estimate: {relevance}%")

        if relevance < 30:
            return jsonify({
                "status": "unsuitable",
                "reason": "unrelated",
                "message": (
                    "The uploaded image does not contain enough visual evidence "
                    "of an Earth, planetary, or space observation."
                ),
                "relevance": relevance,
                "original_image": image_to_data_url(image),
            })

        detections = detect_features(features)

        public_detections = []

        for d in detections:
            public_detections.append({
                "id": d["id"],
                "name": d["name"],
                "description": d["description"],
                "confidence": round(float(d["confidence"]), 1),
                "area_percent": d["area_percent"],
                "type": d["type"],
                "evidence": d.get("evidence", []),
            })

        annotated = overlay_features(features["image"], detections)

        if not detections:
            return jsonify({
                "status": "no_features",
                "message": (
                    "The image is relevant, but no supported feature "
                    "reached the confidence threshold."
                ),
                "relevance": relevance,
                "detections": [],
                "explanation": build_explanation([]),
                "original_image": image_to_data_url(image),
                "annotated_image": image_to_data_url(annotated),
                "model_note": (
                    "The improved lightweight CPU engine uses multiple "
                    "visual cues and spatial checks. It intentionally avoids "
                    "claiming a feature when evidence is weak."
                ),
            })

        return jsonify({
            "status": "success",
            "message": "Observation successfully analyzed.",
            "relevance": relevance,
            "detections": public_detections,
            "explanation": build_explanation(detections),
            "original_image": image_to_data_url(image),
            "annotated_image": image_to_data_url(annotated),
            "model_note": (
                "Detections are CPU-based visual estimates using color, "
                "texture, spatial context, edges, connected regions, and "
                "shape cues. They are not scientific measurements or official "
                "meteorological classifications."
            ),
        })

    except Exception as error:
        print("=" * 60)
        print("EARTHSCOPE SERVER ERROR")
        print("=" * 60)
        print(repr(error))
        print("=" * 60)

        return jsonify({
            "status": "error",
            "message": "The observation engine encountered a server-side error.",
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
