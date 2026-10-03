from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter
import io
import base64
import numpy as np
import cv2
import torch

from transformers import (
    CLIPProcessor,
    CLIPModel,
    CLIPSegProcessor,
    CLIPSegForImageSegmentation
)


# ============================================================
# EARTHSCOPE
# AI PLANETARY OBSERVATION SYSTEM
# ============================================================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"


device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("EARTHSCOPE AI // ANALYSIS ENGINE")
print("=" * 60)
print(f"Device: {device}")
print("Loading visual models...")
print()


# ============================================================
# LOAD CLIP
# ============================================================

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model.to(device)
clip_model.eval()


# ============================================================
# LOAD CLIPSEG
# ============================================================

clipseg_processor = CLIPSegProcessor.from_pretrained(
    CLIPSEG_MODEL_NAME
)

clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    CLIPSEG_MODEL_NAME
)

clipseg_model.to(device)
clipseg_model.eval()


print()
print("EARTHSCOPE AI // ANALYSIS ENGINE READY")
print("=" * 60)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = {

    "cloud": {
        "name": "Cloud formation",

        "description":
            "A visible concentration of atmospheric clouds or cloud cover.",

        "prompts": [
            "atmospheric clouds viewed from space",
            "clouds floating above Earth's surface",
            "white cloud formations in Earth's atmosphere",
            "weather satellite cloud cover"
        ],

        "verification_prompts": [
            "atmospheric cloud cover",
            "clouds above Earth",
            "surface ice or snow",
            "polar ice sheet"
        ],

        "type": "region"
    },


    "ice": {
        "name": "Ice / snow",

        "description":
            "A surface region visually consistent with snow, glacier ice, or a polar ice sheet.",

        "prompts": [
            "polar ice sheet covering land",
            "glacier ice viewed from space",
            "snow covered land viewed from orbit",
            "Antarctic ice sheet",
            "Arctic sea ice",
            "large frozen surface viewed from space"
        ],

        "verification_prompts": [
            "surface ice and snow",
            "polar ice sheet",
            "glacier",
            "atmospheric clouds"
        ],

        "type": "region"
    },


    "water": {
        "name": "Open water",

        "description":
            "A surface region visually consistent with ocean, sea, lake, or other open water.",

        "prompts": [
            "blue ocean viewed from space",
            "open ocean surface viewed from satellite",
            "sea water viewed from orbit",
            "large body of water seen from space",
            "dark blue ocean covering Earth's surface",
            "lake viewed from satellite"
        ],

        "verification_prompts": [
            "open ocean water",
            "sea water",
            "lake or river",
            "clouds",
            "ice"
        ],

        "type": "region"
    },


    "forest": {
        "name": "Forest / vegetation",

        "description":
            "A surface region containing visually dense vegetation or forest cover.",

        "prompts": [
            "dense forest viewed from space",
            "green vegetation covering land",
            "forest canopy viewed from orbit"
        ],

        "verification_prompts": [
            "forest and vegetation",
            "green land",
            "clouds",
            "desert"
        ],

        "type": "region"
    },


    "urban": {
        "name": "Urban development",

        "description":
            "A surface region visually consistent with cities, buildings, roads, or dense human development.",

        "prompts": [
            "dense city viewed from satellite",
            "urban area viewed from space",
            "buildings and roads seen from orbit"
        ],

        "verification_prompts": [
            "city",
            "urban area",
            "natural landscape",
            "clouds"
        ],

        "type": "region"
    },


    "desert": {
        "name": "Desert / dry terrain",

        "description":
            "A broad surface region visually consistent with dry, sandy, or barren terrain.",

        "prompts": [
            "desert landscape viewed from space",
            "dry barren terrain from satellite",
            "sandy desert viewed from orbit"
        ],

        "verification_prompts": [
            "desert",
            "dry barren land",
            "forest",
            "water"
        ],

        "type": "region"
    },


    "mountains": {
        "name": "Mountain terrain",

        "description":
            "A surface region containing rugged terrain or mountainous relief.",

        "prompts": [
            "mountain range viewed from satellite",
            "rugged mountainous terrain from space",
            "high relief terrain viewed from orbit"
        ],

        "verification_prompts": [
            "mountains",
            "rugged terrain",
            "flat surface",
            "clouds"
        ],

        "type": "region"
    },


    "crater": {
        "name": "Impact crater",

        "description":
            "A circular or roughly circular depression in a planetary surface that may be consistent with an impact crater.",

        "prompts": [
            "impact crater on planetary surface",
            "large circular impact crater viewed from space",
            "crater on the Moon",
            "cratered lunar surface",
            "circular depression on rocky planetary terrain",
            "meteor impact crater viewed from orbit",
            "multiple impact craters on a planetary surface"
        ],

        "verification_prompts": [
            "impact crater",
            "lunar crater",
            "cratered planetary surface",
            "clouds",
            "ice"
        ],

        "type": "region"
    },


    "fire": {
        "name": "Fire / smoke plume",

        "description":
            "A region that may be visually consistent with active fire, smoke, or a smoke plume.",

        "prompts": [
            "wildfire and smoke plume viewed from satellite",
            "active fire on Earth's surface",
            "large smoke plume seen from space"
        ],

        "verification_prompts": [
            "wildfire",
            "smoke plume",
            "normal landscape",
            "clouds"
        ],

        "type": "phenomenon"
    },


    "storm": {
        "name": "Storm system",

        "description":
            "A large organized atmospheric cloud structure that may be consistent with a storm system.",

        "prompts": [
            "large organized storm system viewed from space",
            "cyclonic cloud system over Earth",
            "large atmospheric storm viewed from satellite"
        ],

        "verification_prompts": [
            "large storm system",
            "cyclonic atmospheric cloud",
            "ordinary cloud cover",
            "polar ice sheet"
        ],

        "type": "phenomenon"
    }
}


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_data_url(image):
    """
    Convert PIL image to browser-readable base64 image.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# IMAGE QUALITY
# ============================================================

def check_image_quality(image):

    width, height = image.size

    min_side = min(
        width,
        height
    )

    if min_side < 160:

        return {
            "valid": False,
            "reason": "low_resolution",
            "message":
                "The image resolution is too low for reliable visual analysis."
        }


    gray = np.array(
        image.convert("L")
    )


    contrast = float(
        gray.std()
    )


    if contrast < 8:

        return {
            "valid": False,
            "reason": "low_information",
            "message":
                "The image contains too little visual contrast or information."
        }


    # Laplacian variance
    # measures image sharpness.

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )


    if sharpness < 20:

        return {
            "valid": False,
            "reason": "blurry",
            "message":
                "The observation is too blurry for reliable visual analysis."
        }


    return {
        "valid": True
    }


# ============================================================
# CLIP IMAGE CLASSIFICATION
# ============================================================

@torch.no_grad()
def clip_compare(image, labels):

    inputs = clip_processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    outputs = clip_model(
        **inputs
    )


    logits = outputs.logits_per_image[0]

    probabilities = torch.softmax(
        logits,
        dim=0
    )


    results = {}

    for label, probability in zip(
        labels,
        probabilities
    ):

        results[label] = float(
            probability.item()
        )


    return results


# ============================================================
# EARTH RELEVANCE
# ============================================================

def check_relevance(image):

    labels = [

        "Earth satellite observation",
        "Earth viewed from space",
        "weather satellite image",
        "planetary surface",
        "ocean viewed from space",

        "ordinary photograph",
        "person portrait",
        "indoor room",
        "food photograph",
        "product photograph"
    ]


    scores = clip_compare(
        image,
        labels
    )


    earth_labels = labels[:5]

    unrelated_labels = labels[5:]


    earth_score = np.mean([
        scores[label]
        for label in earth_labels
    ])


    unrelated_score = np.mean([
        scores[label]
        for label in unrelated_labels
    ])


    relevance = (
        earth_score /
        max(
            earth_score + unrelated_score,
            1e-8
        )
    )


    relevance_percent = int(
        round(
            relevance * 100
        )
    )


    return relevance_percent


# ============================================================
# SPECIAL ICE VS CLOUD CLASSIFIER
# ============================================================

@torch.no_grad()
def classify_ice_vs_cloud(image):

    labels = [

        "surface ice and snow on Earth",
        "polar ice sheet",
        "glacier or snow covered land",

        "atmospheric cloud cover",
        "clouds floating above Earth"
    ]


    scores = clip_compare(
        image,
        labels
    )


    ice_score = np.mean([
        scores[
            "surface ice and snow on Earth"
        ],
        scores[
            "polar ice sheet"
        ],
        scores[
            "glacier or snow covered land"
        ]
    ])


    cloud_score = np.mean([
        scores[
            "atmospheric cloud cover"
        ],
        scores[
            "clouds floating above Earth"
        ]
    ])


    total = (
        ice_score +
        cloud_score +
        1e-8
    )


    ice_ratio = ice_score / total

    cloud_ratio = cloud_score / total


    return {
        "ice": float(ice_ratio),
        "cloud": float(cloud_ratio)
    }


# ============================================================
# CLIPSEG MASK
# ============================================================

@torch.no_grad()
def generate_mask(image, prompts):

    inputs = clipseg_processor(
        text=prompts,
        images=[image] * len(prompts),
        padding=True,
        return_tensors="pt"
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    outputs = clipseg_model(
        **inputs
    )


    logits = outputs.logits


    # Average the multiple prompts.
    logits = logits.mean(
        dim=0
    )


    probabilities = torch.sigmoid(
        logits
    )


    mask = probabilities.cpu().numpy()


    # Resize to original image size.

    mask = cv2.resize(
        mask,
        image.size,
        interpolation=cv2.INTER_LINEAR
    )


    return mask


# ============================================================
# CLEAN MASK
# ============================================================

def clean_mask(mask):

    mask = np.clip(
        mask,
        0,
        1
    )


    binary = (
        mask > 0.50
    ).astype(
        np.uint8
    )


    kernel = np.ones(
        (7, 7),
        np.uint8
    )


    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )


    return binary.astype(
        np.float32
    )


# ============================================================
# MASK AREA
# ============================================================

def mask_area_percent(mask):

    if mask is None:
        return 0.0


    pixels = np.sum(
        mask > 0.5
    )


    total = mask.size


    return float(
        (pixels / total) * 100
    )


# ============================================================
# MASK OVERLAP
# ============================================================

def mask_overlap(mask_a, mask_b):

    a = mask_a > 0.5
    b = mask_b > 0.5


    intersection = np.logical_and(
        a,
        b
    ).sum()


    smaller_area = min(
        a.sum(),
        b.sum()
    )


    if smaller_area == 0:
        return 0.0


    return float(
        intersection /
        smaller_area
    )
# ============================================================
# WATER DETECTOR
# ============================================================

def detect_water(image):

    prompts = FEATURES["water"]["prompts"]

    raw_mask = generate_mask(
        image,
        prompts
    )

    water_mask = clean_mask(
        raw_mask
    )

    area = mask_area_percent(
        water_mask
    )

    # Very small regions are ignored.
    if area < 1.0:
        return None

    # Image-level verification.
    labels = [
        "open ocean water",
        "sea water",
        "lake or river",
        "clouds",
        "ice",
        "land surface"
    ]

    scores = clip_compare(
        image,
        labels
    )

    water_score = np.mean([
        scores["open ocean water"],
        scores["sea water"],
        scores["lake or river"]
    ])

    competing_score = np.mean([
        scores["clouds"],
        scores["ice"],
        scores["land surface"]
    ])

    # Water needs a meaningful image-level signal.
    if water_score < 0.25:
        return None

    # Prevent obvious ice/cloud scenes from becoming water.
    if water_score <= competing_score * 0.75:
        return None

    confidence = min(
        0.97,
        0.42 + water_score * 0.55
    )

    return {
        "id": "water",
        "name": FEATURES["water"]["name"],
        "description": FEATURES["water"]["description"],
        "confidence": confidence * 100,
        "area_percent": round(area, 1),
        "mask": water_mask,
        "type": "region"
    }
# ============================================================
# CRATER DETECTOR
# ============================================================

def detect_craters(image):

    # --------------------------------------------------------
    # First determine whether the scene actually looks like
    # a rocky planetary / lunar surface.
    # --------------------------------------------------------

    scene_labels = [

        "Moon surface viewed from space",
        "planetary surface with impact craters",
        "rocky extraterrestrial surface",
        "lunar landscape",

        "Earth atmosphere",
        "Earth cloud cover",
        "Earth ocean",
        "Earth ice sheet"
    ]

    scene_scores = clip_compare(
        image,
        scene_labels
    )

    planetary_score = np.mean([
        scene_scores[
            "Moon surface viewed from space"
        ],
        scene_scores[
            "planetary surface with impact craters"
        ],
        scene_scores[
            "rocky extraterrestrial surface"
        ],
        scene_scores[
            "lunar landscape"
        ]
    ])

    earth_score = np.mean([
        scene_scores[
            "Earth atmosphere"
        ],
        scene_scores[
            "Earth cloud cover"
        ],
        scene_scores[
            "Earth ocean"
        ],
        scene_scores[
            "Earth ice sheet"
        ]
    ])


    # --------------------------------------------------------
    # Generate crater segmentation mask.
    # --------------------------------------------------------

    prompts = FEATURES[
        "crater"
    ]["prompts"]

    raw_mask = generate_mask(
        image,
        prompts
    )

    crater_mask = clean_mask(
        raw_mask
    )

    area = mask_area_percent(
        crater_mask
    )


    if area < 0.3:
        return None


    # --------------------------------------------------------
    # Dedicated crater classification.
    # --------------------------------------------------------

    crater_labels = [

        "impact crater",
        "large circular impact crater",
        "lunar crater",
        "cratered planetary surface",

        "ordinary Earth landscape",
        "clouds",
        "ice and snow",
        "ocean"
    ]

    crater_scores = clip_compare(
        image,
        crater_labels
    )


    crater_score = np.mean([
        crater_scores[
            "impact crater"
        ],
        crater_scores[
            "large circular impact crater"
        ],
        crater_scores[
            "lunar crater"
        ],
        crater_scores[
            "cratered planetary surface"
        ]
    ])


    competing_score = np.mean([
        crater_scores[
            "ordinary Earth landscape"
        ],
        crater_scores[
            "clouds"
        ],
        crater_scores[
            "ice and snow"
        ],
        crater_scores[
            "ocean"
        ]
    ])


    # --------------------------------------------------------
    # Crater verification.
    # --------------------------------------------------------

    if crater_score < 0.25:
        return None


    if crater_score <= competing_score:
        return None


    # A planetary/lunar scene gets a bonus.
    if planetary_score > earth_score:

        confidence = (
            0.48
            +
            crater_score * 0.45
        )

    else:

        confidence = (
            0.35
            +
            crater_score * 0.40
        )


    confidence = min(
        confidence,
        0.98
    )


    if confidence < 0.45:
        return None


    return {
        "id": "crater",
        "name": FEATURES["crater"]["name"],
        "description": FEATURES["crater"]["description"],
        "confidence": confidence * 100,
        "area_percent": round(area, 1),
        "mask": crater_mask,
        "type": "region"
    }

# ============================================================
# SURFACE ICE DETECTOR
# ============================================================

def detect_surface_ice(image):

    prompts = FEATURES[
        "ice"
    ]["prompts"]


    raw_mask = generate_mask(
        image,
        prompts
    )


    mask = clean_mask(
        raw_mask
    )


    area = mask_area_percent(
        mask
    )


    # Image-level verification.
    comparison = classify_ice_vs_cloud(
        image
    )


    ice_ratio = comparison[
        "ice"
    ]


    cloud_ratio = comparison[
        "cloud"
    ]


    # Strong evidence of surface ice.
    is_ice = (

        (
            ice_ratio >= 0.48
        )

        or

        (
            area >= 8
            and
            ice_ratio >= 0.42
        )
    )


    if not is_ice:
        return None


    confidence = (
        0.55
        +
        (
            ice_ratio * 0.45
        )
    )


    confidence = min(
        confidence,
        0.99
    )


    return {
        "id": "ice",
        "name": FEATURES["ice"]["name"],
        "description": FEATURES["ice"]["description"],
        "confidence": confidence * 100,
        "area_percent": round(
            area,
            1
        ),
        "mask": mask,
        "type": "region"
    }


# ============================================================
# CLOUD DETECTOR
# ============================================================

def detect_clouds(
    image,
    ice_detection=None
):

    prompts = FEATURES[
        "cloud"
    ]["prompts"]


    raw_mask = generate_mask(
        image,
        prompts
    )


    cloud_mask = clean_mask(
        raw_mask
    )


    area = mask_area_percent(
        cloud_mask
    )


    # --------------------------------------------------------
    # Image-level ice/cloud competition
    # --------------------------------------------------------

    comparison = classify_ice_vs_cloud(
        image
    )


    ice_ratio = comparison[
        "ice"
    ]


    cloud_ratio = comparison[
        "cloud"
    ]


    # --------------------------------------------------------
    # CRITICAL FIX
    #
    # If the model thinks a large amount of the image is
    # cloud but the image-level classifier strongly identifies
    # the scene as surface ice, reject the cloud detection.
    # --------------------------------------------------------

    if ice_detection is not None:

        overlap = mask_overlap(
            cloud_mask,
            ice_detection["mask"]
        )


        # Huge cloud mask sitting over the same region as
        # detected ice is usually the classic CLIPSeg failure
        # mode we are correcting.

        if (
            overlap >= 0.35
            and
            ice_ratio > cloud_ratio
            and
            area > 20
        ):

            print(
                "[VISION] Cloud detection suppressed:"
                f" ice/cloud={ice_ratio:.2f}/{cloud_ratio:.2f}"
                f" overlap={overlap:.2f}"
                f" area={area:.1f}%"
            )

            return None


        # Even without massive overlap, a dominant ice signal
        # should prevent an enormous cloud classification.

        if (
            ice_ratio >= 0.60
            and
            area > 35
            and
            cloud_ratio < ice_ratio
        ):

            print(
                "[VISION] Large cloud detection rejected"
                " because surface ice dominates."
            )

            return None


    # --------------------------------------------------------
    # Normal cloud confidence
    # --------------------------------------------------------

    confidence = (
        0.45
        +
        (
            cloud_ratio * 0.45
        )
    )


    confidence = min(
        confidence,
        0.98
    )


    # Require reasonable image-level cloud evidence.

    if cloud_ratio < 0.38:

        return None


    if area < 1.0:

        return None


    return {
        "id": "cloud",
        "name": FEATURES["cloud"]["name"],
        "description": FEATURES["cloud"]["description"],
        "confidence": confidence * 100,
        "area_percent": round(
            area,
            1
        ),
        "mask": cloud_mask,
        "type": "region"
    }


# ============================================================
# GENERIC FEATURE DETECTOR
# ============================================================

def detect_generic_feature(
    image,
    feature_id,
    minimum_confidence=0.38
):

    feature = FEATURES[feature_id]


    raw_mask = generate_mask(
        image,
        feature["prompts"]
    )


    mask = clean_mask(
        raw_mask
    )


    area = mask_area_percent(
        mask
    )


    if area < 0.8:
        return None


    # Image-level verification.

    verification = clip_compare(
        image,
        feature["verification_prompts"]
    )


    target_scores = []


    # The first prompts are always the positive prompts.
    positive_count = max(
        1,
        len(feature["verification_prompts"]) // 2
    )


    for label in feature[
        "verification_prompts"
    ][:positive_count]:

        target_scores.append(
            verification[label]
        )


    positive_score = np.mean(
        target_scores
    )


    confidence = min(
        0.98,
        (
            0.35
            +
            positive_score * 0.65
        )
    )


    if confidence < minimum_confidence:
        return None


    return {
        "id": feature_id,
        "name": feature["name"],
        "description": feature["description"],
        "confidence": confidence * 100,
        "area_percent": round(
            area,
            1
        ),
        "mask": mask,
        "type": feature["type"]
    }


# ============================================================
# STORM DETECTOR
# ============================================================

def detect_storm(image):

    labels = [

        "large organized storm system",
        "cyclonic cloud system",
        "ordinary cloud cover",
        "surface ice sheet",
        "normal Earth surface"
    ]


    scores = clip_compare(
        image,
        labels
    )


    storm_score = np.mean([
        scores[
            "large organized storm system"
        ],
        scores[
            "cyclonic cloud system"
        ]
    ])


    ordinary_cloud = scores[
        "ordinary cloud cover"
    ]


    ice_score = scores[
        "surface ice sheet"
    ]


    # Storm should only be considered when the storm signal
    # clearly exceeds ordinary cloud cover and ice.

    if storm_score < 0.30:
        return None


    if storm_score <= ordinary_cloud:
        return None


    if storm_score <= ice_score:
        return None


    confidence = min(
        0.97,
        0.40 +
        storm_score * 0.55
    )


    return {
        "id": "storm",
        "name": FEATURES["storm"]["name"],
        "description": FEATURES["storm"]["description"],
        "confidence": confidence * 100,
        "area_percent": None,
        "mask": None,
        "type": "phenomenon"
    }


# ============================================================
# FIRE DETECTOR
# ============================================================

def detect_fire(image):

    result = detect_generic_feature(
        image,
        "fire",
        minimum_confidence=0.45
    )


    return result


# ============================================================
# ALL FEATURE DETECTION
# ============================================================

def detect_features(image):

    detections = []

    print()
    print("[VISION] Starting feature analysis...")


    # ========================================================
    # 1. ICE
    # ========================================================

    ice_detection = detect_surface_ice(
        image
    )

    if ice_detection:

        detections.append(
            ice_detection
        )

        print(
            "[VISION] Ice detected:",
            f"{ice_detection['confidence']:.1f}%"
        )


    # ========================================================
    # 2. CLOUDS
    # ========================================================

    cloud_detection = detect_clouds(
        image,
        ice_detection
    )

    if cloud_detection:

        detections.append(
            cloud_detection
        )

        print(
            "[VISION] Cloud detected:",
            f"{cloud_detection['confidence']:.1f}%"
        )


    # ========================================================
    # 3. WATER
    # ========================================================

    water_detection = detect_water(
        image
    )

    if water_detection:

        detections.append(
            water_detection
        )

        print(
            "[VISION] Water detected:",
            f"{water_detection['confidence']:.1f}%"
        )


    # ========================================================
    # 4. CRATERS
    # ========================================================

    crater_detection = detect_craters(
        image
    )

    if crater_detection:

        detections.append(
            crater_detection
        )

        print(
            "[VISION] Crater detected:",
            f"{crater_detection['confidence']:.1f}%"
        )


    # ========================================================
    # 5. STORM
    # ========================================================

    storm_detection = detect_storm(
        image
    )

    if storm_detection:

        detections.append(
            storm_detection
        )

        print(
            "[VISION] Storm detected:",
            f"{storm_detection['confidence']:.1f}%"
        )


    # ========================================================
    # 6. OTHER SURFACE FEATURES
    # ========================================================

    for feature_id in [
        "forest",
        "urban",
        "desert",
        "mountains"
    ]:

        detection = detect_generic_feature(
            image,
            feature_id
        )

        if detection:

            detections.append(
                detection
            )


    # ========================================================
    # 7. FIRE
    # ========================================================

    fire_detection = detect_fire(
        image
    )

    if fire_detection:

        detections.append(
            fire_detection
        )


    # ========================================================
    # REMOVE DUPLICATE / CONFLICTING DETECTIONS
    # ========================================================

    # If water is extremely strong, don't allow ice to
    # dominate the same scene unless ice is clearly stronger.

    if (
        water_detection
        and
        ice_detection
    ):

        if (
            water_detection["confidence"]
            >
            ice_detection["confidence"] + 15
        ):

            detections = [
                d
                for d in detections
                if d["id"] != "ice"
            ]


    # ========================================================
    # SORT BY CONFIDENCE
    # ========================================================

    detections.sort(
        key=lambda x:
            x["confidence"],
        reverse=True
    )


    # Maximum number of displayed features.
    detections = detections[:8]


    print(
        f"[VISION] Final detections: {len(detections)}"
    )

    print()

    return detections


    # --------------------------------------------------------
    # CLOUDS SECOND
    #
    # Cloud detection now knows whether the same region
    # has strong surface-ice evidence.
    # --------------------------------------------------------

    cloud_detection = detect_clouds(
        image,
        ice_detection
    )


    if cloud_detection:

        detections.append(
            cloud_detection
        )

        print(
            "[VISION] Cloud detected:",
            f"{cloud_detection['confidence']:.1f}%"
        )


    # --------------------------------------------------------
    # STORM
    # --------------------------------------------------------

    storm_detection = detect_storm(
        image
    )


    if storm_detection:

        detections.append(
            storm_detection
        )

        print(
            "[VISION] Storm detected:",
            f"{storm_detection['confidence']:.1f}%"
        )


    # --------------------------------------------------------
    # OTHER SURFACE FEATURES
    # --------------------------------------------------------

    for feature_id in [
        "water",
        "forest",
        "urban",
        "desert",
        "mountains"
    ]:

        detection = detect_generic_feature(
            image,
            feature_id
        )


        if detection:

            detections.append(
                detection
            )


    # --------------------------------------------------------
    # FIRE
    # --------------------------------------------------------

    fire_detection = detect_fire(
        image
    )


    if fire_detection:

        detections.append(
            fire_detection
        )


    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    detections.sort(
        key=lambda x:
            x["confidence"],
        reverse=True
    )


    # Keep maximum 8 features.

    detections = detections[:8]


    return detections


# ============================================================
# OVERLAY GENERATION
# ============================================================

def overlay_features(
    image,
    detections
):

    base = np.array(
        image.convert("RGB")
    ).copy()


    overlay = base.copy()


    # Colors for different feature types.

    colors = {

    "ice":
        (80, 190, 255),

    "cloud":
        (80, 255, 210),

    "water":
        (50, 130, 255),

    "forest":
        (60, 255, 130),

    "urban":
        (255, 190, 50),

    "desert":
        (255, 170, 70),

    "mountains":
        (180, 130, 255),

    "crater":
        (255, 150, 60),

    "fire":
        (255, 80, 40),

    "storm":
        (255, 80, 180)
}


    for detection in detections:

        mask = detection.get(
            "mask"
        )


        if mask is None:
            continue


        feature_id = detection["id"]


        color = colors.get(
                feature_id,
                (80, 220, 255)
            )


        binary = (
            mask > 0.50
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Find contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


        # Ignore tiny contours.

        filtered_contours = []


        for contour in contours:

            area = cv2.contourArea(
                    contour
                )


            if area > 150:

                filtered_contours.append(
                    contour
                )


        # ----------------------------------------------------
        # Fill mask softly
        # ----------------------------------------------------

        color_array = np.zeros_like(
            base
        )


        color_array[:, :] = color


        alpha = (
            binary[..., None]
            * 0.18
        )


        overlay = (
            overlay * (1 - alpha)
            +
            color_array * alpha
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Draw contour
        # ----------------------------------------------------

        cv2.drawContours(
            overlay,
            filtered_contours,
            -1,
            color,
            3
        )


    result = Image.fromarray(
        overlay
    )


    return result


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def build_explanation(
    detections
):

    if not detections:

        return (
            "The image passed the basic quality and relevance "
            "checks, but the visual evidence was not strong "
            "enough to confidently identify a supported feature."
        )


    names = [
        detection["name"]
        for detection in detections
    ]


    statements = []


    if "Storm system" in names:

        statements.append(
            "A large organized atmospheric structure is visible."
        )


    if "Cloud formation" in names:

        statements.append(
            "Cloud cover is visible in the atmosphere."
        )


    if "Ice / snow" in names:

        statements.append(
            "A large surface region is visually consistent "
            "with snow or ice."
        )


    if "Open water" in names:

        statements.append(
            "A broad surface region is visually consistent "
            "with open water such as an ocean, sea, lake, or river."
        )


    if "Impact crater" in names:

        statements.append(
            "A roughly circular surface depression is visible "
            "and is visually consistent with an impact crater."
        )


    if "Forest / vegetation" in names:

        statements.append(
            "Vegetation or forest cover is visible."
        )


    if "Urban development" in names:

        statements.append(
            "Dense human development is visible."
        )


    if "Desert / dry terrain" in names:

        statements.append(
            "Dry or barren terrain is visible."
        )


    if "Mountain terrain" in names:

        statements.append(
            "Rugged or mountainous terrain is visible."
        )


    if "Fire / smoke plume" in names:

        statements.append(
            "A region may contain fire or smoke, "
            "but dedicated thermal/fire data would be "
            "needed for confirmation."
        )


    if not statements:

        statements.append(
            "The image contains visually distinctive "
            "Earth-surface or atmospheric structures."
        )


    explanation = " ".join(
            statements
        )


    explanation += (
        " EarthScope separates visible evidence "
        "from interpretation; the image alone cannot "
        "establish precise meteorological measurements "
        "such as wind speed, pressure, temperature, "
        "storm intensity, or an official storm category."
    )


    return explanation


# ============================================================
# ANALYSIS ENDPOINT
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    try:

        # ----------------------------------------------------
        # File exists?
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "status": "invalid",
                "message":
                    "No image was supplied."
            }), 400


        file = request.files["image"]


        if not file.filename:

            return jsonify({
                "status": "invalid",
                "message":
                    "No image was selected."
            }), 400


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        raw = file.read()


        image = Image.open(
                io.BytesIO(raw)
            ).convert(
                "RGB"
            )


        # ----------------------------------------------------
        # Quality gate
        # ----------------------------------------------------

        quality = check_image_quality(
                image
            )


        if not quality["valid"]:

            return jsonify({

                "status":
                    "unsuitable",

                "reason":
                    quality["reason"],

                "message":
                    quality["message"],

                "original_image":
                    image_to_data_url(
                        image
                    )
            })


        # ----------------------------------------------------
        # Earth relevance
        # ----------------------------------------------------

        relevance = check_relevance(
                image
            )


        print(
            f"[VISION] Relevance: {relevance}%"
        )


        if relevance < 28:

            return jsonify({

                "status":
                    "unsuitable",

                "reason":
                    "unrelated",

                "message":
                    "The uploaded image does not contain enough evidence of an Earth, planetary, or space observation.",

                "relevance":
                    relevance,

                "original_image":
                    image_to_data_url(
                        image
                    )
            })


        # ----------------------------------------------------
        # Feature detection
        # ----------------------------------------------------

        detections = detect_features(
                image
            )


        # ----------------------------------------------------
        # Remove internal masks from JSON
        # ----------------------------------------------------

        public_detections = []


        for detection in detections:

            public_detections.append({

                "id":
                    detection["id"],

                "name":
                    detection["name"],

                "description":
                    detection["description"],

                "confidence":
                    round(
                        detection["confidence"],
                        1
                    ),

                "area_percent":
                    detection["area_percent"],

                "type":
                    detection["type"]
            })


        # ----------------------------------------------------
        # Annotated image
        # ----------------------------------------------------

        annotated = overlay_features(
                image,
                detections
            )


        # ----------------------------------------------------
        # No confident features
        # ----------------------------------------------------

        if len(detections) == 0:

            return jsonify({

                "status":
                    "no_features",

                "message":
                    "The image is relevant, but no supported feature reached the confidence threshold.",

                "relevance":
                    relevance,

                "detections":
                    [],

                "explanation":
                    build_explanation(
                        []
                    ),

                "original_image":
                    image_to_data_url(
                        image
                    ),

                "annotated_image":
                    image_to_data_url(
                        annotated
                    ),

                "model_note":
                    "The system found an Earth/space observation but did not obtain enough visual evidence for a supported feature."
            })


        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        return jsonify({

            "status":
                "success",

            "message":
                "Observation successfully analyzed.",

            "relevance":
                relevance,

            "detections":
                public_detections,

            "explanation":
                build_explanation(
                    detections
                ),

            "original_image":
                image_to_data_url(
                    image
                ),

            "annotated_image":
                image_to_data_url(
                    annotated
                ),

            "model_note":
                "Detections are visual estimates generated by general-purpose vision models. They are not scientific measurements or official meteorological classifications."
        })


    except Exception as error:

        print()
        print("=" * 60)
        print("EARTHSCOPE SERVER ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)


        return jsonify({

            "status":
                "error",

            "message":
                "The observation engine encountered a server-side error."
        }), 500


# ============================================================
# MAIN
# ============================================================




if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )