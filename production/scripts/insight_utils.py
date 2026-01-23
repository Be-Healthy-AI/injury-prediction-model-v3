"""Shared utilities for production insight models and predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

BODY_PART_MAP: Dict[str, Tuple[str, ...]] = {
    "lower_leg": (
        "ankle",
        "foot",
        "toe",
        "achilles",
        "fibula",
        "tibia",
        "metatarsal",
        "peroneal",
        "calcaneus",
        "plantar",
        "heel",
        "shin",
        "sole",
        "talus",
        "navicular",
        "cuboid",
        "cuneiform",
        "phalanx",
        "sesamoid",
        "tendinitis",
        "tendonitis",
        "tendinopathy",
        "fascia",
        "fasciitis",
        "calf",
        "gastrocnemius",
        "soleus",
        "tibiotarsal",
        "syndesmosis",
        "ankle sprain",
        "ankle fracture",
        "foot sprain",
        "foot fracture",
        "toe fracture",
        "toe sprain",
        "peroneal tendon",
    ),
    "knee": (
        "knee",
        "patella",
        "meniscus",
        "acl",
        "pcl",
        "lcl",
        "mcl",
        "ligament",
        "cruciate",
        "patellar",
        "cartilage",
        "arthroscopy",
        "chondral",
        "osteochondral",
        "condyle",
        "bursitis",
        "prepatellar",
        "infrapatellar",
        "pes anserine",
        "knee sprain",
        "knee fracture",
    ),
    "upper_leg": (
        "thigh",
        "quad",
        "hamstring",
        "adductor",
        "quadriceps",
        "biceps femoris",
        "rectus femoris",
        "iliopsoas",
        "tensor fasciae latae",
        "sartorius",
        "gracilis",
        "semimembranosus",
        "semitendinosus",
        "vastus",
        "femoral",
        "muscle strain",
        "muscle tear",
        "muscle injury",
        "muscle problems",
        "muscle tension",
        "muscle fatigue",
        "muscle fibers",
        "breakdown of muscle fibers",
        "leg injury",
    ),
    "hip": (
        "hip",
        "pelvis",
        "pelvic",
        "pubis",
        "pubalgia",
        "glute",
        "gluteus",
        "lumbar",
        "lower back",
        "groin",
        "sacroiliac",
        "abdominal",
        "core",
        "piriformis",
        "iliac",
        "ischial",
        "coccyx",
        "tailbone",
        "osteitis pubis",
        "sports hernia",
        "inguinal",
        "psoas",
        "back problems",
        "back injury",
        "lumbago",
        "belly muscles",
    ),
    "upper_body": (
        "shoulder",
        "arm",
        "elbow",
        "wrist",
        "hand",
        "finger",
        "thumb",
        "clavicle",
        "humerus",
        "radius",
        "ulna",
        "scapula",
        "bicep",
        "tricep",
        "forearm",
        "rotator cuff",
        "labrum",
        "acromion",
        "sternoclavicular",
        "acromioclavicular",
        "carpal",
        "metacarpal",
        "tendon",
        "tendinitis",
        "tendonitis",
        "bursitis",
        "impingement",
        "neck",
        "cervical",
        "cervicalgia",
        "spine",
        "spinal",
        "vertebra",
        "vertebral",
        "disc",
        "herniated",
        "sciatica",
        "back",
        "thoracic",
        "sacral",
        "rib",
        "chest",
        "capsule injury",
        "tendon inflammation",
    ),
    "head": (
        "head",
        "face",
        "eye",
        "nose",
        "mouth",
        "concussion",
        "skull",
        "jaw",
        "cheekbone",
        "ear",
        "temple",
        "brain",
        "cranium",
        "facial",
        "orbital",
        "zygomatic",
        "maxilla",
        "mandible",
        "temporal",
        "frontal",
        "parietal",
        "occipital",
        "nasal",
        "dental",
        "tooth",
    ),
    "illness": (
        "illness",
        "sick",
        "fever",
        "cold",
        "flu",
        "virus",
        "infection",
        "influenza",
        "covid",
        "respiratory",
        "bronchitis",
        "pneumonia",
        "disease",
        "sars",
        "gastroenteritis",
        "gastric",
        "diarrhea",
        "nausea",
        "vomiting",
        "dehydration",
        "fatigue",
        "exhaustion",
        "shock",
        "heat stroke",
        "hypothermia",
    ),
}

SEVERITY_BINS = [0, 7, 28, 90, np.inf]
SEVERITY_LABELS = ["minor", "moderate", "severe", "long_term"]

# Original 8-level risk classification
RISK_THRESHOLDS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.01]
RISK_CLASS_LABELS = ["Very Low", "Low", "Moderate", "Guarded", "Elevated", "High", "Very High", "Critical"]
RISK_CLASS_COLORS = {
    "Very Low": "#2ecc71",      # Bright green
    "Low": "#27ae60",           # Green
    "Moderate": "#f39c12",      # Yellow-orange
    "Guarded": "#e67e22",       # Orange
    "Elevated": "#e74c3c",      # Red-orange
    "High": "#c0392b",          # Red
    "Very High": "#a93226",     # Dark red
    "Critical": "#7b241c",      # Very dark red
}

# 4-level risk classification (configurable)
RISK_THRESHOLDS_4LEVEL = [0.0, 0.3, 0.6, 0.8, 1.01]
RISK_CLASS_LABELS_4LEVEL = ["Low", "Medium", "High", "Very High"]
RISK_CLASS_COLORS_4LEVEL = {
    "Low": "#2ecc71",        # Green
    "Medium": "#f39c12",    # Yellow-orange
    "High": "#e74c3c",      # Red-orange
    "Very High": "#a93226", # Dark red
}


def normalize_text(text: str | None) -> str:
    return "" if text is None else str(text).lower()


def categorize_body_part(injury_type: str | None) -> str:
    injury_lower = normalize_text(injury_type)
    if not injury_lower:
        return "other"

    for label, keywords in BODY_PART_MAP.items():
        if any(keyword in injury_lower for keyword in keywords):
            return label
    return "other"


def severity_label(days: float | int | None) -> str | None:
    if days is None:
        return None
    if isinstance(days, str):
        extracted = "".join(ch for ch in days if (ch.isdigit() or ch == "."))
        days_value = float(extracted) if extracted else np.nan
    else:
        days_value = float(days)
    if np.isnan(days_value):
        return None
    idx = np.digitize([days_value], SEVERITY_BINS, right=False)[0] - 1
    if idx < 0 or idx >= len(SEVERITY_LABELS):
        return None
    return SEVERITY_LABELS[idx]


def load_pipeline(path_prefix: Path):
    model_path = path_prefix.with_suffix(".pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found: {model_path}")
    metadata_path = path_prefix.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), metadata


def rank_body_part_probabilities(probabilities: np.ndarray, classes: Iterable[str]) -> List[Tuple[str, float]]:
    pairs = list(zip(classes, probabilities))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


def format_severity_probabilities(probabilities: np.ndarray, classes: Iterable[str]) -> Dict[str, float]:
    return dict(zip(classes, probabilities))


def compute_shap_top_features(
    model,
    feature_df: pd.DataFrame,
    top_n: int = 10,
) -> List[List[Dict[str, float]]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    feature_names = feature_df.columns.to_list()
    top_features: List[List[Dict[str, float]]] = []
    for row_values in shap_values:
        indices = np.argsort(np.abs(row_values))[::-1][:top_n]
        features = [
            {
                "name": feature_names[idx],
                "value": float(row_values[idx]),
                "abs_value": float(abs(row_values[idx])),
            }
            for idx in indices
        ]
        top_features.append(features)
    return top_features


def classify_risk(probability: float) -> Dict[str, object]:
    """Original 8-level risk classification."""
    for idx, upper in enumerate(RISK_THRESHOLDS[1:], start=0):
        lower = RISK_THRESHOLDS[idx]
        if probability <= upper:
            label = RISK_CLASS_LABELS[idx]
            return {
                "index": idx,
                "label": label,
                "lower": lower,
                "upper": upper,
            }
    return {
        "index": len(RISK_CLASS_LABELS) - 1,
        "label": RISK_CLASS_LABELS[-1],
        "lower": RISK_THRESHOLDS[-2],
        "upper": RISK_THRESHOLDS[-1],
    }


def classify_risk_4level(probability: float, thresholds=None, labels=None) -> Dict[str, object]:
    """4-level risk classification with configurable thresholds."""
    if thresholds is None:
        thresholds = RISK_THRESHOLDS_4LEVEL
    if labels is None:
        labels = RISK_CLASS_LABELS_4LEVEL
    
    for idx, upper in enumerate(thresholds[1:], start=0):
        lower = thresholds[idx]
        if probability <= upper:
            label = labels[idx]
            return {
                "index": idx,
                "label": label,
                "lower": lower,
                "upper": upper,
            }
    return {
        "index": len(labels) - 1,
        "label": labels[-1],
        "lower": thresholds[-2],
        "upper": thresholds[-1],
    }


def compute_risk_series(probabilities: Iterable[float]) -> pd.DataFrame:
    probs = np.array(list(probabilities), dtype=float)
    labels = []
    indices = []
    lowers = []
    uppers = []
    for prob in probs:
        info = classify_risk(prob)
        indices.append(info["index"])
        labels.append(info["label"])
        lowers.append(info["lower"])
        uppers.append(info["upper"])
    return pd.DataFrame(
        {
            "probability": probs,
            "risk_index": indices,
            "risk_label": labels,
            "risk_lower": lowers,
            "risk_upper": uppers,
        }
    )


def compute_trend_metrics(probabilities: pd.Series) -> Dict[str, float]:
    series = probabilities.dropna()
    if series.empty:
        return {
            "max_jump": 0.0,
            "slope": 0.0,
            "sustained_elevated_days": 0,
            "final_deviation": 0.0,
            "final_probability": np.nan,
        }

    probs = series.values
    indices = np.arange(len(probs))
    baseline_window = min(10, len(probs))
    baseline = probs[:baseline_window].mean()
    std = probs.std() if len(probs) > 1 else 0.0
    max_jump = np.max(np.abs(np.diff(probs))) if len(probs) > 1 else 0.0
    slope = (
        float(np.polyfit(indices, probs, 1)[0]) if len(probs) > 1 else 0.0
    )
    final_prob = probs[-1]
    deviation = (final_prob - baseline) / (std + 1e-6)

    risk_df = compute_risk_series(probs)
    sustained = 0
    current = 0
    for idx in risk_df["risk_index"]:
        if idx >= 2:  # Elevated or higher
            current += 1
            sustained = max(sustained, current)
        else:
            current = 0

    return {
        "max_jump": float(max_jump),
        "slope": slope,
        "sustained_elevated_days": sustained,
        "final_deviation": float(deviation),
        "final_probability": float(final_prob),
        "baseline": float(baseline),
    }


def predict_insights(
    feature_row: pd.Series | None,
    bodypart_pipeline,
    severity_pipeline,
) -> Dict[str, object]:
    """Predict body part and severity insights from a feature row."""
    if feature_row is None:
        return {
            "bodypart_rank": [],
            "severity_probs": {},
            "severity_label": None,
        }
    X = feature_row.to_frame().T
    if hasattr(bodypart_pipeline, "feature_names_in_"):
        X = X.reindex(columns=bodypart_pipeline.feature_names_in_, fill_value=np.nan)

    body_probs = bodypart_pipeline.predict_proba(X)[0]
    severity_probs = severity_pipeline.predict_proba(X)[0]

    body_rank = rank_body_part_probabilities(body_probs, bodypart_pipeline.classes_)
    severity_dict = format_severity_probabilities(severity_probs, severity_pipeline.classes_)
    severity_label_val = severity_pipeline.classes_[int(np.argmax(severity_probs))]

    return {
        "bodypart_rank": body_rank,
        "severity_probs": severity_dict,
        "severity_label": severity_label_val,
    }

