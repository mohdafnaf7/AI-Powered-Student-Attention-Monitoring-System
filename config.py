"""
config.py
=========
Central configuration module for the AI-Powered Real-Time Student Attention
and Engagement Monitoring System.

This file contains ONLY constants and settings. No business logic lives here.
Every other module (utils.py, ml_engine.py, database.py, main.py,
streamlit_app.py) imports its configuration from this single source of truth
so that thresholds, paths, and weights never need to be edited in more than
one place.
"""

import os
from dotenv import load_dotenv

# -----------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------
# Load variables defined in .env (if present) into the process environment.
# This allows deployment-specific overrides without touching source code.
load_dotenv()


def _env_float(key: str, default: float) -> float:
    """
    Read a float value from the environment, falling back to a default.

    Args:
        key: Environment variable name.
        default: Value to use if the variable is missing or invalid.

    Returns:
        The parsed float value.
    """
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    """
    Read an integer value from the environment, falling back to a default.

    Args:
        key: Environment variable name.
        default: Value to use if the variable is missing or invalid.

    Returns:
        The parsed integer value.
    """
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# SQLite database file (created automatically on first run).
DATABASE_NAME: str = os.getenv("DATABASE_NAME", "attention_monitor.db")
DATABASE_PATH: str = os.path.join(BASE_DIR, DATABASE_NAME)

# Trained model bundle (model + scaler + label encoder), saved via joblib.
MODEL_DIR: str = BASE_DIR
MODEL_FILE_NAME: str = os.getenv("MODEL_FILE_NAME", "attention_model.joblib")
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

# Directory for exported CSV reports.
EXPORT_DIR: str = os.path.join(BASE_DIR, "exports")

# -----------------------------------------------------------------------
# Webcam / Computer Vision settings
# -----------------------------------------------------------------------
CAMERA_INDEX: int = _env_int("CAMERA_INDEX", 0)
FRAME_WIDTH: int = _env_int("FRAME_WIDTH", 640)
FRAME_HEIGHT: int = _env_int("FRAME_HEIGHT", 480)

# MediaPipe Face Mesh configuration.
MAX_NUM_FACES: int = 1
MIN_DETECTION_CONFIDENCE: float = 0.5
MIN_TRACKING_CONFIDENCE: float = 0.5
REFINE_LANDMARKS: bool = True  # Enables iris landmarks for gaze estimation.

# -----------------------------------------------------------------------
# Facial landmark indices (MediaPipe Face Mesh - 468/478 point model)
# -----------------------------------------------------------------------
# Left/right eye contour points used for Eye Aspect Ratio (EAR).
# Ordered as: [P1, P2, P3, P4, P5, P6] per the classic EAR formulation.
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]

# Mouth contour points used for Mouth Aspect Ratio (MAR).
MOUTH_MAR_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

# Iris center landmarks (requires refine_landmarks=True).
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

# Eye corner landmarks used as bounds for gaze-direction normalization.
LEFT_EYE_CORNERS_IDX = [362, 263]
RIGHT_EYE_CORNERS_IDX = [33, 133]

# 6 stable landmark points + matching 3D model points used for head pose
# estimation via cv2.solvePnP (nose tip, chin, eye corners, mouth corners).
HEAD_POSE_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# -----------------------------------------------------------------------
# Behavioural thresholds
# -----------------------------------------------------------------------
# Eye Aspect Ratio below this value indicates a closed eye (blink).
EAR_BLINK_THRESHOLD: float = _env_float("EAR_BLINK_THRESHOLD", 0.21)

# Number of consecutive frames the eye must stay below threshold to count
# as one valid blink (filters out single-frame noise).
EAR_CONSEC_FRAMES: int = _env_int("EAR_CONSEC_FRAMES", 2)

# Mouth Aspect Ratio above this value indicates a yawn.
MAR_YAWN_THRESHOLD: float = _env_float("MAR_YAWN_THRESHOLD", 0.60)

# Number of consecutive frames the mouth must stay above threshold to
# count as one valid yawn.
MAR_CONSEC_FRAMES: int = _env_int("MAR_CONSEC_FRAMES", 8)

# Head pose angles (degrees) beyond which the student is considered
# to be looking away from the screen.
HEAD_YAW_THRESHOLD: float = _env_float("HEAD_YAW_THRESHOLD", 25.0)
HEAD_PITCH_THRESHOLD: float = _env_float("HEAD_PITCH_THRESHOLD", 20.0)

# Normalized horizontal iris position bounds used to classify gaze
# direction as "left" / "center" / "right".
GAZE_LEFT_BOUND: float = 0.38
GAZE_RIGHT_BOUND: float = 0.62

# -----------------------------------------------------------------------
# Attention score weights
# -----------------------------------------------------------------------
# The attention score (0-100) is a weighted combination of behavioural
# signals. Weights must sum to 1.0. Tuned so that eye openness and gaze
# direction dominate (strongest attention indicators), while blink/yawn
# frequency and head stability act as penalising factors.
ATTENTION_SCORE_WEIGHTS: dict = {
    "eye_openness": 0.30,      # Derived from EAR relative to blink threshold.
    "gaze_center": 0.25,       # 1.0 if looking at screen center, decays outward.
    "head_stability": 0.20,    # 1.0 if pitch/yaw within threshold, decays outward.
    "low_blink_rate": 0.15,    # Penalises excessive blinking.
    "low_yawn_rate": 0.10,     # Penalises yawning (fatigue indicator).
}

# Score thresholds used to map a continuous attention score to a class
# label. These MUST be consistent with the ML model's training labels.
SCORE_HIGH_THRESHOLD: float = 70.0   # score >= 70 -> Highly Attentive
SCORE_LOW_THRESHOLD: float = 40.0    # score < 40 -> Distracted (else Passive)

# -----------------------------------------------------------------------
# Prediction classes
# -----------------------------------------------------------------------
ATTENTION_CLASSES = ["Distracted", "Passive", "Highly Attentive"]
CLASS_COLOR_MAP = {
    "Highly Attentive": "#2ECC71",  # green
    "Passive": "#F1C40F",           # amber
    "Distracted": "#E74C3C",        # red
}

# -----------------------------------------------------------------------
# Machine Learning settings
# -----------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
SYNTHETIC_SAMPLES_PER_CLASS: int = _env_int("SYNTHETIC_SAMPLES_PER_CLASS", 600)

# Feature columns used by the ML model, in a fixed, guaranteed order.
# ALL modules (training, prediction, database) must respect this order.
FEATURE_COLUMNS = [
    "ear",
    "mar",
    "blink_rate",
    "yawn_count",
    "pitch",
    "yaw",
    "roll",
    "eye_direction_score",
    "face_presence",
    "session_duration",
]

# -----------------------------------------------------------------------
# Session / logging settings
# -----------------------------------------------------------------------
# How often (in seconds) main.py writes a new row to attention_logs.
LOGGING_INTERVAL_SECONDS: float = _env_float("LOGGING_INTERVAL_SECONDS", 2.0)

# Frames-per-second assumption used for blink-rate/yawn-rate normalization
# when the actual camera FPS cannot be measured reliably.
ASSUMED_FPS: int = _env_int("ASSUMED_FPS", 20)

# -----------------------------------------------------------------------
# Streamlit dashboard settings
# -----------------------------------------------------------------------
APP_TITLE: str = "AI-Powered Student Attention Monitoring System"
APP_ICON: str = "🎓"
PAGE_LAYOUT: str = "wide"
DASHBOARD_REFRESH_SECONDS: float = _env_float("DASHBOARD_REFRESH_SECONDS", 1.0)