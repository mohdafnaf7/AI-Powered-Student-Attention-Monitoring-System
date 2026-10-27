"""
utils.py
========
Pure computer-vision and signal-processing helper functions for the
Attention Monitoring System.

This module has NO knowledge of the database, the ML model, or Streamlit.
It only knows how to turn MediaPipe Face Mesh landmarks into numeric
behavioural features (EAR, MAR, head pose, gaze direction, blink/yawn
counts) and how to combine those features into a single attention score.

All thresholds and landmark indices are imported from config.py so that
tuning the system never requires touching this file.
"""

import math
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

import config


# =========================================================================
# Basic geometry helpers
# =========================================================================
def euclidean_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two 2D points.

    Args:
        point_a: (x, y) coordinate.
        point_b: (x, y) coordinate.

    Returns:
        The straight-line distance between the two points.
    """
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def landmarks_to_pixel_coords(
    landmarks, indices: List[int], frame_width: int, frame_height: int
) -> List[Tuple[float, float]]:
    """
    Convert a subset of normalized MediaPipe landmarks to pixel coordinates.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList.landmark (iterable of
            landmarks with .x, .y in [0, 1]).
        indices: Landmark indices to extract.
        frame_width: Width of the source frame in pixels.
        frame_height: Height of the source frame in pixels.

    Returns:
        List of (x, y) pixel coordinates in the same order as `indices`.
    """
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((lm.x * frame_width, lm.y * frame_height))
    return coords


# =========================================================================
# Eye Aspect Ratio (EAR)
# =========================================================================
def calculate_ear(
    landmarks, eye_indices: List[int], frame_width: int, frame_height: int
) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

    A high EAR indicates an open eye; EAR drops sharply toward zero when
    the eye closes (blink).

    Args:
        landmarks: MediaPipe normalized landmark list for the current frame.
        eye_indices: 6 landmark indices [P1..P6] defining the eye contour,
            ordered per the classic Soukupova & Cech EAR formulation.
        frame_width: Frame width in pixels (for de-normalization).
        frame_height: Frame height in pixels (for de-normalization).

    Returns:
        The Eye Aspect Ratio as a float. Returns 0.0 if the eye is
        degenerate (zero horizontal distance) to avoid division by zero.
    """
    pts = landmarks_to_pixel_coords(landmarks, eye_indices, frame_width, frame_height)
    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_average_ear(landmarks, frame_width: int, frame_height: int) -> float:
    """
    Calculate the average EAR across both eyes for a stable single value.

    Args:
        landmarks: MediaPipe normalized landmark list for the current frame.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        The mean of the left-eye and right-eye EAR values.
    """
    left_ear = calculate_ear(landmarks, config.LEFT_EYE_EAR_IDX, frame_width, frame_height)
    right_ear = calculate_ear(landmarks, config.RIGHT_EYE_EAR_IDX, frame_width, frame_height)
    return (left_ear + right_ear) / 2.0


# =========================================================================
# Mouth Aspect Ratio (MAR)
# =========================================================================
def calculate_mar(landmarks, frame_width: int, frame_height: int) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR), used to detect yawning.

    MAR = (||top_lip - bottom_lip|| average) / ||mouth_corner_to_corner||

    Args:
        landmarks: MediaPipe normalized landmark list for the current frame.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        The Mouth Aspect Ratio as a float. Returns 0.0 for a degenerate
        (zero-width) mouth region.
    """
    idx = config.MOUTH_MAR_IDX
    pts = landmarks_to_pixel_coords(landmarks, idx, frame_width, frame_height)
    left_corner, right_corner, top_1, bottom_1, top_2, bottom_2, top_3, bottom_3 = pts

    vertical_1 = euclidean_distance(top_1, bottom_1)
    vertical_2 = euclidean_distance(top_2, bottom_2)
    vertical_3 = euclidean_distance(top_3, bottom_3)
    horizontal = euclidean_distance(left_corner, right_corner)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)


# =========================================================================
# Head Pose Estimation
# =========================================================================
# Generic 3D face model points (in millimetres, arbitrary reference frame)
# corresponding to config.HEAD_POSE_LANDMARK_IDX
# [nose tip, chin, left-eye-outer, right-eye-outer, mouth-left, mouth-right].
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye, left corner
        (225.0, 170.0, -135.0),   # Right eye, right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ],
    dtype=np.float64,
)


def estimate_head_pose(
    landmarks, frame_width: int, frame_height: int
) -> Tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) in degrees using solvePnP.

    Args:
        landmarks: MediaPipe normalized landmark list for the current frame.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        A tuple (pitch, yaw, roll) in degrees. Returns (0.0, 0.0, 0.0) if
        pose estimation fails numerically.
    """
    image_points = np.array(
        landmarks_to_pixel_coords(landmarks, config.HEAD_POSE_LANDMARK_IDX, frame_width, frame_height),
        dtype=np.float64,
    )

    # Approximate camera intrinsics (no calibration file available; this
    # pinhole approximation is standard practice for webcam-based pose
    # estimation and is accurate enough for coarse attention classification).
    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion.

    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

    # Decompose rotation matrix into Euler angles (pitch, yaw, roll).
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    return pitch_deg, yaw_deg, roll_deg


# =========================================================================
# Eye Gaze / Direction Estimation
# =========================================================================
def estimate_eye_direction(
    landmarks, frame_width: int, frame_height: int
) -> Tuple[str, float]:
    """
    Estimate horizontal eye gaze direction using iris landmark position
    relative to the eye corners.

    Requires MediaPipe Face Mesh to be initialized with
    refine_landmarks=True so iris landmarks (468-477) are available.

    Args:
        landmarks: MediaPipe normalized landmark list for the current frame.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        A tuple (direction_label, direction_score) where direction_label is
        one of "left", "center", "right", and direction_score is a
        normalized value in [0, 1] where 0.5 means perfectly centered and
        values near 0 or 1 mean looking far left/right. Returns
        ("center", 0.5) if iris landmarks are unavailable.
    """
    try:
        left_iris = landmarks_to_pixel_coords(landmarks, config.LEFT_IRIS_IDX, frame_width, frame_height)
        left_corners = landmarks_to_pixel_coords(
            landmarks, config.LEFT_EYE_CORNERS_IDX, frame_width, frame_height
        )
    except IndexError:
        # Iris landmarks not present (refine_landmarks was False).
        return "center", 0.5

    iris_center_x = np.mean([p[0] for p in left_iris])
    eye_left_x, eye_right_x = left_corners[0][0], left_corners[1][0]

    eye_width = eye_right_x - eye_left_x
    if eye_width == 0:
        return "center", 0.5

    # Normalize iris position within the eye's horizontal span [0, 1].
    relative_pos = (iris_center_x - eye_left_x) / eye_width
    relative_pos = float(np.clip(relative_pos, 0.0, 1.0))

    if relative_pos < config.GAZE_LEFT_BOUND:
        label = "right"  # Mirrored: webcam feed is naturally flipped for the viewer.
    elif relative_pos > config.GAZE_RIGHT_BOUND:
        label = "left"
    else:
        label = "center"

    return label, relative_pos


# =========================================================================
# Blink Detection (stateful)
# =========================================================================
class BlinkDetector:
    """
    Stateful blink counter using a consecutive-frames-below-threshold rule.

    A blink is only counted once the eye has been below EAR_BLINK_THRESHOLD
    for at least EAR_CONSEC_FRAMES frames AND has since reopened, which
    filters out single-frame landmark jitter from being miscounted as
    a blink.
    """

    def __init__(self) -> None:
        """Initialize the blink detector with a zeroed internal state."""
        self._consec_low_frames: int = 0
        self.total_blinks: int = 0

    def update(self, ear: float) -> bool:
        """
        Feed the current frame's EAR value into the detector.

        Args:
            ear: The current frame's Eye Aspect Ratio.

        Returns:
            True if a new blink was just registered on this call, else False.
        """
        blink_registered = False

        if ear < config.EAR_BLINK_THRESHOLD:
            self._consec_low_frames += 1
        else:
            if self._consec_low_frames >= config.EAR_CONSEC_FRAMES:
                self.total_blinks += 1
                blink_registered = True
            self._consec_low_frames = 0

        return blink_registered

    def reset(self) -> None:
        """Reset the blink counter and internal state (e.g., new session)."""
        self._consec_low_frames = 0
        self.total_blinks = 0


# =========================================================================
# Yawn Detection (stateful)
# =========================================================================
class YawnDetector:
    """
    Stateful yawn counter using a consecutive-frames-above-threshold rule.

    A yawn is counted once the mouth has stayed open (MAR above threshold)
    for at least MAR_CONSEC_FRAMES frames and then closes again.
    """

    def __init__(self) -> None:
        """Initialize the yawn detector with a zeroed internal state."""
        self._consec_high_frames: int = 0
        self.total_yawns: int = 0

    def update(self, mar: float) -> bool:
        """
        Feed the current frame's MAR value into the detector.

        Args:
            mar: The current frame's Mouth Aspect Ratio.

        Returns:
            True if a new yawn was just registered on this call, else False.
        """
        yawn_registered = False

        if mar > config.MAR_YAWN_THRESHOLD:
            self._consec_high_frames += 1
        else:
            if self._consec_high_frames >= config.MAR_CONSEC_FRAMES:
                self.total_yawns += 1
                yawn_registered = True
            self._consec_high_frames = 0

        return yawn_registered

    def reset(self) -> None:
        """Reset the yawn counter and internal state (e.g., new session)."""
        self._consec_high_frames = 0
        self.total_yawns = 0


# =========================================================================
# Attention Score
# =========================================================================
def calculate_attention_score(
    ear: float,
    gaze_score: float,
    pitch: float,
    yaw: float,
    blink_rate: float,
    yawn_count: int,
) -> float:
    """
    Combine behavioural signals into a single 0-100 attention score using
    the weights defined in config.ATTENTION_SCORE_WEIGHTS.

    Sub-score definitions:
        eye_openness   -> EAR normalized against the blink threshold, capped at 1.0.
        gaze_center    -> 1.0 at gaze_score == 0.5 (dead center), decaying to 0
                          at the extremes (0.0 or 1.0).
        head_stability -> 1.0 when pitch/yaw are within their configured
                          thresholds, decaying linearly to 0 beyond 2x threshold.
        low_blink_rate -> 1.0 at 0 blinks/min, decaying as blink_rate rises
                          past a normal resting rate (~20 blinks/min).
        low_yawn_rate  -> 1.0 with no yawns, decaying with each additional yawn.

    Args:
        ear: Current (averaged) Eye Aspect Ratio.
        gaze_score: Normalized horizontal gaze position in [0, 1]
            (0.5 = centered).
        pitch: Head pitch angle in degrees.
        yaw: Head yaw angle in degrees.
        blink_rate: Blinks per minute (already normalized externally).
        yawn_count: Total yawns observed in the current session.

    Returns:
        Attention score as a float in the range [0.0, 100.0].
    """
    weights = config.ATTENTION_SCORE_WEIGHTS

    # Eye openness: EAR of ~0.30+ is a fully open eye; scale relative to
    # the blink threshold so a closed eye scores near 0.
    eye_openness = np.clip(ear / 0.30, 0.0, 1.0)

    # Gaze centering: distance from perfect center (0.5), scaled to [0, 1].
    gaze_center = 1.0 - np.clip(abs(gaze_score - 0.5) / 0.5, 0.0, 1.0)

    # Head stability: penalize pitch/yaw deviation beyond configured thresholds.
    pitch_penalty = np.clip(abs(pitch) / (config.HEAD_PITCH_THRESHOLD * 2.0), 0.0, 1.0)
    yaw_penalty = np.clip(abs(yaw) / (config.HEAD_YAW_THRESHOLD * 2.0), 0.0, 1.0)
    head_stability = 1.0 - max(pitch_penalty, yaw_penalty)

    # Blink rate penalty: a healthy resting rate is ~15-20 blinks/min;
    # excessive blinking suggests fatigue/distraction.
    low_blink_rate = 1.0 - np.clip(blink_rate / 40.0, 0.0, 1.0)

    # Yawn penalty: each yawn reduces the sub-score, floored at 0.
    low_yawn_rate = 1.0 - np.clip(yawn_count / 5.0, 0.0, 1.0)

    score = (
        weights["eye_openness"] * eye_openness
        + weights["gaze_center"] * gaze_center
        + weights["head_stability"] * head_stability
        + weights["low_blink_rate"] * low_blink_rate
        + weights["low_yawn_rate"] * low_yawn_rate
    )

    return round(float(np.clip(score * 100.0, 0.0, 100.0)), 2)


def score_to_class(score: float) -> str:
    """
    Map a continuous attention score to a discrete class label.

    Args:
        score: Attention score in [0, 100].

    Returns:
        One of "Highly Attentive", "Passive", "Distracted".
    """
    if score >= config.SCORE_HIGH_THRESHOLD:
        return "Highly Attentive"
    if score < config.SCORE_LOW_THRESHOLD:
        return "Distracted"
    return "Passive"


# =========================================================================
# Drawing / Annotation Helpers
# =========================================================================
def draw_annotations(
    frame: np.ndarray,
    prediction: str,
    attention_score: float,
    ear: float,
    mar: float,
    blink_count: int,
    yawn_count: int,
) -> np.ndarray:
    """
    Draw a heads-up-display overlay of current metrics onto the video frame.

    Args:
        frame: BGR image (as returned by OpenCV) to annotate in place.
        prediction: Predicted attention class label.
        attention_score: Current attention score (0-100).
        ear: Current Eye Aspect Ratio.
        mar: Current Mouth Aspect Ratio.
        blink_count: Total blinks so far this session.
        yawn_count: Total yawns so far this session.

    Returns:
        The annotated frame (same array, modified in place, also returned
        for convenient chaining).
    """
    color_map = {
        "Highly Attentive": (46, 204, 113),   # BGR green
        "Passive": (19, 196, 241),            # BGR amber
        "Distracted": (60, 76, 231),          # BGR red
    }
    color = color_map.get(prediction, (255, 255, 255))

    overlay_lines = [
        f"Prediction: {prediction}",
        f"Attention Score: {attention_score:.1f}",
        f"EAR: {ear:.3f}  MAR: {mar:.3f}",
        f"Blinks: {blink_count}  Yawns: {yawn_count}",
    ]

    # Semi-transparent background box for readability.
    box_h = 25 * len(overlay_lines) + 15
    sub_img = frame[0:box_h, 0:320]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    frame[0:box_h, 0:320] = cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 0)

    for i, line in enumerate(overlay_lines):
        y = 30 + i * 25
        text_color = color if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return frame


def draw_face_mesh_points(frame: np.ndarray, landmarks, frame_width: int, frame_height: int) -> np.ndarray:
    """
    Draw small circles on key landmarks (eyes, mouth) used in feature
    extraction, for visual debugging / demo purposes.

    Args:
        frame: BGR image to annotate in place.
        landmarks: MediaPipe normalized landmark list for the current frame.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        The annotated frame.
    """
    key_indices = (
        config.LEFT_EYE_EAR_IDX
        + config.RIGHT_EYE_EAR_IDX
        + config.MOUTH_MAR_IDX
    )
    points = landmarks_to_pixel_coords(landmarks, key_indices, frame_width, frame_height)
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
    return frame


# =========================================================================
# Feature vector assembly
# =========================================================================
def build_feature_dict(
    ear: float,
    mar: float,
    blink_rate: float,
    yawn_count: int,
    pitch: float,
    yaw: float,
    roll: float,
    eye_direction_score: float,
    face_presence: float,
    session_duration: float,
) -> Dict[str, float]:
    """
    Assemble a single feature dictionary in the exact column order expected
    by ml_engine.py and database.py (see config.FEATURE_COLUMNS).

    Args:
        ear: Averaged Eye Aspect Ratio.
        mar: Mouth Aspect Ratio.
        blink_rate: Blinks per minute.
        yawn_count: Total yawns so far this session.
        pitch: Head pitch angle in degrees.
        yaw: Head yaw angle in degrees.
        roll: Head roll angle in degrees.
        eye_direction_score: Normalized gaze position in [0, 1].
        face_presence: 1.0 if a face was detected this frame, else 0.0.
        session_duration: Elapsed session time in seconds.

    Returns:
        Dictionary mapping each name in config.FEATURE_COLUMNS to its value.
    """
    values = {
        "ear": ear,
        "mar": mar,
        "blink_rate": blink_rate,
        "yawn_count": yawn_count,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "eye_direction_score": eye_direction_score,
        "face_presence": face_presence,
        "session_duration": session_duration,
    }
    # Guarantee consistent ordering via config.FEATURE_COLUMNS.
    return {col: values[col] for col in config.FEATURE_COLUMNS}