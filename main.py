"""
main.py
=======
Real-time capture and prediction pipeline for the Attention Monitoring
System.

Responsibilities:
    - Open the webcam and run MediaPipe Face Landmarker on each frame.
    - Extract behavioural features via utils.py (EAR, MAR, head pose,
      gaze direction, blink/yawn counts).
    - Feed the assembled feature vector to ml_engine.py for real-time
      prediction of the attention class.
    - Compute a transparent attention score alongside the ML prediction.
    - Persist periodic measurements to SQLite via database.py.
    - Render a live annotated video window for demonstration purposes.

Requires:
    A downloaded face_landmarker.task model file in the project root.
    Get it with:
        curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

    (This project was migrated to MediaPipe's new Tasks API because the
    legacy `mp.solutions.face_mesh` API used in earlier versions of this
    file has been removed from recent MediaPipe releases.)

Run directly with:
    python main.py

Press 'q' in the video window to stop monitoring and close the session.
"""

import os
import time
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
import utils
import ml_engine
import database as db


FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)


def ensure_model_ready() -> None:
    """
    Guarantee a trained model bundle exists on disk before the monitoring
    loop starts. If no model has been trained yet, bootstrap one from the
    synthetic domain-informed dataset.
    """
    try:
        ml_engine.load_model()
        print(f"[INFO] Loaded existing model bundle from '{config.MODEL_PATH}'.")
    except FileNotFoundError:
        print("[INFO] No trained model found. Training a new model from the "
              "synthetic bootstrap dataset...")
        results = ml_engine.train_and_save_default_model()
        print(f"[INFO] Training complete. Best model: {results['best_model_name']}")
        for name, res in results["all_results"].items():
            print(
                f"        {name}: accuracy={res['accuracy']:.3f} "
                f"f1={res['f1_score']:.3f}"
            )


def ensure_face_landmarker_model() -> None:
    """
    Guarantee the MediaPipe Face Landmarker .task model file is present
    before the monitoring loop starts, and fail early with a clear message
    if it isn't.
    """
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            "Could not find 'face_landmarker.task' in the project directory.\n"
            "Download it with:\n"
            "    curl -o face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task\n"
            f"Expected location: {FACE_LANDMARKER_MODEL_PATH}"
        )


def create_face_landmarker() -> mp_vision.FaceLandmarker:
    """
    Build and return a MediaPipe FaceLandmarker configured for VIDEO mode,
    using the project's existing config thresholds.
    """
    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
    landmarker_options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=config.MAX_NUM_FACES,
        min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_face_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return mp_vision.FaceLandmarker.create_from_options(landmarker_options)


def extract_features_from_frame(
    landmarks,
    frame_width: int,
    frame_height: int,
    blink_detector: utils.BlinkDetector,
    yawn_detector: utils.YawnDetector,
    session_start_time: float,
) -> Dict[str, float]:
    """
    Run the full feature-extraction pipeline on one frame's face landmarks.

    Args:
        landmarks: MediaPipe normalized landmark list for the current frame
            (results.face_landmarks[0] from FaceLandmarker).
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        blink_detector: Stateful BlinkDetector instance for this session.
        yawn_detector: Stateful YawnDetector instance for this session.
        session_start_time: time.time() value captured when the session
            began, used to compute elapsed duration and blink rate.

    Returns:
        Dictionary of features in config.FEATURE_COLUMNS order, plus the
        raw (non-model) values needed for logging and display:
        "ear", "mar", "blink_count", "yawn_count", "eye_direction_label".
    """
    ear = utils.calculate_average_ear(landmarks, frame_width, frame_height)
    mar = utils.calculate_mar(landmarks, frame_width, frame_height)
    pitch, yaw, roll = utils.estimate_head_pose(landmarks, frame_width, frame_height)
    eye_direction_label, eye_direction_score = utils.estimate_eye_direction(
        landmarks, frame_width, frame_height
    )

    blink_detector.update(ear)
    yawn_detector.update(mar)

    elapsed_seconds = max(time.time() - session_start_time, 1e-6)
    blink_rate = (blink_detector.total_blinks / elapsed_seconds) * 60.0

    feature_dict = utils.build_feature_dict(
        ear=ear,
        mar=mar,
        blink_rate=blink_rate,
        yawn_count=yawn_detector.total_yawns,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        eye_direction_score=eye_direction_score,
        face_presence=1.0,
        session_duration=elapsed_seconds,
    )

    # Attach extra display/logging-only values not part of the model input.
    feature_dict["_blink_count"] = blink_detector.total_blinks
    feature_dict["_yawn_count"] = yawn_detector.total_yawns
    feature_dict["_eye_direction_label"] = eye_direction_label

    return feature_dict


def build_absent_face_features(
    blink_detector: utils.BlinkDetector,
    yawn_detector: utils.YawnDetector,
    session_start_time: float,
) -> Dict[str, float]:
    """
    Build a feature dictionary for frames where no face was detected, so
    the session can still log continuous data (face_presence = 0.0).

    Args:
        blink_detector: Stateful BlinkDetector instance for this session.
        yawn_detector: Stateful YawnDetector instance for this session.
        session_start_time: time.time() value captured when the session began.

    Returns:
        A feature dictionary matching extract_features_from_frame()'s shape,
        with neutral/zeroed CV values and face_presence = 0.0.
    """
    elapsed_seconds = max(time.time() - session_start_time, 1e-6)
    blink_rate = (blink_detector.total_blinks / elapsed_seconds) * 60.0

    feature_dict = utils.build_feature_dict(
        ear=0.0,
        mar=0.0,
        blink_rate=blink_rate,
        yawn_count=yawn_detector.total_yawns,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        eye_direction_score=0.5,
        face_presence=0.0,
        session_duration=elapsed_seconds,
    )
    feature_dict["_blink_count"] = blink_detector.total_blinks
    feature_dict["_yawn_count"] = yawn_detector.total_yawns
    feature_dict["_eye_direction_label"] = "unknown"

    return feature_dict


def run_monitoring_session() -> None:
    """
    Main entry point: open the webcam, run the real-time detection loop,
    display an annotated video feed, log periodic measurements to SQLite,
    and cleanly close the session when the user presses 'q'.
    """
    ensure_model_ready()
    ensure_face_landmarker_model()
    db.initialize_database()

    session_id = db.create_session()
    print(f"[INFO] Started new session (session_id={session_id}). Press 'q' to stop.")

    face_landmarker = create_face_landmarker()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX in config.py / .env.")
        db.end_session(session_id)
        return

    blink_detector = utils.BlinkDetector()
    yawn_detector = utils.YawnDetector()
    session_start_time = time.time()
    last_log_time = 0.0
    last_timestamp_ms = -1

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] Failed to read frame from webcam. Retrying...")
                continue

            # Mirror the frame for a natural "selfie view" during the demo.
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # FaceLandmarker in VIDEO mode requires a strictly increasing
            # timestamp (ms) per frame.
            timestamp_ms = int((time.time() - session_start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                feature_dict = extract_features_from_frame(
                    landmarks, frame_width, frame_height,
                    blink_detector, yawn_detector, session_start_time,
                )
                utils.draw_face_mesh_points(frame, landmarks, frame_width, frame_height)
            else:
                feature_dict = build_absent_face_features(
                    blink_detector, yawn_detector, session_start_time
                )

            # Separate model-input features from display-only metadata.
            model_features = {k: v for k, v in feature_dict.items() if not k.startswith("_")}

            predicted_class, confidence, _ = ml_engine.predict(model_features)

            attention_score = utils.calculate_attention_score(
                ear=model_features["ear"],
                gaze_score=model_features["eye_direction_score"],
                pitch=model_features["pitch"],
                yaw=model_features["yaw"],
                blink_rate=model_features["blink_rate"],
                yawn_count=model_features["yawn_count"],
            )

            utils.draw_annotations(
                frame,
                prediction=predicted_class,
                attention_score=attention_score,
                ear=model_features["ear"],
                mar=model_features["mar"],
                blink_count=feature_dict["_blink_count"],
                yawn_count=feature_dict["_yawn_count"],
            )

            cv2.imshow("Student Attention Monitoring System - Press 'q' to quit", frame)

            # Log to database at a fixed interval rather than every frame,
            # to avoid flooding SQLite with near-duplicate rows.
            now = time.time()
            if now - last_log_time >= config.LOGGING_INTERVAL_SECONDS:
                db.insert_attention_log(
                    session_id=session_id,
                    prediction=predicted_class,
                    attention_score=attention_score,
                    ear=model_features["ear"],
                    mar=model_features["mar"],
                    blink_count=feature_dict["_blink_count"],
                    yawn_count=feature_dict["_yawn_count"],
                    pitch=model_features["pitch"],
                    yaw=model_features["yaw"],
                    roll=model_features["roll"],
                    eye_direction=feature_dict["_eye_direction_label"],
                    session_duration=model_features["session_duration"],
                )
                last_log_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] 'q' pressed. Ending session...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        db.end_session(session_id)
        print(f"[INFO] Session {session_id} ended and saved to database.")


if __name__ == "__main__":
    run_monitoring_session()