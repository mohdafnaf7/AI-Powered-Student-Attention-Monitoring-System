# 🎓 AI-Powered Student Attention Monitoring System

A real-time computer vision system that monitors student attention and engagement during study or class sessions using webcam-based facial analysis, and classifies attentiveness with a trained machine learning model. Includes a full Streamlit dashboard for live monitoring, historical analytics, and model performance tracking.

---

## Features

- **Real-time facial analysis** via MediaPipe's `FaceLandmarker` (Tasks API) — eye aspect ratio (EAR), mouth aspect ratio (MAR), head pose (pitch/yaw/roll), and gaze/eye-direction estimation.
- **Blink and yawn detection** with consecutive-frame filtering to reduce noise.
- **ML-based attention classification** into `Highly Attentive`, `Passive`, or `Distracted`, using a Random Forest classifier (compared against Decision Tree and Logistic Regression, auto-selected by weighted F1 score).
- **Transparent rule-based attention score (0–100)** computed alongside the ML prediction, as a weighted combination of eye openness, gaze centering, head stability, blink rate, and yawn rate.
- **Live annotated video window** (OpenCV) showing face mesh points, prediction, and score during a monitoring session.
- **SQLite persistence** of sessions and periodic attention logs.
- **5-tab Streamlit dashboard**: Live Attention Monitor, Session History & Analytics, Model Performance, and System Configuration.
- **Retraining pipeline** that can blend real accumulated session data with the synthetic bootstrap dataset.

---

## Project Structure

```
.
├── main.py               # Real-time capture, feature extraction & prediction loop
├── streamlit_app.py      # Streamlit dashboard (live monitor, analytics, model info, settings)
├── ml_engine.py           # Synthetic data generation, training, evaluation, prediction, retraining
├── utils.py               # Feature extraction helpers (EAR, MAR, head pose, gaze, attention score)
├── database.py             # SQLite schema, session/log CRUD, analytics queries, CSV export
├── config.py               # Central configuration: paths, thresholds, weights, feature order
├── requirements.txt        # Python dependencies
├── .env                     # Environment variable overrides (see below)
├── attention_monitor.db     # SQLite database (auto-created on first run)
├── attention_model.joblib   # Trained model bundle (auto-created on first run)
└── face_landmarker.task     # MediaPipe Face Landmarker model (must be downloaded, see below)
```

---

## How It Works

1. **Capture** — `main.py` opens the webcam and runs MediaPipe's `FaceLandmarker` (VIDEO mode) on each frame.
2. **Feature extraction** — `utils.py` computes EAR, MAR, head pose (via `cv2.solvePnP`), and normalized gaze direction from the 478-point face landmark set.
3. **Stateful detectors** — `BlinkDetector` and `YawnDetector` track consecutive-frame thresholds to count real blinks/yawns rather than single-frame noise.
4. **Prediction** — the assembled feature vector (in the fixed order defined by `config.FEATURE_COLUMNS`) is scaled and passed to the trained model in `ml_engine.py` to predict one of three attention classes with a confidence score.
5. **Scoring** — `utils.calculate_attention_score()` independently computes a transparent 0–100 score from the same signals, using the weights defined in `config.ATTENTION_SCORE_WEIGHTS`.
6. **Logging** — every `LOGGING_INTERVAL_SECONDS`, a row is written to the `attention_logs` table in SQLite (`database.py`), linked to a `sessions` record.
7. **Dashboard** — `streamlit_app.py` reads from the same database and model bundle to show live status, historical trends, and model diagnostics.

### On training data

The classifier is bootstrapped from a **synthetic, domain-informed dataset** (`ml_engine.generate_training_dataset()`), since a brand-new session has no ground-truth attentiveness labels. Feature distributions per class are tuned from established attention/fatigue heuristics (e.g. attentive students have high EAR, centered gaze, stable head pose, low blink/yawn rates), with realistic noise and class overlap so the model isn't trivially perfect. `ml_engine.retrain_from_logs()` allows retraining later on real accumulated session data, optionally blended with fresh synthetic samples for stability.

---

## Requirements

- Python 3.11 (developed/tested on Windows)
- A working webcam
- Dependencies listed in `requirements.txt`:
  - `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`
  - `streamlit>=1.33.0`, `plotly`, `matplotlib`
  - `joblib`, `python-dotenv`

---

## Setup

**1. Clone/copy the project and create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the MediaPipe Face Landmarker model**

The legacy `mp.solutions.face_mesh` API has been removed from recent MediaPipe releases, so this project uses the newer Tasks API, which requires a separately downloaded model file placed in the project root:

```bash
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**4. (Optional) Configure environment variables**

Copy/edit `.env` to override any default in `config.py` without touching source code:

```env
DATABASE_NAME=attention_monitor.db
MODEL_FILE_NAME=attention_model.joblib

CAMERA_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480

EAR_BLINK_THRESHOLD=0.21
EAR_CONSEC_FRAMES=2
MAR_YAWN_THRESHOLD=0.60
MAR_CONSEC_FRAMES=8
HEAD_YAW_THRESHOLD=25.0
HEAD_PITCH_THRESHOLD=20.0

SYNTHETIC_SAMPLES_PER_CLASS=600

LOGGING_INTERVAL_SECONDS=2.0
ASSUMED_FPS=20

DASHBOARD_REFRESH_SECONDS=1.0
```

> On first run, if no trained model exists at `MODEL_PATH`, the system automatically bootstraps one from the synthetic dataset — no manual training step is required.

---

## Usage

### Run the real-time monitor (webcam + OpenCV window)

```bash
python main.py
```

- Opens a webcam feed with live face mesh overlay, predicted attention class, and attention score.
- Logs a new row to SQLite every `LOGGING_INTERVAL_SECONDS`.
- Press **`q`** in the video window to end the session (the session end time is saved to the database).

### Launch the dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard has four sections, selectable from the sidebar:

| Page | Description |
|---|---|
| 🖥️ **Live Attention Monitor** | Real-time status and current session view |
| 📊 **Session History & Analytics** | Past sessions, trends, and exportable logs |
| 🧠 **Model Performance** | Model name, evaluation metrics, and feature importance |
| ⚙️ **System Configuration** | View/adjust active thresholds and settings |

---

## Data & Model Files

- `attention_monitor.db` — SQLite database with two tables: `sessions` (session start/end metadata) and `attention_logs` (periodic feature/prediction records per session). Auto-created on first run via `database.initialize_database()`.
- `attention_model.joblib` — Serialized bundle containing the trained model, feature scaler, label encoder, feature column order, and evaluation metrics. Auto-created on first run via `ml_engine.train_and_save_default_model()`.

Both files are safe to delete to reset the system to a clean state — they will be regenerated automatically the next time `main.py` or the dashboard runs.

---

## Notes & Known Limitations

- This is a **prototype/portfolio project**, not a validated psychological attention-measurement tool. The synthetic training labels are heuristic approximations, not ground truth.
- Attention classification is a **proxy signal** derived from observable facial/behavioural cues (eye state, gaze, head pose, blink/yawn frequency) — it does not measure cognitive attention directly.
- Camera lighting, glasses, and camera angle can all affect landmark accuracy and downstream feature quality.

---

## Tech Stack

**Computer Vision:** OpenCV, MediaPipe (FaceLandmarker Tasks API)
**ML:** scikit-learn (Random Forest, Decision Tree, Logistic Regression), joblib
**Data:** pandas, NumPy, SQLite
**Dashboard:** Streamlit, Plotly, Matplotlib
**Config:** python-dotenv
