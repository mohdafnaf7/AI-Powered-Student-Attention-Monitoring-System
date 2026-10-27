"""
ml_engine.py
============
Machine learning core of the Attention Monitoring System.

Responsibilities:
    - Synthetic (domain-informed) training dataset generation, used to
      bootstrap the classifier before enough real session data exists.
    - Data cleaning (missing values, invalid records, normalization).
    - Training and comparing Random Forest, Decision Tree, and Logistic
      Regression classifiers.
    - Automatic best-model selection based on F1 score.
    - Full evaluation suite: accuracy, precision, recall, F1, confusion
      matrix, ROC curve/AUC, classification report.
    - Model persistence (joblib) and loading.
    - Real-time single-sample prediction for main.py.
    - Retraining from real accumulated database records.

NOTE ON TRAINING DATA:
A brand-new webcam session has no ground-truth "was this student actually
attentive?" label. This module bootstraps the model with a synthetic,
domain-informed dataset (feature distributions tuned per class based on
established attention/fatigue research heuristics: e.g. attentive students
have high EAR, centered gaze, stable head pose, low blink/yawn rates).
This is standard practice for prototyping this class of system and is
NOT presented as real collected data. `retrain_from_logs()` allows the
model to be retrained later on genuine accumulated session data once
enough has been logged.
"""

from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

import config
import utils


# =========================================================================
# Module-level cache so the model is loaded from disk only once per process.
# =========================================================================
_MODEL_CACHE: Optional[Dict[str, Any]] = None


# =========================================================================
# Synthetic training data generation
# =========================================================================
def _generate_class_samples(class_label: str, n_samples: int) -> pd.DataFrame:
    """
    Generate `n_samples` synthetic feature rows for a single attention class
    using hand-tuned, research-informed distributions with realistic noise
    and class overlap (so the resulting model is not trivially 100% accurate).

    Args:
        class_label: One of config.ATTENTION_CLASSES.
        n_samples: Number of synthetic rows to generate.

    Returns:
        DataFrame with columns config.FEATURE_COLUMNS + "label".
    """
    rng = np.random.default_rng(config.RANDOM_STATE)

    if class_label == "Highly Attentive":
        ear = rng.normal(0.32, 0.03, n_samples)
        mar = rng.normal(0.25, 0.05, n_samples)
        blink_rate = rng.normal(15, 4, n_samples)
        yawn_count = rng.poisson(0.2, n_samples)
        pitch = rng.normal(0, 5, n_samples)
        yaw = rng.normal(0, 5, n_samples)
        roll = rng.normal(0, 5, n_samples)
        eye_direction_score = rng.normal(0.5, 0.05, n_samples)
        face_presence = rng.uniform(0.95, 1.0, n_samples)

    elif class_label == "Passive":
        ear = rng.normal(0.26, 0.04, n_samples)
        mar = rng.normal(0.30, 0.07, n_samples)
        blink_rate = rng.normal(22, 5, n_samples)
        yawn_count = rng.poisson(0.8, n_samples)
        pitch = rng.normal(0, 10, n_samples)
        yaw = rng.normal(0, 12, n_samples)
        roll = rng.normal(0, 8, n_samples)
        eye_direction_score = rng.normal(0.5, 0.12, n_samples)
        face_presence = rng.uniform(0.80, 1.0, n_samples)

    elif class_label == "Distracted":
        ear = rng.normal(0.20, 0.05, n_samples)
        mar = rng.normal(0.40, 0.10, n_samples)
        blink_rate = rng.normal(32, 7, n_samples)
        yawn_count = rng.poisson(2.5, n_samples)
        pitch = rng.normal(15, 10, n_samples) * rng.choice([-1, 1], n_samples)
        yaw = rng.normal(20, 12, n_samples) * rng.choice([-1, 1], n_samples)
        roll = rng.normal(10, 8, n_samples) * rng.choice([-1, 1], n_samples)
        # Bimodal gaze: looking far left or far right of center.
        eye_direction_score = np.where(
            rng.random(n_samples) < 0.5,
            rng.normal(0.20, 0.10, n_samples),
            rng.normal(0.80, 0.10, n_samples),
        )
        face_presence = rng.uniform(0.5, 0.95, n_samples)

    else:
        raise ValueError(f"Unknown class label: {class_label}")

    session_duration = rng.uniform(0, 3600, n_samples)

    df = pd.DataFrame(
        {
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
    )
    df["label"] = class_label
    return df


def generate_training_dataset(samples_per_class: Optional[int] = None) -> pd.DataFrame:
    """
    Build the full synthetic, domain-informed training dataset spanning
    all three attention classes.

    Args:
        samples_per_class: Number of rows to generate per class. Defaults
            to config.SYNTHETIC_SAMPLES_PER_CLASS.

    Returns:
        A shuffled DataFrame with columns config.FEATURE_COLUMNS + "label".
    """
    n = samples_per_class or config.SYNTHETIC_SAMPLES_PER_CLASS
    frames = [_generate_class_samples(cls, n) for cls in config.ATTENTION_CLASSES]
    full_df = pd.concat(frames, ignore_index=True)

    # Clip features to physically plausible ranges.
    full_df["ear"] = full_df["ear"].clip(0.05, 0.50)
    full_df["mar"] = full_df["mar"].clip(0.05, 0.90)
    full_df["blink_rate"] = full_df["blink_rate"].clip(0, 60)
    full_df["yawn_count"] = full_df["yawn_count"].clip(0, 20)
    full_df["pitch"] = full_df["pitch"].clip(-60, 60)
    full_df["yaw"] = full_df["yaw"].clip(-60, 60)
    full_df["roll"] = full_df["roll"].clip(-45, 45)
    full_df["eye_direction_score"] = full_df["eye_direction_score"].clip(0, 1)
    full_df["face_presence"] = full_df["face_presence"].clip(0, 1)
    full_df["session_duration"] = full_df["session_duration"].clip(0, None)

    # Shuffle rows so classes are interleaved.
    full_df = full_df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return full_df


# =========================================================================
# Data Cleaning
# =========================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw feature DataFrame prior to training or analysis.

    Steps performed:
        1. Drop rows with missing values in any required feature column.
        2. Drop rows with a missing/invalid label (if a label column exists).
        3. Remove physically invalid records (e.g. negative session
           duration, EAR/MAR/face_presence outside plausible bounds).
        4. Drop exact duplicate rows.

    Args:
        df: Raw DataFrame containing at least config.FEATURE_COLUMNS.

    Returns:
        A cleaned copy of the DataFrame with invalid rows removed.
    """
    cleaned = df.copy()

    required_cols = list(config.FEATURE_COLUMNS)
    if "label" in cleaned.columns:
        required_cols = required_cols + ["label"]

    # 1. Drop missing values in required columns.
    cleaned = cleaned.dropna(subset=[c for c in required_cols if c in cleaned.columns])

    # 2. Remove physically invalid records.
    if "ear" in cleaned.columns:
        cleaned = cleaned[(cleaned["ear"] >= 0.0) & (cleaned["ear"] <= 0.6)]
    if "mar" in cleaned.columns:
        cleaned = cleaned[(cleaned["mar"] >= 0.0) & (cleaned["mar"] <= 1.0)]
    if "face_presence" in cleaned.columns:
        cleaned = cleaned[(cleaned["face_presence"] >= 0.0) & (cleaned["face_presence"] <= 1.0)]
    if "session_duration" in cleaned.columns:
        cleaned = cleaned[cleaned["session_duration"] >= 0.0]
    if "blink_rate" in cleaned.columns:
        cleaned = cleaned[cleaned["blink_rate"] >= 0.0]
    if "yawn_count" in cleaned.columns:
        cleaned = cleaned[cleaned["yawn_count"] >= 0]

    # 3. Drop exact duplicates.
    cleaned = cleaned.drop_duplicates()

    return cleaned.reset_index(drop=True)


# =========================================================================
# Model Training & Comparison
# =========================================================================
def _get_candidate_models() -> Dict[str, Any]:
    """
    Instantiate the three candidate classifiers with fixed random states
    for reproducibility.

    Returns:
        Dictionary mapping model name to an unfitted sklearn estimator.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=config.RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
    }


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute a full evaluation report for a fitted classifier.

    Args:
        model: A fitted sklearn classifier exposing predict() and
            predict_proba().
        X_test: Scaled test feature matrix.
        y_test: Encoded true labels for the test set.
        class_names: Class names in label-encoder order, used for
            human-readable reports and ROC curves.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, confusion
        matrix, per-class ROC curve data (fpr/tpr/auc), and the sklearn
        classification report (as a dict).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Multiclass ROC curve via one-vs-rest binarization.
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    roc_data: Dict[str, Dict[str, Any]] = {}
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_data[cname] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc(fpr, tpr)),
        }

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_curve": roc_data,
    }


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and compare Random Forest, Decision Tree, and Logistic Regression
    classifiers on the given labeled dataset, then select the best model
    by weighted F1 score.

    Args:
        df: Cleaned DataFrame containing config.FEATURE_COLUMNS + "label".

    Returns:
        Dictionary with keys:
            "best_model_name": str
            "bundle": dict ready for saving (model, scaler, label_encoder,
                feature_columns, model_name)
            "all_results": dict mapping each model name to its
                evaluate_model() output
    """
    df = clean_data(df)

    X = df[config.FEATURE_COLUMNS].values
    y_raw = df["label"].values

    label_encoder = LabelEncoder()
    label_encoder.fit(config.ATTENTION_CLASSES)  # Fixed, known class order.
    y = label_encoder.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_names = list(label_encoder.classes_)
    candidates = _get_candidate_models()

    all_results: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model
        all_results[name] = evaluate_model(model, X_test_scaled, y_test, class_names)

    # Select the best model automatically by weighted F1 score.
    best_model_name = max(all_results, key=lambda n: all_results[n]["f1_score"])
    best_model = fitted_models[best_model_name]

    # Feature importance (Random Forest / Decision Tree expose it natively;
    # Logistic Regression uses absolute mean coefficient magnitude instead).
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(config.FEATURE_COLUMNS, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        mean_abs_coef = np.mean(np.abs(best_model.coef_), axis=0)
        importances = dict(zip(config.FEATURE_COLUMNS, mean_abs_coef.tolist()))
    else:
        importances = {}

    bundle = {
        "model": best_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": config.FEATURE_COLUMNS,
        "model_name": best_model_name,
        "metrics": all_results[best_model_name],
        "feature_importance": importances,
    }

    return {
        "best_model_name": best_model_name,
        "bundle": bundle,
        "all_results": all_results,
    }


# =========================================================================
# Model Persistence
# =========================================================================
def save_model(bundle: Dict[str, Any]) -> None:
    """
    Persist a trained model bundle (model + scaler + label encoder +
    metadata) to disk using joblib.

    Args:
        bundle: Dictionary as produced by train_models()["bundle"].
    """
    joblib.dump(bundle, config.MODEL_PATH)


def load_model(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load the trained model bundle from disk, caching it in memory so
    repeated calls (e.g. once per video frame) don't hit the filesystem.

    Args:
        force_reload: If True, bypass the in-memory cache and reload from
            disk (used after retraining).

    Returns:
        The model bundle dictionary (model, scaler, label_encoder,
        feature_columns, model_name, metrics, feature_importance).

    Raises:
        FileNotFoundError: If no trained model exists at config.MODEL_PATH.
    """
    global _MODEL_CACHE

    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    try:
        bundle = joblib.load(config.MODEL_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No trained model found at '{config.MODEL_PATH}'. "
            "Run train_and_save_default_model() first."
        ) from exc

    _MODEL_CACHE = bundle
    return bundle


def train_and_save_default_model() -> Dict[str, Any]:
    """
    Convenience function: generate the synthetic bootstrap dataset, train
    and compare all three models, save the best one to disk, and return
    the full training summary. Intended to be called once on first run.

    Returns:
        The dictionary returned by train_models() (best_model_name,
        bundle, all_results).
    """
    dataset = generate_training_dataset()
    results = train_models(dataset)
    save_model(results["bundle"])
    return results


# =========================================================================
# Real-Time Prediction
# =========================================================================
def predict(features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict the attention class for a single real-time feature sample.

    Args:
        features: Dictionary mapping each name in config.FEATURE_COLUMNS to
            its current value (e.g. as produced by utils.build_feature_dict).

    Returns:
        A tuple (predicted_class, confidence, class_probabilities) where:
            predicted_class: One of config.ATTENTION_CLASSES.
            confidence: Probability assigned to the predicted class [0, 1].
            class_probabilities: Dict mapping every class name to its
                predicted probability.
    """
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]

    # Enforce exact training-time feature order.
    row = np.array([[features[col] for col in feature_columns]], dtype=np.float64)
    row_scaled = scaler.transform(row)

    proba = model.predict_proba(row_scaled)[0]
    pred_idx = int(np.argmax(proba))
    predicted_class = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    class_probabilities = {
        str(cls): float(p) for cls, p in zip(label_encoder.classes_, proba)
    }

    return predicted_class, confidence, class_probabilities


# =========================================================================
# Retraining from real accumulated data
# =========================================================================
def retrain_from_logs(logs_df: pd.DataFrame, blend_with_synthetic: bool = True) -> Dict[str, Any]:
    """
    Retrain the model using real session data accumulated in the database,
    optionally blended with the synthetic bootstrap dataset for stability
    when real data volume is still small.

    Real logs read from the `attention_logs` table do not contain a
    ground-truth label column; this function derives a label for each row
    from its stored attention_score using utils.score_to_class(), which
    keeps the label definition consistent across the whole system.

    Args:
        logs_df: DataFrame of real records (must contain config.FEATURE_COLUMNS
            and an "attention_score" column).
        blend_with_synthetic: If True, combine real logs with a freshly
            generated synthetic dataset so the model doesn't overfit to a
            small number of real samples.

    Returns:
        The dictionary returned by train_models() (best_model_name,
        bundle, all_results), after saving the retrained model to disk.
    """
    logs_df = logs_df.copy()
    if "attention_score" in logs_df.columns:
        logs_df["label"] = logs_df["attention_score"].apply(utils.score_to_class)

    logs_df = clean_data(logs_df)

    if blend_with_synthetic:
        synthetic_df = generate_training_dataset(samples_per_class=config.SYNTHETIC_SAMPLES_PER_CLASS)
        combined = pd.concat(
            [synthetic_df, logs_df[config.FEATURE_COLUMNS + ["label"]]], ignore_index=True
        )
    else:
        combined = logs_df[config.FEATURE_COLUMNS + ["label"]]

    results = train_models(combined)
    save_model(results["bundle"])
    load_model(force_reload=True)
    return results


def get_model_info() -> Dict[str, Any]:
    """
    Retrieve metadata about the currently loaded model for display in the
    Streamlit dashboard (model name, metrics, feature importance).

    Returns:
        Dictionary with "model_name", "metrics", and "feature_importance".
        Returns empty defaults if no model has been trained yet.
    """
    try:
        bundle = load_model()
    except FileNotFoundError:
        return {"model_name": "Not Trained", "metrics": {}, "feature_importance": {}}

    return {
        "model_name": bundle.get("model_name", "Unknown"),
        "metrics": bundle.get("metrics", {}),
        "feature_importance": bundle.get("feature_importance", {}),
    }