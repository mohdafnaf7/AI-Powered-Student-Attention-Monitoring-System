"""
database.py
===========
SQLite persistence layer for the Attention Monitoring System.

Responsibilities:
    - Database connection management.
    - Schema creation for the `sessions` and `attention_logs` tables.
    - Inserting session and per-frame attention records.
    - Reading records back as pandas DataFrames for analytics.
    - Computing summary statistics for the dashboard.
    - Exporting data to CSV.

Design notes:
    - Only derived numeric behavioural features are stored. No images,
      video frames, or personally identifiable information are ever
      written to the database, in line with the project's privacy
      requirements.
    - `sessions` holds one row per monitoring session (start/end time,
      aggregate stats). `attention_logs` holds one row per logged
      measurement (every LOGGING_INTERVAL_SECONDS), linked to its parent
      session via a foreign key.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, Any, Generator

import pandas as pd

import config


# =========================================================================
# Connection management
# =========================================================================
@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding a SQLite connection with foreign keys enabled.

    The connection is committed and closed automatically on exit; on
    exception, the transaction is rolled back before the error propagates.

    Yields:
        An open sqlite3.Connection to config.DATABASE_PATH.
    """
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# =========================================================================
# Schema creation
# =========================================================================
def initialize_database() -> None:
    """
    Create the `sessions` and `attention_logs` tables if they do not
    already exist. Safe to call on every application startup.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds REAL DEFAULT 0.0,
                avg_attention_score REAL,
                total_blinks INTEGER DEFAULT 0,
                total_yawns INTEGER DEFAULT 0,
                dominant_prediction TEXT
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attention_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                attention_score REAL NOT NULL,
                ear REAL,
                mar REAL,
                blink_count INTEGER,
                yawn_count INTEGER,
                pitch REAL,
                yaw REAL,
                roll REAL,
                eye_direction TEXT,
                session_duration REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    ON DELETE CASCADE
            );
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_session_id ON attention_logs (session_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON attention_logs (timestamp);"
        )


# =========================================================================
# Session management
# =========================================================================
def create_session() -> int:
    """
    Insert a new row into `sessions` marking the start of a monitoring
    session.

    Returns:
        The newly created session_id.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (start_time) VALUES (?);",
            (datetime.now().isoformat(timespec="seconds"),),
        )
        return int(cursor.lastrowid)


def end_session(session_id: int) -> None:
    """
    Finalize a session: set its end_time and compute aggregate statistics
    (duration, average attention score, total blinks/yawns, dominant
    prediction) from its associated attention_logs rows.

    Args:
        session_id: The session to finalize.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Pull start time to compute duration.
        cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?;", (session_id,))
        row = cursor.fetchone()
        if row is None:
            return
        start_time = datetime.fromisoformat(row[0])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Aggregate stats from logs belonging to this session.
        cursor.execute(
            """
            SELECT AVG(attention_score), MAX(blink_count), MAX(yawn_count)
            FROM attention_logs WHERE session_id = ?;
            """,
            (session_id,),
        )
        avg_score, total_blinks, total_yawns = cursor.fetchone()

        cursor.execute(
            """
            SELECT prediction, COUNT(*) as cnt
            FROM attention_logs WHERE session_id = ?
            GROUP BY prediction ORDER BY cnt DESC LIMIT 1;
            """,
            (session_id,),
        )
        dominant_row = cursor.fetchone()
        dominant_prediction = dominant_row[0] if dominant_row else None

        cursor.execute(
            """
            UPDATE sessions
            SET end_time = ?, duration_seconds = ?, avg_attention_score = ?,
                total_blinks = ?, total_yawns = ?, dominant_prediction = ?
            WHERE session_id = ?;
            """,
            (
                end_time.isoformat(timespec="seconds"),
                duration,
                avg_score,
                total_blinks or 0,
                total_yawns or 0,
                dominant_prediction,
                session_id,
            ),
        )


# =========================================================================
# Log insertion
# =========================================================================
def insert_attention_log(
    session_id: int,
    prediction: str,
    attention_score: float,
    ear: float,
    mar: float,
    blink_count: int,
    yawn_count: int,
    pitch: float,
    yaw: float,
    roll: float,
    eye_direction: str,
    session_duration: float,
) -> int:
    """
    Insert a single attention measurement row into `attention_logs`.

    Args:
        session_id: Parent session ID (must already exist in `sessions`).
        prediction: Predicted attention class label.
        attention_score: Attention score in [0, 100].
        ear: Eye Aspect Ratio at time of measurement.
        mar: Mouth Aspect Ratio at time of measurement.
        blink_count: Cumulative blink count for the session so far.
        yawn_count: Cumulative yawn count for the session so far.
        pitch: Head pitch angle in degrees.
        yaw: Head yaw angle in degrees.
        roll: Head roll angle in degrees.
        eye_direction: Gaze direction label ("left"/"center"/"right").
        session_duration: Elapsed session time in seconds.

    Returns:
        The newly created log_id.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attention_logs (
                session_id, timestamp, prediction, attention_score, ear, mar,
                blink_count, yawn_count, pitch, yaw, roll, eye_direction, session_duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                session_id,
                datetime.now().isoformat(timespec="seconds"),
                prediction,
                attention_score,
                ear,
                mar,
                blink_count,
                yawn_count,
                pitch,
                yaw,
                roll,
                eye_direction,
                session_duration,
            ),
        )
        return int(cursor.lastrowid)


# =========================================================================
# Reading records
# =========================================================================
def get_all_logs() -> pd.DataFrame:
    """
    Fetch every row from `attention_logs`, ordered chronologically.

    Returns:
        DataFrame of all attention log records (empty DataFrame with the
        correct columns if no records exist yet).
    """
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM attention_logs ORDER BY timestamp ASC;", conn)
    return df


def get_logs_by_session(session_id: int) -> pd.DataFrame:
    """
    Fetch all attention_logs rows for a specific session.

    Args:
        session_id: The session to filter by.

    Returns:
        DataFrame of that session's log records, ordered chronologically.
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM attention_logs WHERE session_id = ? ORDER BY timestamp ASC;",
            conn,
            params=(session_id,),
        )
    return df


def get_recent_logs(limit: int = 50) -> pd.DataFrame:
    """
    Fetch the most recent N attention_logs rows (across all sessions),
    used to drive live-updating dashboard charts.

    Args:
        limit: Maximum number of rows to return.

    Returns:
        DataFrame of the most recent records, ordered chronologically
        (oldest first) so it can be plotted directly as a time series.
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM attention_logs ORDER BY log_id DESC LIMIT ?;",
            conn,
            params=(limit,),
        )
    return df.iloc[::-1].reset_index(drop=True)


def get_all_sessions() -> pd.DataFrame:
    """
    Fetch every row from `sessions`, most recent first.

    Returns:
        DataFrame of all session records.
    """
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY session_id DESC;", conn)
    return df


def get_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a single session's metadata as a dictionary.

    Args:
        session_id: The session to look up.

    Returns:
        Dictionary of the session's columns, or None if not found.
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM sessions WHERE session_id = ?;", conn, params=(session_id,)
        )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# =========================================================================
# Summary statistics
# =========================================================================
def get_summary_statistics() -> Dict[str, Any]:
    """
    Compute overall database summary statistics for the dashboard's
    "Database Summary" panel.

    Returns:
        Dictionary containing:
            total_sessions, total_logs, avg_attention_score,
            class_distribution (dict), avg_blink_count, avg_yawn_count,
            total_monitoring_time_seconds
    """
    logs_df = get_all_logs()
    sessions_df = get_all_sessions()

    if logs_df.empty:
        return {
            "total_sessions": int(len(sessions_df)),
            "total_logs": 0,
            "avg_attention_score": 0.0,
            "class_distribution": {},
            "avg_blink_count": 0.0,
            "avg_yawn_count": 0.0,
            "total_monitoring_time_seconds": 0.0,
        }

    class_distribution = logs_df["prediction"].value_counts().to_dict()
    total_monitoring_time = sessions_df["duration_seconds"].fillna(0).sum()

    return {
        "total_sessions": int(len(sessions_df)),
        "total_logs": int(len(logs_df)),
        "avg_attention_score": float(logs_df["attention_score"].mean()),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "avg_blink_count": float(logs_df["blink_count"].mean()),
        "avg_yawn_count": float(logs_df["yawn_count"].mean()),
        "total_monitoring_time_seconds": float(total_monitoring_time),
    }


def get_trend_data(limit: int = 200) -> pd.DataFrame:
    """
    Fetch attention score trend data (timestamp + attention_score) for
    line-chart visualization, most recent `limit` points.

    Args:
        limit: Maximum number of points to return.

    Returns:
        DataFrame with columns ["timestamp", "attention_score", "prediction"].
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT timestamp, attention_score, prediction
            FROM attention_logs ORDER BY log_id DESC LIMIT ?;
            """,
            conn,
            params=(limit,),
        )
    df = df.iloc[::-1].reset_index(drop=True)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# =========================================================================
# CSV Export
# =========================================================================
def export_logs_to_csv(file_path: Optional[str] = None) -> str:
    """
    Export the entire `attention_logs` table to a CSV file.

    Args:
        file_path: Destination path. Defaults to a timestamped file inside
            config.EXPORT_DIR.

    Returns:
        The absolute path of the written CSV file.
    """
    os.makedirs(config.EXPORT_DIR, exist_ok=True)
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(config.EXPORT_DIR, f"attention_logs_{timestamp}.csv")

    df = get_all_logs()
    df.to_csv(file_path, index=False)
    return file_path


def export_sessions_to_csv(file_path: Optional[str] = None) -> str:
    """
    Export the entire `sessions` table to a CSV file.

    Args:
        file_path: Destination path. Defaults to a timestamped file inside
            config.EXPORT_DIR.

    Returns:
        The absolute path of the written CSV file.
    """
    os.makedirs(config.EXPORT_DIR, exist_ok=True)
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(config.EXPORT_DIR, f"sessions_{timestamp}.csv")

    df = get_all_sessions()
    df.to_csv(file_path, index=False)
    return file_path


def clear_all_data() -> None:
    """
    Delete all rows from both tables (used for demo/reset purposes only).
    Table structure is preserved; auto-increment counters restart.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attention_logs;")
        cursor.execute("DELETE FROM sessions;")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('attention_logs', 'sessions');")