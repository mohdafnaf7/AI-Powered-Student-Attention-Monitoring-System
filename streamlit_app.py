"""
streamlit_app.py
================
Streamlit-based dashboard frontend for the AI-Powered Real-Time Student
Attention and Engagement Monitoring System.

This app features:
    1. Live Attention Monitor: Start/stop webcam-based real-time analysis,
       render annotated feeds, view live attention gauge, and plot active
       time-series charts.
    2. Session History & Analytics: Browse previous monitoring sessions, view
       aggregate KPIs, and analyze details with Plotly graphs.
    3. Model Management: View performance metrics (accuracy, F1, etc.),
       feature importance bars, and trigger model retraining on logged data.
    4. System Configuration: Calibrate CV thresholds in real time and manage database resets.
"""

import os
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import mediapipe as mp

# Import project modules
import config
import database as db
import ml_engine
import utils
from main import (
    ensure_model_ready,
    ensure_face_landmarker_model,
    create_face_landmarker,
    extract_features_from_frame,
    build_absent_face_features,
)

# -------------------------------------------------------------------------
# Streamlit Page Settings & Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom css for modern dark theme and card style overlays
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Apply typography */
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main app background gradient */
    .stApp {
        background: linear-gradient(135deg, #090616 0%, #110c26 50%, #05030d 100%);
        color: #ffffff;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 15px;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(108, 92, 231, 0.3);
        box-shadow: 0 12px 40px 0 rgba(108, 92, 231, 0.15);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
    }
    
    .badge-attentive {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ECC71;
        border: 1px solid rgba(46, 204, 113, 0.30);
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.1);
    }
    
    .badge-passive {
        background-color: rgba(241, 196, 15, 0.15);
        color: #F1C40F;
        border: 1px solid rgba(241, 196, 15, 0.30);
        box-shadow: 0 0 15px rgba(241, 196, 15, 0.1);
    }
    
    .badge-distracted {
        background-color: rgba(231, 76, 60, 0.15);
        color: #E74C3C;
        border: 1px solid rgba(231, 76, 60, 0.30);
        box-shadow: 0 0 15px rgba(231, 76, 60, 0.1);
    }
    
    /* Glowing button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #6c5ce7 0%, #5b4bda 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.5) !important;
        background: linear-gradient(135deg, #7d6ef7 0%, #6858e7 100%) !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Red stop button styling */
    div.stButton > button.stop-button-style {
        background: linear-gradient(135deg, #d63031 0%, #b32d2e 100%) !important;
        box-shadow: 0 4px 15px rgba(214, 48, 49, 0.3) !important;
    }
    div.stButton > button.stop-button-style:hover {
        box-shadow: 0 6px 20px rgba(214, 48, 49, 0.5) !important;
        background: linear-gradient(135deg, #e15f60 0%, #cc3a3b 100%) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def get_attention_badge_html(prediction: str) -> str:
    """Return colored badge HTML corresponding to predicted class."""
    if prediction == "Highly Attentive":
        return f'<div class="metric-badge badge-attentive">{prediction}</div>'
    elif prediction == "Passive":
        return f'<div class="metric-badge badge-passive">{prediction}</div>'
    else:
        return f'<div class="metric-badge badge-distracted">{prediction}</div>'

def render_header():
    """Render beautiful gradient header."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #6C5CE7 0%, #4834DF 100%); 
                    padding: 22px; border-radius: 12px; margin-bottom: 25px; 
                    box-shadow: 0 10px 25px rgba(72, 52, 223, 0.25); text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: 0.5px;">
                🎓 Student Attention & Engagement Monitoring
            </h1>
            <p style="color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 1rem; font-weight: 300;">
                AI-Powered Computer Vision Pipeline & Behavioral Analytics Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Ensure database and model are ready. This is expensive (verifies/loads the
# joblib model bundle and opens a SQLite connection), so guard it behind a
# session-state flag rather than letting it re-run on every single script
# execution — which happens constantly once the live-monitoring fragment
# below is active.
if "app_initialized" not in st.session_state:
    db.initialize_database()
    ensure_model_ready()
    st.session_state.app_initialized = True

# -------------------------------------------------------------------------
# Navigation Sidebar
# -------------------------------------------------------------------------
st.sidebar.markdown(
    f"<h2 style='text-align: center; color: #6c5ce7;'>{config.APP_ICON} SMP System</h2>",
    unsafe_allow_html=True
)

# Quick status display
status_placeholder = st.sidebar.empty()
if "monitoring_active" in st.session_state and st.session_state.monitoring_active:
    status_placeholder.markdown("🟢 **Status:** Active Session running...")
else:
    status_placeholder.markdown("⚪ **Status:** System Idle")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🖥️ Live Attention Monitor",
        "📊 Session History & Analytics",
        "🧠 Model Performance",
        "⚙️ System Configuration",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Google DeepMind Advanced Agentic Coding • Antigravity AI")

# -------------------------------------------------------------------------
# Page 1: Live Attention Monitor
# -------------------------------------------------------------------------
if page == "🖥️ Live Attention Monitor":
    render_header()
    
    st.subheader("Real-Time Webcam Pipeline")
    
    # Check landmarker task is downloaded
    try:
        ensure_face_landmarker_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        
    # Session State Initialization for Video Capture Loop
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False

    col1, col2 = st.columns([7, 3])
    
    with col2:
        st.markdown("### Controls & KPIs")
        control_placeholder = st.empty()
        
        # State: Stopped, offer Start button
        if not st.session_state.monitoring_active:
            st.info("Ensure your webcam is not in use by other applications.")
            if control_placeholder.button("🟢 Start Monitoring", use_container_width=True):
                st.session_state.monitoring_active = True
                
                # Setup DB Session
                st.session_state.session_id = db.create_session()
                st.session_state.blink_detector = utils.BlinkDetector()
                st.session_state.yawn_detector = utils.YawnDetector()
                st.session_state.session_start_time = time.time()
                st.session_state.last_log_time = 0.0
                st.session_state.last_timestamp_ms = -1
                st.session_state.history_scores = []
                st.session_state.history_times = []
                
                # Open webcam
                st.session_state.cap = cv2.VideoCapture(config.CAMERA_INDEX)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                
                # Create MediaPipe Landmarker
                st.session_state.landmarker = create_face_landmarker()
                st.rerun()
                
        # State: Active, offer Stop button
        else:
            # HTML element injected with Javascript to apply custom class
            st.markdown(
                "<script>var btns = window.parent.document.getElementsByTagName('button');"
                "for (var i = 0; i < btns.length; i++) {"
                "  if (btns[i].textContent.includes('🔴 Stop Monitoring')) { btns[i].className += ' stop-button-style'; }"
                "}</script>",
                unsafe_allow_html=True
            )
            if control_placeholder.button("🔴 Stop Monitoring", use_container_width=True):
                st.session_state.monitoring_active = False
                
                # Clean up camera
                if "cap" in st.session_state and st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                if "landmarker" in st.session_state and st.session_state.landmarker is not None:
                    st.session_state.landmarker.close()
                    st.session_state.landmarker = None
                    
                # End Session in Database
                db.end_session(st.session_state.session_id)
                st.success(f"Monitoring Session #{st.session_state.session_id} finalized and saved to database!")
                st.rerun()
                
        # Metric display panel
        metrics_container = st.container()
        
    with col1:
        # Video feed view placeholder
        video_placeholder = st.empty()
        
        if not st.session_state.monitoring_active:
            # Show aesthetic landing content when idle
            st.markdown(
                """
                <div style="background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1); 
                            border-radius: 12px; height: 350px; display: flex; flex-direction: column; 
                            justify-content: center; align-items: center;">
                    <div style="font-size: 4rem; margin-bottom: 15px;">📹</div>
                    <h3 style="color: #a29bfe; margin: 0;">Camera Feed Off</h3>
                    <p style="color: rgba(255,255,255,0.5); margin: 5px 0 0 0; text-align: center; max-width: 320px;">
                        Click "🟢 Start Monitoring" to initialize the MediaPipe pipeline and open your webcam.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # -------------------------------------------------------------------
    # Live frame-processing loop, isolated in an @st.fragment.
    #
    # Why a fragment: previously this block called st.rerun() directly,
    # which re-executes the ENTIRE script every frame (CSS injection,
    # sidebar, ensure_model_ready() checks, full page layout, etc.) on
    # top of the actual per-frame CV/ML work. That full-page teardown
    # and rebuild ~once a second is what caused the visible flicker and
    # made it look like the webcam was blinking on and off.
    #
    # A fragment reruns ONLY this function's body on its own timer
    # (run_every), leaving the sidebar, CSS, and the rest of the page
    # untouched. The video placeholder just gets a new image each cycle
    # instead of the whole app reloading.
    # -------------------------------------------------------------------
    @st.fragment(run_every=0.08)
    def render_live_monitoring():
        cap = st.session_state.cap

        if cap is None or not cap.isOpened():
            st.error("Failed to connect to webcam device. Verify CAMERA_INDEX configuration.")
            st.session_state.monitoring_active = False
            return

        success, frame = cap.read()
        if not success:
            st.session_state.monitoring_active = False
            cap.release()
            if "landmarker" in st.session_state and st.session_state.landmarker is not None:
                st.session_state.landmarker.close()
            st.error("Error: Could not read frame from webcam. Ending session.")
            # This ends the session and needs to change the idle/active
            # layout for the whole page, so it's the one case in this
            # fragment that intentionally asks for a full-app rerun.
            st.rerun(scope="app")
            return

        # 1. Flip and process frame
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - st.session_state.session_start_time) * 1000)
        if timestamp_ms <= st.session_state.last_timestamp_ms:
            timestamp_ms = st.session_state.last_timestamp_ms + 1
        st.session_state.last_timestamp_ms = timestamp_ms

        # Run landmarker detection
        results = st.session_state.landmarker.detect_for_video(mp_image, timestamp_ms)

        # 2. Extract Features
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            feature_dict = extract_features_from_frame(
                landmarks, frame_width, frame_height,
                st.session_state.blink_detector, st.session_state.yawn_detector,
                st.session_state.session_start_time,
            )
            utils.draw_face_mesh_points(frame, landmarks, frame_width, frame_height)
        else:
            feature_dict = build_absent_face_features(
                st.session_state.blink_detector, st.session_state.yawn_detector,
                st.session_state.session_start_time
            )

        model_features = {k: v for k, v in feature_dict.items() if not k.startswith("_")}

        # 3. Model Inference & Scoring
        predicted_class, confidence, _ = ml_engine.predict(model_features)
        attention_score = utils.calculate_attention_score(
            ear=model_features["ear"],
            gaze_score=model_features["eye_direction_score"],
            pitch=model_features["pitch"],
            yaw=model_features["yaw"],
            blink_rate=model_features["blink_rate"],
            yawn_count=model_features["yawn_count"],
        )

        # Annotate opencv frame for the user
        utils.draw_annotations(
            frame,
            prediction=predicted_class,
            attention_score=attention_score,
            ear=model_features["ear"],
            mar=model_features["mar"],
            blink_count=feature_dict["_blink_count"],
            yawn_count=feature_dict["_yawn_count"],
        )

        # Render video frame in app
        rgb_annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_annotated, channels="RGB", use_container_width=True)

        # 4. Save to lists for live rolling chart
        elapsed_time = round(time.time() - st.session_state.session_start_time, 1)
        st.session_state.history_scores.append(attention_score)
        st.session_state.history_times.append(elapsed_time)

        # Limit history size to prevent slow reruns
        if len(st.session_state.history_scores) > 100:
            st.session_state.history_scores.pop(0)
            st.session_state.history_times.pop(0)

        # 5. Database Logging
        now = time.time()
        if now - st.session_state.last_log_time >= config.LOGGING_INTERVAL_SECONDS:
            db.insert_attention_log(
                session_id=st.session_state.session_id,
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
            st.session_state.last_log_time = now

        # 6. Update KPIs column
        with metrics_container:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Attention Classification</div>
                    <div class="metric-value" style="font-size:1.7rem; color:#a29bfe;">{predicted_class}</div>
                    {get_attention_badge_html(predicted_class)}
                </div>
                <div class="metric-card">
                    <div class="metric-title">Attention Score</div>
                    <div class="metric-value" style="color:#2ecc71;">{attention_score:.1f}%</div>
                    <div style="background-color:rgba(255,255,255,0.05); border-radius:5px; height:8px; margin-top:10px;">
                        <div style="background-color:#2ecc71; width:{attention_score}%; height:100%; border-radius:5px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="padding: 12px;">
                        <div class="metric-title" style="font-size:0.75rem;">Blinks</div>
                        <div class="metric-value" style="font-size:1.3rem;">{feature_dict["_blink_count"]}</div>
                    </div>
                    <div class="metric-card" style="padding: 12px;">
                        <div class="metric-title" style="font-size:0.75rem;">EAR</div>
                        <div class="metric-value" style="font-size:1.3rem;">{model_features["ear"]:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with sub_col2:
                st.markdown(
                    f"""
                    <div class="metric-card" style="padding: 12px;">
                        <div class="metric-title" style="font-size:0.75rem;">Yawns</div>
                        <div class="metric-value" style="font-size:1.3rem;">{feature_dict["_yawn_count"]}</div>
                    </div>
                    <div class="metric-card" style="padding: 12px;">
                        <div class="metric-title" style="font-size:0.75rem;">Gaze</div>
                        <div class="metric-value" style="font-size:1.3rem; text-transform:capitalize;">{feature_dict["_eye_direction_label"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Draw real-time trend chart
        if len(st.session_state.history_scores) > 1:
            st.markdown("### Real-Time Attention Trend")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.history_times,
                    y=st.session_state.history_scores,
                    mode="lines+markers",
                    name="Attention Score",
                    line=dict(color="#6c5ce7", width=3),
                    marker=dict(size=4, color="#a29bfe"),
                    fill="tozeroy",
                    fillcolor="rgba(108, 92, 231, 0.15)",
                )
            )
            # Class boundary overlays
            fig.add_hline(y=config.SCORE_HIGH_THRESHOLD, line_dash="dash", line_color="#2ecc71",
                         annotation_text="Highly Attentive Bound", annotation_position="top left")
            fig.add_hline(y=config.SCORE_LOW_THRESHOLD, line_dash="dash", line_color="#e74c3c",
                         annotation_text="Distracted Bound", annotation_position="bottom left")

            fig.update_layout(
                xaxis_title="Time (seconds elapsed)",
                yaxis_title="Attention Score (%)",
                yaxis_range=[0, 105],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(color="#ffffff"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                height=240,
            )
            st.plotly_chart(fig, use_container_width=True)
        # No manual time.sleep()/st.rerun() here — run_every=0.08 above
        # (~12.5 fps) re-triggers just this fragment on its own.

    # Active monitoring thread simulation, now scoped to a fragment
    if st.session_state.monitoring_active:
        render_live_monitoring()

# -------------------------------------------------------------------------
# Page 2: Session History & Analytics
# -------------------------------------------------------------------------
elif page == "📊 Session History & Analytics":
    render_header()
    
    st.subheader("Historical Analytics & Reports")
    
    sessions_df = db.get_all_sessions()
    
    if sessions_df.empty:
        st.info("No recorded sessions found in the database. Run a live monitoring session to generate data.")
    else:
        # 1. Main KPI metrics banner
        stats = db.get_summary_statistics()
        
        banner_col1, banner_col2, banner_col3, banner_col4 = st.columns(4)
        with banner_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Sessions</div>
                    <div class="metric-value">{stats["total_sessions"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with banner_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Monitored Time</div>
                    <div class="metric-value">{round(stats["total_monitoring_time_seconds"] / 60.0, 1)}m</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with banner_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Mean Attention</div>
                    <div class="metric-value" style="color:#2ecc71;">{stats["avg_attention_score"]:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with banner_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Blinks/Frame</div>
                    <div class="metric-value" style="color:#a29bfe;">{stats["avg_blink_count"]:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.divider()
        
        # 2. Dropdown Selector
        st.markdown("### Browse Sessions")
        session_list = []
        for idx, row in sessions_df.iterrows():
            lbl = (f"Session #{row['session_id']} | Start: {row['start_time']} "
                   f"({int(row['duration_seconds'] or 0)}s) | Avg Score: {row['avg_attention_score']:.1f}%")
            session_list.append((row['session_id'], lbl))
            
        selected_session_id = st.selectbox(
            "Select Session to Analyze",
            options=[item[0] for item in session_list],
            format_func=lambda sid: next(item[1] for item in session_list if item[0] == sid)
        )
        
        # 3. Retrieve Session Details
        sess_meta = db.get_session_by_id(selected_session_id)
        sess_logs = db.get_logs_by_session(selected_session_id)
        
        if sess_meta and not sess_logs.empty:
            # Single session info cards
            st.markdown("#### Session Meta Summary")
            subcol1, subcol2, subcol3, subcol4 = st.columns(4)
            with subcol1:
                st.write("**Start Time:**", sess_meta["start_time"])
                st.write("**End Time:**", sess_meta["end_time"] or "Active / Aborted")
            with subcol2:
                st.write("**Total Duration:**", f"{int(sess_meta['duration_seconds'] or 0)} seconds")
                st.write("**Avg Attention:**", f"{sess_meta['avg_attention_score']:.1f}%")
            with subcol3:
                st.write("**Total Blinks:**", sess_meta["total_blinks"])
                st.write("**Total Yawns:**", sess_meta["total_yawns"])
            with subcol4:
                st.write("**Dominant Class:**", sess_meta["dominant_prediction"] or "Unknown")
                
            st.divider()
            
            # Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("##### Attention Score Timeline")
                fig_timeline = px.line(
                    sess_logs,
                    x="session_duration",
                    y="attention_score",
                    title="Attention Score Trend (%)",
                    labels={"session_duration": "Duration (seconds)", "attention_score": "Score"},
                )
                fig_timeline.add_hline(y=config.SCORE_HIGH_THRESHOLD, line_dash="dash", line_color="green")
                fig_timeline.add_hline(y=config.SCORE_LOW_THRESHOLD, line_dash="dash", line_color="red")
                fig_timeline.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
            with chart_col2:
                st.markdown("##### State Distribution")
                dist_data = sess_logs["prediction"].value_counts().reset_index()
                dist_data.columns = ["prediction", "count"]
                
                # Map colors based on config
                colors = [config.CLASS_COLOR_MAP.get(cname, "#ffffff") for cname in dist_data["prediction"]]
                
                fig_pie = px.pie(
                    dist_data,
                    values="count",
                    names="prediction",
                    title="Proportion of Monitored Attention Classes",
                    color_discrete_sequence=colors,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            chart_col3, chart_col4 = st.columns(2)
            
            with chart_col3:
                st.markdown("##### Gaze Direction Count")
                gaze_data = sess_logs["eye_direction"].value_counts().reset_index()
                gaze_data.columns = ["eye_direction", "count"]
                fig_gaze = px.bar(
                    gaze_data,
                    x="eye_direction",
                    y="count",
                    title="Gaze Classification Frequencies",
                    color="eye_direction",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_gaze.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_gaze, use_container_width=True)
                
            with chart_col4:
                st.markdown("##### Eye & Mouth Metrics")
                fig_biometrics = go.Figure()
                fig_biometrics.add_trace(go.Scatter(x=sess_logs["session_duration"], y=sess_logs["ear"], 
                                                     name="EAR (Eyes Openness)", line=dict(color="#3498db")))
                fig_biometrics.add_trace(go.Scatter(x=sess_logs["session_duration"], y=sess_logs["mar"], 
                                                     name="MAR (Mouth Openness)", line=dict(color="#e67e22")))
                fig_biometrics.update_layout(
                    title="EAR vs MAR Bio-signals Progression",
                    xaxis_title="Duration (seconds)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_biometrics, use_container_width=True)
                
            # Export reports
            st.subheader("Export Center")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                logs_csv = sess_logs.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Session Logs as CSV",
                    data=logs_csv,
                    file_name=f"attention_session_{selected_session_id}_logs.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_col2:
                # Full sessions log
                all_sessions_csv = db.get_all_sessions().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Sessions List as CSV",
                    data=all_sessions_csv,
                    file_name="attention_sessions_all.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.warning("Could not load session data.")

# -------------------------------------------------------------------------
# Page 3: Model Performance
# -------------------------------------------------------------------------
elif page == "🧠 Model Performance":
    render_header()
    
    st.subheader("Machine Learning Engine Info")
    
    model_info = ml_engine.get_model_info()
    
    if model_info["model_name"] == "Not Trained":
        st.warning("Model bundle has not been trained yet.")
        if st.button("🚀 Bootstrap Default Model Bundle"):
            with st.spinner("Generating synthetic data and training classifier..."):
                ml_engine.train_and_save_default_model()
            st.success("Default model bootstrapped successfully! Refreshing...")
            st.rerun()
    else:
        # Display general info
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 5px solid #6c5ce7;">
                <div class="metric-title">Active Estimator</div>
                <div class="metric-value" style="font-size:1.6rem; color:#a29bfe;">{model_info["model_name"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display primary metrics
        metrics = model_info.get("metrics", {})
        if metrics:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.1f}%")
            with m_col2:
                st.metric("Precision", f"{metrics.get('precision', 0.0)*100:.1f}%")
            with m_col3:
                st.metric("Recall", f"{metrics.get('recall', 0.0)*100:.1f}%")
            with m_col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0.0)*100:.1f}%")
                
            st.divider()
            
            # Two column view
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("#### Feature Importances")
                importance_dict = model_info.get("feature_importance", {})
                if importance_dict:
                    importance_df = pd.DataFrame(
                        {"Feature": list(importance_dict.keys()), "Importance": list(importance_dict.values())}
                    ).sort_values("Importance", ascending=True)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Relative Feature weights in predictions",
                        color="Importance",
                        color_continuous_scale="Purples"
                    )
                    fig_importance.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ffffff"),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance data is unavailable for this model.")
                    
            with perf_col2:
                st.markdown("#### Detailed Classification Report")
                report_dict = metrics.get("classification_report", {})
                if report_dict:
                    # Render as formatted dataframe
                    report_rows = []
                    for k, v in report_dict.items():
                        if isinstance(v, dict):
                            report_rows.append({
                                "Class": k,
                                "Precision": f"{v.get('precision', 0.0)*100:.1f}%",
                                "Recall": f"{v.get('recall', 0.0)*100:.1f}%",
                                "F1-Score": f"{v.get('f1-score', 0.0)*100:.1f}%",
                                "Support": int(v.get('support', 0))
                            })
                    report_df = pd.DataFrame(report_rows)
                    st.dataframe(report_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Classification report details are unavailable.")
                    
        st.divider()
        
        # Model retraining
        st.subheader("Model Retraining Pipeline")
        st.markdown(
            "Improve classifier predictions by training it on real behavioral signals logged during student sessions. "
            "If database log volume is small, we recommend blending real records with domain-informed synthetic data to prevent overfitting."
        )
        
        # Check current data size
        all_logs = db.get_all_logs()
        st.write(f"**Current SQLite log records available for training:** {len(all_logs)} rows")
        
        blend_check = st.checkbox("Blend real records with synthetic domain data", value=True)
        
        if st.button("🚀 Retrain Model on Available Data"):
            if len(all_logs) < 10 and not blend_check:
                st.error("Too few logged entries in the database to train a standalone model. "
                         "Please enable 'Blend real records with synthetic domain data' or record more sessions first.")
            else:
                with st.spinner("Processing logs, generating training splits, and fitting classifiers..."):
                    # If we don't blend, we pass only real logs; otherwise retrain_from_logs handles both
                    try:
                        results = ml_engine.retrain_from_logs(all_logs, blend_with_synthetic=blend_check)
                        st.success(f"ML Classifier successfully retrained! Selected best model: **{results['best_model_name']}**")
                        time.sleep(1.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retraining failed: {str(e)}")

# -------------------------------------------------------------------------
# Page 4: System Configuration
# -------------------------------------------------------------------------
elif page == "⚙️ System Configuration":
    render_header()
    
    st.subheader("Settings & Calibrations")
    
    # 1. Behavior Threshold Sliders
    st.markdown("#### Behavioural Parameter Calibration")
    st.info("Tune these parameters to calibrate the system for different illumination or webcam angles. "
            "Changes apply immediately to live monitoring sessions.")
            
    # EAR Slider
    st.markdown("**Eye Aspect Ratio (EAR) Threshold**")
    config.EAR_BLINK_THRESHOLD = st.slider(
        "Value below which eyes are classified as closed / blink",
        min_value=0.10,
        max_value=0.35,
        value=config.EAR_BLINK_THRESHOLD,
        step=0.01,
        help="Typical values range from 0.18 to 0.24 depending on webcam distance."
    )
    
    # MAR Slider
    st.markdown("**Mouth Aspect Ratio (MAR) Threshold**")
    config.MAR_YAWN_THRESHOLD = st.slider(
        "Value above which mouth is classified as open / yawn",
        min_value=0.30,
        max_value=0.85,
        value=config.MAR_YAWN_THRESHOLD,
        step=0.02,
        help="Higher values avoid false positives from speaking or smiling."
    )
    
    # Head Pose Yaw Slider
    st.markdown("**Head Turn Thresholds (Degrees)**")
    config.HEAD_YAW_THRESHOLD = st.slider(
        "Head Turn angle (Left/Right) bounds",
        min_value=10.0,
        max_value=45.0,
        value=config.HEAD_YAW_THRESHOLD,
        step=1.0,
        help="Angle deviation where the student is flagged as looking away."
    )
    
    # Head Pose Pitch Slider
    config.HEAD_PITCH_THRESHOLD = st.slider(
        "Head Pitch angle (Up/Down) bounds",
        min_value=10.0,
        max_value=45.0,
        value=config.HEAD_PITCH_THRESHOLD,
        step=1.0,
        help="Pitch deviation indicating looking up/down."
    )
    
    st.divider()
    
    # 2. Hardware configuration
    st.markdown("#### Hardware Settings")
    config.CAMERA_INDEX = st.number_input(
        "Webcam Index (0 is standard integrated webcam)",
        min_value=0,
        max_value=10,
        value=config.CAMERA_INDEX,
        step=1,
    )
    
    st.divider()
    
    # 3. Database reset utility
    st.markdown("#### Database Maintenance")
    st.warning("Clearing all data deletes all historical sessions and attention logs. This action is irreversible.")
    
    confirm_text = st.text_input("Type **RESET** in all caps to unlock the database clear button:")
    
    if confirm_text == "RESET":
        if st.button("🗑️ Clear All Database Records"):
            db.clear_all_data()
            st.success("Database tables cleared successfully!")
            time.sleep(1.0)
            st.rerun()
    else:
        st.button("🗑️ Clear All Database Records", disabled=True)