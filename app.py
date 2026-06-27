import time
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
import math


# PAGE CONFIG
st.set_page_config(
    page_title="FormChecker · Exercise Analyser",
    layout="wide",
    # initial_sidebar_state="expanded", 
)


# CSS
st.markdown(
    """
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
          integrity="sha512-Avb2QiuTy/Sz92F5aAMBxQ5c9+P5bQPlbF0ixC5nFWM5KKx0c6Jj5zJqPJAlbVOT4E2kFGqVoBrGaJL0TkA=="
          crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
    /* ── PALETTE ── */
    :root {
        --bg: #090b0e;
        --surface: #0f1217;
        --panel: #13161c;
        --border: #1e2330;
        --border-hi: #2e3448;
        --accent: #e8f000;
        --accent-dim: rgba(232,240,0,0.08);
        --danger: #ff3b3b;
        --danger-dim: rgba(255,59,59,0.10);
        --ok: #00d97e;
        --ok-dim: rgba(0,217,126,0.10);
        --muted: #4a5168;
        --muted-hi: #6b7590;
        --text: #c8cdd8;
        --text-hi: #eef0f5;
        --mono: 'IBM Plex Mono', monospace;
        --display: 'Barlow Condensed', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--mono) !important;
    }
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        display: block !important;
        width: 21rem !important;
    }
    [data-testid="stSidebar"] * { font-family: var(--mono) !important; }
    h1, h2, h3, h4 {
        font-family: var(--display) !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    

    /* buttons */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border-hi) !important;
        color: var(--muted-hi) !important;
        font-family: var(--mono) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.12em;
        border-radius: 2px !important;
        padding: 8px 14px !important;
        transition: all 0.15s ease !important;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-dim) !important;
    }

    button[kind="primary"] {
        border-color: var(--accent) !important;
        color: var(--bg) !important;
        background: var(--accent) !important;
        font-weight: 500 !important;
    }

    button[kind="primary"]:hover {
        background: #fff176 !important;
        border-color: #fff176 !important;
        color: var(--bg) !important;
    }

    /* selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        color: var(--text) !important;
        font-family: var(--mono) !important;
        font-size: 0.8rem !important;
    }

    /* metrics */
    [data-testid="stMetric"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-top: 2px solid var(--border-hi) !important;
        border-radius: 2px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-family: var(--mono) !important;
        font-size: 0.62rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-hi) !important;
        font-family: var(--display) !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 2px !important;
        border: 1px solid var(--border) !important;
    }
    hr { border-color: var(--border) !important; margin: 12px 0 !important; }

    /* ── COMPONENTS ── */
    .wordmark {
        font-family: var(--display) !important;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        color: var(--text-hi);
        text-transform: uppercase;
    }
    .wordmark span { color: var(--accent); }

    .section-label {
        font-family: var(--mono) !important;
        font-size: 0.60rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    .page-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding: 18px 0 14px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
    }
    .page-title {
        font-family: var(--display) !important;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: var(--text-hi);
        line-height: 1;
        text-transform: uppercase;
    }
    .page-title span { color: var(--accent); }
    .page-subtitle {
        font-family: var(--mono) !important;
        font-size: 0.65rem;
        letter-spacing: 0.16em;
        color: var(--muted);
        text-transform: uppercase;
        margin-top: 6px;
    }
    .page-badge {
        font-family: var(--mono) !important;
        font-size: 0.60rem;
        letter-spacing: 0.15em;
        color: var(--accent);
        border: 1px solid var(--accent);
        padding: 4px 10px;
        border-radius: 2px;
        text-transform: uppercase;
    }

    .fb {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
        border-radius: 2px;
        font-family: var(--display) !important;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .fb i { font-size: 1rem; }
    .fb-warn { background: var(--danger-dim); border: 1px solid var(--danger); color: var(--danger); border-left: 3px solid var(--danger); }
    .fb-ok { background: var(--ok-dim); border: 1px solid var(--ok); color: var(--ok); border-left: 3px solid var(--ok); }
    .fb-neutral { background: var(--accent-dim); border: 1px solid var(--border-hi); color: var(--muted-hi); border-left: 3px solid var(--border-hi); }
    .fb-calib { background: rgba(0,180,255,0.07); border: 1px solid #0099cc; color: #00b8ff; border-left: 3px solid #00b8ff; }

    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: var(--mono) !important;
        font-size: 0.70rem;
        letter-spacing: 0.08em;
        color: var(--muted-hi);
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .dot-ok { background: var(--ok); box-shadow: 0 0 6px var(--ok); }
    .dot-warn { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
    .dot-neutral { background: var(--muted); }
    .dot-accent { background: var(--accent); box-shadow: 0 0 6px var(--accent); }

    .calib-bar-wrap { background: var(--border); border-radius: 2px; height: 4px; width: 100%; margin-top: 8px; }
    .calib-bar-fill { height: 4px; border-radius: 2px; background: var(--accent); transition: width 0.3s ease; }

    .roadmap-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 0;
        font-family: var(--mono) !important;
        font-size: 0.70rem;
        letter-spacing: 0.06em;
        border-bottom: 1px solid var(--border);
        color: var(--muted);
    }
    .roadmap-item:last-child { border-bottom: none; }
    .roadmap-item.active { color: var(--accent); }
    .roadmap-item i { width: 14px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


# MEDIAPIPE
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSE = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0, # Set as 0 for better performance
    smooth_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LM = mp_pose.PoseLandmark
LEFT_HIP = LM.LEFT_HIP.value
RIGHT_HIP = LM.RIGHT_HIP.value
LEFT_KNEE = LM.LEFT_KNEE.value
RIGHT_KNEE = LM.RIGHT_KNEE.value
LEFT_ANKLE = LM.LEFT_ANKLE.value
RIGHT_ANKLE = LM.RIGHT_ANKLE.value
LEFT_SHOULDER = LM.LEFT_SHOULDER.value
RIGHT_SHOULDER = LM.RIGHT_SHOULDER.value


# GEOMETRY
def _lm_xy(landmarks, idx): # Helper function to extract (x, y) coordinates from a landmark
    p = landmarks[idx] # Get the landmark point at the specified index
    return np.array([p.x, p.y]) # Return the (x, y) coordinates as a NumPy array for easier calculations

def angle_between(a, b, c): # Function to calculate the angle at point b formed by points a and c
    ba = a - b # Vector from b to a
    bc = c - b # Vector from b to c
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9) # Calculate the cosine of the angle using the dot product formula, adding a small epsilon to prevent division by zero
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))) 

def hip_y(landmarks): # Calculate the average y-coordinate of the left and right hip landmarks to determine vertical position of the hips
    return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0 # Add the y-coordinates of both hips then divide by 2 to get average height

def knee_angle(landmarks): # Calculate the average knee angle for both legs using the angle_between function
    l = angle_between(_lm_xy(landmarks, LEFT_HIP),  _lm_xy(landmarks, LEFT_KNEE),  _lm_xy(landmarks, LEFT_ANKLE)) # Calculate the angle at the left knee using the hip, knee, and ankle landmarks
    r = angle_between(_lm_xy(landmarks, RIGHT_HIP), _lm_xy(landmarks, RIGHT_KNEE), _lm_xy(landmarks, RIGHT_ANKLE)) # Calculate the angle at the right knee using the hip, knee, and ankle landmarks
    return (l + r) / 2.0


# SESSION STATE
def _init_state(): # Function to initialize session state variables with default values if they don't already exist
    defaults = dict(
        is_counting_down=False,
        countdown_start=0.0,
        phase="STANDING",
        rep_count=0,
        calib_mode=False,
        calib_reps_collected=0,
        calib_depths=[],
        calib_depth=None,
        calib_current_max_y=0.0,
        feedback="Position yourself laterally to the camera and begin",
        feedback_type="neutral",
        depth_ratio=0.90,
        exercise="Squat",
        fps=0.0,
        t_last=time.time(),
        knee_angle_disp=0.0,
        hip_y_disp=0.0,
        prev_hip_y=0.0,
        standing_hip_y=0.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ss = st.session_state


# FSM CONSTANTS
DESCENT_THRESHOLD = 0.03
ASCENT_THRESHOLD = 0.03
STANDING_KNEE_MIN = 150


# FSM
def run_fsm(hy: float, ka: float):
    prev = ss.prev_hip_y
    delta = hy - prev 

    if ss.phase == "STANDING":
        ss.feedback = "Begin your squat, descend when ready"
        ss.feedback_type = "neutral"
        if delta > DESCENT_THRESHOLD:
            ss.phase = "DESCENDING"
            ss.standing_hip_y = prev
            if ss.calib_mode:
                ss.calib_current_max_y = hy

    elif ss.phase == "DESCENDING":
        ss.feedback = "Keep descending, control the movement"
        ss.feedback_type = "neutral"
        if ss.calib_mode and hy > ss.calib_current_max_y:
            ss.calib_current_max_y = hy
        if delta < -ASCENT_THRESHOLD:
            if ss.calib_depth is not None:
                required = ss.standing_hip_y + (
                    (ss.calib_depth - ss.standing_hip_y) * ss.depth_ratio
                )
                if hy < required:
                    ss.feedback = "Insufficient depth, go lower before ascending"
                    ss.feedback_type = "warn"
                    ss.prev_hip_y = hy
                    return
            ss.phase = "ASCENDING"

    elif ss.phase == "ASCENDING":
        ss.feedback = "Drive through, extend fully at the top"
        ss.feedback_type = "neutral"
        if ka >= STANDING_KNEE_MIN:
            ss.rep_count += 1
            ss.feedback = f"Rep {ss.rep_count} complete, good work"
            ss.feedback_type = "ok"
            ss.phase = "STANDING"
            if ss.calib_mode:
                ss.calib_depths.append(ss.calib_current_max_y)
                ss.calib_reps_collected += 1
                ss.calib_current_max_y = 0.0
                if ss.calib_reps_collected >= 3:
                    ss.calib_depth = float(np.mean(ss.calib_depths))
                    ss.calib_mode = False
                    ss.feedback = "Calibration complete, system locked to your ROM"
                    ss.feedback_type = "ok"
                    st.rerun()

    ss.prev_hip_y = hy


# FRAME PROCESSOR
def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    now = time.time()
    ss.fps = 1.0 / max(now - ss.t_last, 1e-6)
    ss.t_last = now

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = POSE.process(rgb)

    # subtle vignette
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (9, 11, 14), -1)
    cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0, frame_bgr)

    if result.pose_landmarks:
        lms = result.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame_bgr,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 230, 232), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 210, 220), thickness=1),
        )

        hy = hip_y(lms)
        ka = knee_angle(lms)
        ss.hip_y_disp = round(hy, 4)
        ss.knee_angle_disp = round(ka, 1)

        if ss.is_counting_down:
            time_left = 5.0 - (now - ss.countdown_start)
            if time_left > 0:
                txt = f"{math.ceil(time_left)}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.5, 5)
                cv2.putText(frame_bgr, txt, ((w - tw) // 2, (h + th) // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 3.5, (232, 240, 0), 5, cv2.LINE_AA)
                sub = "GET INTO POSITION"
                (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.putText(frame_bgr, sub, ((w - sw) // 2, (h + th) // 2 + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 165, 180), 1, cv2.LINE_AA)
            else:
                ss.is_counting_down = False
                ss.feedback = "Calibration active — perform 3 full squats"
                ss.feedback_type = "calib"
                run_fsm(hy, ka)
        else:
            run_fsm(hy, ka)

        phase_col = {
            "STANDING": (232, 240, 0), # Yellow for standing phase
            "DESCENDING": (0, 165, 255), # Orange for descending phase
            "ASCENDING": (0, 217, 126), # Green for ascending phase
        }.get(ss.phase, (180, 180, 180))

        # top accent line
        cv2.line(frame_bgr, (0, 0), (w, 0), (232, 240, 0), 2)

        cv2.putText(frame_bgr, ss.phase, (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.70, phase_col, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"REP {ss.rep_count}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"KNEE {ka:.0f} deg", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 100, 120),  1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"{ss.fps:.0f} fps", (w - 68, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 62, 78), 1, cv2.LINE_AA)

        if ss.calib_mode:
            filled = int((ss.calib_reps_collected / 3) * w)
            cv2.rectangle(frame_bgr, (0, h - 4), (w, h), (22, 26, 36), -1)
            cv2.rectangle(frame_bgr, (0, h - 4), (filled, h), (232, 240, 0), -1)
            label = f"CALIBRATING {ss.calib_reps_collected}/3"
            cv2.putText(frame_bgr, label, (12, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (232, 240, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame_bgr, "NO POSE DETECTED", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 59, 59), 2, cv2.LINE_AA)

    return frame_bgr


# SIDEBAR
with st.sidebar:
    st.markdown('<div class="wordmark">FORM<span>CHECKER</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.60rem;letter-spacing:0.18em;color:var(--muted);margin-bottom:16px;font-family:var(--mono)">'
        'EXERCISE ANALYSIS SYSTEM</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<div class="section-label"><i class="fa-solid fa-dumbbell"></i> Exercise</div>', unsafe_allow_html=True)
    exercise = st.selectbox(
        "exercise",
        ["Squat", "Deadlift (Will Be Implemeted Later)", "Overhead Press (Will Be Implemeted Later)"],
        index=0,
        label_visibility="collapsed",
    )
    ss.exercise = exercise

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label"><i class="fa-solid fa-crosshairs"></i> Calibration</div>', unsafe_allow_html=True)

    if ss.calib_depth is not None:
        dot_cls, dot_txt = "dot-ok", f"Calibrated &mdash; depth {ss.calib_depth:.4f}"
    elif ss.calib_mode:
        dot_cls, dot_txt = "dot-accent", f"Collecting &mdash; rep {ss.calib_reps_collected}/3"
    else:
        dot_cls, dot_txt = "dot-neutral", "Not calibrated"

    st.markdown(
        f'<div class="status-row"><span class="dot {dot_cls}"></span>{dot_txt}</div>',
        unsafe_allow_html=True,
    )

    if ss.calib_mode:
        pct = int((ss.calib_reps_collected / 3) * 100)
        st.markdown(
            f'<div class="calib-bar-wrap"><div class="calib-bar-fill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_r = st.columns(2)
    with col_c:
        # st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("Calibrate", use_container_width=True):
            ss.calib_mode = True
            ss.calib_reps_collected = 0
            ss.calib_depths = []
            ss.calib_depth = None
            ss.calib_current_max_y = 0.0
            ss.rep_count = 0
            ss.phase = "STANDING"
            ss.is_counting_down = True
            ss.countdown_start = time.time()
            ss.feedback = "Get into position — calibration starts in 5s"
            ss.feedback_type = "calib"
        # st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        if st.button("Reset", use_container_width=True):
            for key in list(ss.keys()):
                del ss[key]
            _init_state()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label"><i class="fa-solid fa-sliders"></i> Thresholds</div>', unsafe_allow_html=True)
    depth_pct = st.slider(
        "Required Depth (%)",
        min_value=70, max_value=100, value=90, step=5,
        help="% of calibrated depth required before ascending is accepted",
    )
    ss.depth_ratio = depth_pct / 100.0



# MAIN
st.markdown(
    """
    <div class="page-header">
        <div>
            <div class="page-title">Exercise <span>Form</span> Analyser</div>
            <div class="page-subtitle">
                <i class="fa-solid fa-circle" style="color:#00d97e;font-size:0.45rem;vertical-align:middle"></i>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_reps, col_phase, col_knee, col_depth, col_fps = st.columns(5)
metric_reps = col_reps.empty()
metric_phase = col_phase.empty()
metric_knee = col_knee.empty()
metric_depth = col_depth.empty()
metric_fps = col_fps.empty()

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
feedback_ph = st.empty()
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
video_ph = st.empty()

stop_col, _ = st.columns([1, 5])
with stop_col:
    # stop = st.button("Stop Webcam", use_container_width=True) # Changed to toggle button
    webcam_active = st.toggle("Toggle To Turn Webcam On/Off", value=True)


# FEEDBACK RENDERER
FB_MAP = {
    "warn": ("fb-warn", "fa-triangle-exclamation"),
    "ok": ("fb-ok", "fa-circle-check"),
    "neutral": ("fb-neutral", "fa-arrow-right"),
    "calib": ("fb-calib", "fa-crosshairs"),
}

def render_feedback():
    cls, icon = FB_MAP.get(ss.feedback_type, ("fb-neutral", "fa-arrow-right"))
    if ss.feedback:
        feedback_ph.markdown(
            f'<div class="fb {cls}"><i class="fa-solid {icon}"></i>{ss.feedback}</div>',
            unsafe_allow_html=True,
        )


# WEBCAM LOOP
# if not stop:
if webcam_active:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("Webcam not detected. Ensure it is connected and not in use by another application.")
        st.stop()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            annotated = process_frame(frame)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            metric_reps.metric("REPS", ss.rep_count)
            metric_phase.metric("PHASE", ss.phase.capitalize())
            metric_knee.metric("KNEE ANGLE", f"{ss.knee_angle_disp:.1f}°")
            metric_depth.metric("CALIB DEPTH", f"{ss.calib_depth:.4f}" if ss.calib_depth else "—")
            metric_fps.metric("FPS", f"{ss.fps:.0f}")

            render_feedback()
            video_ph.image(rgb, channels="RGB", use_container_width=True)

            time.sleep(0.01)

    except Exception as exc:
        st.error(f"Runtime error: {exc}")
    finally:
        cap.release()
else:
    st.markdown(
        # '<div class="fb fb-neutral"><i class="fa-solid fa-pause"></i>Webcam stopped — click Reset to restart the session</div>',
        # unsafe_allow_html=True,
        '<div class="fb fb-neutral"><i class="fa-solid fa-pause"></i>Webcam Has Been Toggled Off. Please Toggle On To Resume</div>',
        unsafe_allow_html=True, # Changed to toggle instead of turn off button
    )