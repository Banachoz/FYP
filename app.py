import time
import cv2
import mediapipe as mp
import streamlit as st

from core.angle_calculator import avg_knee_angle, tracked_y_for_exercise, torso_angle
from core.phase_detector import run_fsm
import exercises.squat as _squat
from ui.overlay import (
    apply_vignette, draw_pose, draw_hud,
    draw_countdown, draw_calibration_bar, draw_no_pose,
)


# PAGE CONFIG
st.set_page_config(
    page_title="FormChecker · Exercise Analyser",
    layout="wide",
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
    .fb-warn    { background: var(--danger-dim); border: 1px solid var(--danger); color: var(--danger); border-left: 3px solid var(--danger); }
    .fb-ok      { background: var(--ok-dim);     border: 1px solid var(--ok);     color: var(--ok);     border-left: 3px solid var(--ok);     }
    .fb-neutral { background: var(--accent-dim); border: 1px solid var(--border-hi); color: var(--muted-hi); border-left: 3px solid var(--border-hi); }
    .fb-calib   { background: rgba(0,180,255,0.07); border: 1px solid #0099cc; color: #00b8ff; border-left: 3px solid #00b8ff; }

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
    .dot-ok      { background: var(--ok);     box-shadow: 0 0 6px var(--ok);     }
    .dot-warn    { background: var(--danger);  box-shadow: 0 0 6px var(--danger);  }
    .dot-neutral { background: var(--muted);   }
    .dot-accent  { background: var(--accent);  box-shadow: 0 0 6px var(--accent);  }

    .calib-bar-wrap { background: var(--border); border-radius: 2px; height: 4px; width: 100%; margin-top: 8px; }
    .calib-bar-fill { height: 4px; border-radius: 2px; background: var(--accent); transition: width 0.3s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)


# MEDIAPIPE
_mp_pose    = mp.solutions.pose
POSE = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# SESSION STATE
def _init_state():
    defaults = dict(
        is_counting_down=False,
        countdown_start=0.0,
        phase="STANDING",
        rep_count=0,
        calib_mode=False,
        calib_reps_collected=0,
        calib_depths=[],
        calib_depth=None,
        calib_peak=0.0,
        calib_peak_torso=0.0,
        calib_peak_knee=180.0,
        calib_torso_angles=[],
        calib_knee_angles=[],
        calib_torso_angle=None,
        calib_knee_angle=None,
        calib_just_completed=False,
        feedback="Position yourself laterally to the camera and begin",
        feedback_type="neutral",
        depth_ratio=0.90,
        exercise="Squat",
        fps=0.0,
        t_last=time.time(),
        knee_angle_disp=0.0,
        prev_tracked_y=0.0,
        standing_tracked_y=0.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ss = st.session_state


# FSM HELPERS — pack/unpack session state to/from the pure FSM dict
def _pack_state():
    return {
        "phase":               ss.phase,
        "rep_count":           ss.rep_count,
        "calib_mode":          ss.calib_mode,
        "calib_reps_collected":ss.calib_reps_collected,
        "calib_depths":        ss.calib_depths,
        "calib_depth":         ss.calib_depth,
        "calib_peak":          ss.calib_peak,
        "calib_peak_torso":    ss.calib_peak_torso,
        "calib_peak_knee":     ss.calib_peak_knee,
        "calib_torso_angles":  ss.calib_torso_angles,
        "calib_knee_angles":   ss.calib_knee_angles,
        "calib_torso_angle":   ss.calib_torso_angle,
        "calib_knee_angle":    ss.calib_knee_angle,
        "feedback":            ss.feedback,
        "feedback_type":       ss.feedback_type,
        "depth_ratio":         ss.depth_ratio,
        "prev_tracked_y":      ss.prev_tracked_y,
        "standing_tracked_y":  ss.standing_tracked_y,
    }

def _unpack_state(result):
    ss.phase                = result["phase"]
    ss.rep_count            = result["rep_count"]
    ss.calib_mode           = result["calib_mode"]
    ss.calib_reps_collected = result["calib_reps_collected"]
    ss.calib_depths         = result["calib_depths"]
    ss.calib_depth          = result["calib_depth"]
    ss.calib_peak           = result["calib_peak"]
    ss.calib_peak_torso     = result["calib_peak_torso"]
    ss.calib_peak_knee      = result["calib_peak_knee"]
    ss.calib_torso_angles   = result["calib_torso_angles"]
    ss.calib_knee_angles    = result["calib_knee_angles"]
    ss.calib_torso_angle    = result["calib_torso_angle"]
    ss.calib_knee_angle     = result["calib_knee_angle"]
    ss.feedback             = result["feedback"]
    ss.feedback_type        = result["feedback_type"]
    ss.prev_tracked_y       = result["prev_tracked_y"]
    ss.standing_tracked_y   = result["standing_tracked_y"]
    if result["needs_rerun"]:
        ss.calib_just_completed = True
        st.rerun()


# CALIBRATION COMPLETE DIALOG
@st.dialog("Calibration Complete")
def _calib_done_dialog():
    st.markdown(f"**{ss.exercise}** baseline locked in — depth recorded at `{ss.calib_depth:.4f}`.")
    st.markdown("Live form analysis will begin once you continue.")
    if st.button("Begin Analysis", type="primary", use_container_width=True):
        ss.calib_just_completed = False
        st.rerun()


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
        ["Squat", "Deadlift", "Overhead Press"],
        index=["Squat", "Deadlift", "Overhead Press"].index(ss.exercise),
        label_visibility="collapsed",
    )
    if exercise != ss.exercise:
        # Reset when exercise changes
        for key in list(ss.keys()):
            del ss[key]
        st.session_state["exercise"] = exercise
        _init_state()
        st.rerun()

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
        if st.button("Calibrate", use_container_width=True):
            ss.calib_mode           = True
            ss.calib_reps_collected = 0
            ss.calib_depths         = []
            ss.calib_depth          = None
            ss.calib_peak           = 0.0
            ss.rep_count            = 0
            ss.phase                = "STANDING"
            ss.is_counting_down     = True
            ss.countdown_start      = time.time()
            ss.feedback             = "Get into position — calibration starts in 5s"
            ss.feedback_type        = "calib"
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


# MAIN LAYOUT
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
metric_reps  = col_reps.empty()
metric_phase = col_phase.empty()
metric_knee  = col_knee.empty()
metric_depth = col_depth.empty()
metric_fps   = col_fps.empty()

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
feedback_ph = st.empty()
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
video_ph = st.empty()

if ss.calib_just_completed:
    _calib_done_dialog()
    st.stop()

stop_col, _ = st.columns([1, 5])
with stop_col:
    webcam_active = st.toggle("Toggle To Turn Webcam On/Off", value=True)


# FEEDBACK RENDERER
_FB_MAP = {
    "warn":    ("fb-warn",    "fa-triangle-exclamation"),
    "ok":      ("fb-ok",      "fa-circle-check"),
    "neutral": ("fb-neutral", "fa-arrow-right"),
    "calib":   ("fb-calib",   "fa-crosshairs"),
}

def render_feedback():
    cls, icon = _FB_MAP.get(ss.feedback_type, ("fb-neutral", "fa-arrow-right"))
    if ss.feedback:
        feedback_ph.markdown(
            f'<div class="fb {cls}"><i class="fa-solid {icon}"></i>{ss.feedback}</div>',
            unsafe_allow_html=True,
        )


# WEBCAM LOOP
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

            now = time.time()
            ss.fps  = 1.0 / max(now - ss.t_last, 1e-6)
            ss.t_last = now

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = POSE.process(rgb)

            frame = apply_vignette(frame)

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                ty = tracked_y_for_exercise(lms, ss.exercise)
                ka = avg_knee_angle(lms)
                ta = torso_angle(lms)
                ss.knee_angle_disp = round(ka, 1)

                if ss.is_counting_down:
                    time_left = 5.0 - (now - ss.countdown_start)
                    if time_left > 0:
                        frame = draw_countdown(frame, time_left, w, h)
                    else:
                        ss.is_counting_down = False
                        ss.feedback      = "Calibration active — perform 3 full reps"
                        ss.feedback_type = "calib"
                        _unpack_state(run_fsm(_pack_state(), ty, ka, ta, ss.exercise))
                else:
                    _unpack_state(run_fsm(_pack_state(), ty, ka, ta, ss.exercise))

                # Form analysis (squat only for now — deadlift/OHP stubs still return [])
                # Tier 1 Red Zone checks run always; Tier 2 baseline checks only after calibration.
                has_error = False
                if ss.exercise == "Squat":
                    baseline = (
                        {"torso_angle": ss.calib_torso_angle, "knee_angle": ss.calib_knee_angle}
                        if ss.calib_torso_angle is not None else None
                    )
                    errors = _squat.analyze(lms, baseline, ss.phase)
                    if errors:
                        ss.feedback      = errors[0]["message"]
                        ss.feedback_type = "warn"
                        has_error        = True

                frame = draw_pose(frame, result, has_error)
                frame = draw_hud(frame, ss.phase, ss.rep_count, ka, ss.fps, w, h)

                if ss.calib_mode:
                    frame = draw_calibration_bar(frame, ss.calib_reps_collected, 3, w, h)
            else:
                frame = draw_no_pose(frame)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            metric_reps.metric("REPS",        ss.rep_count)
            metric_phase.metric("PHASE",       ss.phase.capitalize())
            metric_knee.metric("KNEE ANGLE",   f"{ss.knee_angle_disp:.1f}°")
            metric_depth.metric("CALIB DEPTH", f"{ss.calib_depth:.4f}" if ss.calib_depth else "—")
            metric_fps.metric("FPS",           f"{ss.fps:.0f}")

            render_feedback()
            video_ph.image(rgb_out, channels="RGB", use_container_width=True)

            time.sleep(0.01)

    except Exception as exc:
        st.error(f"Runtime error: {exc}")
    finally:
        cap.release()
else:
    st.markdown(
        '<div class="fb fb-neutral"><i class="fa-solid fa-pause"></i>Webcam Has Been Toggled Off. Please Toggle On To Resume</div>',
        unsafe_allow_html=True,
    )
