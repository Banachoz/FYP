import base64
import time
import cv2
import mediapipe as mp
import streamlit as st

from core.angle_calculator import avg_knee_angle, avg_elbow_angle, tracked_y_for_exercise, torso_angle, signed_torso_angle, hip_ankle_offset
from core.phase_detector import run_fsm
from core.tts_coach import TTSCoach
import exercises.squat    as _squat
import exercises.deadlift as _deadlift
import exercises.ohp      as _ohp
from ui.overlay import (
    apply_vignette, draw_pose, draw_hud,
    draw_countdown, draw_calibration_bar, draw_no_pose,
)

_TTS_ERROR_MAP = {
    "red_zone_spine":       "Stop. Forward lean",
    "forward_rounding":     "Back rounding",
    "calib_lean":           "Back rounding",
    "hips_first":           "Hips rising first",
    "calib_hips_first":     "Hips rising first",
    "knee_travel":          "Knees forward",
    "hip_forward":          "Hips pushing forward",
    "hip_forward_absolute": "Hips pushing forward",
    "calib_hip":            "Sit back and down",
    "hyperextension":       "Avoid hyperextension",
    "calib_depth":          "Go deeper",
    "hips_too_high":        "Lower your hips",
    "shoulders_behind":     "Shoulders forward",
    "bar_too_far":          "Keep the bar close",
    "back_arch":            "Avoid back arch",
    "calib_arch":           "Avoid back arch",
    "wrist_too_forward":    "Bar too far forward",
}


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
        phase="ASCENDING" if st.session_state.get("exercise", "Squat") in ("Deadlift", "Overhead Press") else "STANDING",
        rep_count=0,
        calib_mode=False,
        calib_reps_collected=0,
        calib_depths=[],
        calib_depth=None,
        calib_peak=0.0,
        calib_peak_torso=0.0,
        calib_peak_knee=180.0,
        calib_peak_signed_torso=0.0,
        calib_torso_angles=[],
        calib_knee_angles=[],
        calib_signed_torso_angles=[],
        calib_standing_signed_torso_angles=[],
        calib_torso_angle=None,
        calib_knee_angle=None,
        calib_signed_torso_angle=None,
        calib_standing_signed_torso_angle=None,
        last_signed_torso_val=0.0,
        last_hip_ankle_offset=0.0,
        calib_hip_ankle_offsets=[],
        calib_hip_ankle_offset=None,
        completed_rep_label="",
        calib_just_completed=False,
        feedback="Position yourself laterally to the camera and begin",
        feedback_type="neutral",
        exercise="Squat",
        fps=0.0,
        t_last=time.time(),
        knee_angle_disp=0.0,
        prev_tracked_y=0.0,
        standing_tracked_y=0.0,
        standing_knee_max=0.0,
        asc_min_y=99.0,
        descent_max_y=0.0,
        standing_torso_buffer=[],
        standing_hao_buffer=[],
        pre_rep_standing_torso=0.0,
        pre_rep_standing_hao=0.0,
        bottom_knee_ref=0.0,
        dl_at_bottom=False,
        dl_reached_lockout=False,
        sq_descent_min_knee=180.0,
        ohp_bottom_ref=0.0,
        feedback_hold_until=0.0,
        held_feedback="",
        current_rep_errors=[],
        rep_log=[],
        calib_ever_started=False,
        countdown_duration=5,
        rep_target=0,
        set_just_completed=False,
        set_stopped=False,
        voice_enabled=True,
        analysis_countdown=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ss = st.session_state


def _initial_phase():
    return "ASCENDING" if ss.exercise in ("Deadlift", "Overhead Press") else "STANDING"


@st.cache_resource
def _get_tts_coach() -> TTSCoach:
    return TTSCoach()

_tts = _get_tts_coach()


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
        "calib_peak_torso":          ss.calib_peak_torso,
        "calib_peak_knee":           ss.calib_peak_knee,
        "calib_peak_signed_torso":   ss.calib_peak_signed_torso,
        "calib_torso_angles":        ss.calib_torso_angles,
        "calib_knee_angles":         ss.calib_knee_angles,
        "calib_signed_torso_angles":          ss.calib_signed_torso_angles,
        "calib_standing_signed_torso_angles": ss.calib_standing_signed_torso_angles,
        "calib_torso_angle":                  ss.calib_torso_angle,
        "calib_knee_angle":                   ss.calib_knee_angle,
        "calib_signed_torso_angle":           ss.calib_signed_torso_angle,
        "calib_standing_signed_torso_angle":  ss.calib_standing_signed_torso_angle,
        "last_signed_torso_val":              ss.last_signed_torso_val,
        "last_hip_ankle_offset":              ss.last_hip_ankle_offset,
        "calib_hip_ankle_offsets":            ss.calib_hip_ankle_offsets,
        "calib_hip_ankle_offset":             ss.calib_hip_ankle_offset,
        "completed_rep_label":                ss.completed_rep_label,
        "feedback":            ss.feedback,
        "feedback_type":       ss.feedback_type,
        "prev_tracked_y":      ss.prev_tracked_y,
        "standing_tracked_y":  ss.standing_tracked_y,
        "standing_knee_max":   ss.standing_knee_max,
        "asc_min_y":           ss.asc_min_y,
        "descent_max_y":       ss.descent_max_y,
        "standing_torso_buffer":  ss.standing_torso_buffer,
        "standing_hao_buffer":    ss.standing_hao_buffer,
        "pre_rep_standing_torso": ss.pre_rep_standing_torso,
        "pre_rep_standing_hao":   ss.pre_rep_standing_hao,
        "bottom_knee_ref":        ss.bottom_knee_ref,
        "dl_at_bottom":           ss.dl_at_bottom,
        "dl_reached_lockout":     ss.dl_reached_lockout,
        "sq_descent_min_knee":    ss.sq_descent_min_knee,
        "ohp_bottom_ref":         ss.ohp_bottom_ref,
    }

def _unpack_state(result):
    ss.phase                = result["phase"]
    ss.rep_count            = result["rep_count"]
    ss.calib_mode           = result["calib_mode"]
    ss.calib_reps_collected = result["calib_reps_collected"]
    ss.calib_depths         = result["calib_depths"]
    ss.calib_depth          = result["calib_depth"]
    ss.calib_peak           = result["calib_peak"]
    ss.calib_peak_torso          = result["calib_peak_torso"]
    ss.calib_peak_knee           = result["calib_peak_knee"]
    ss.calib_peak_signed_torso   = result["calib_peak_signed_torso"]
    ss.calib_torso_angles        = result["calib_torso_angles"]
    ss.calib_knee_angles         = result["calib_knee_angles"]
    ss.calib_signed_torso_angles          = result["calib_signed_torso_angles"]
    ss.calib_standing_signed_torso_angles = result["calib_standing_signed_torso_angles"]
    ss.calib_torso_angle                  = result["calib_torso_angle"]
    ss.calib_knee_angle                   = result["calib_knee_angle"]
    ss.calib_signed_torso_angle           = result["calib_signed_torso_angle"]
    ss.calib_standing_signed_torso_angle  = result["calib_standing_signed_torso_angle"]
    ss.last_signed_torso_val              = result["last_signed_torso_val"]
    ss.last_hip_ankle_offset              = result["last_hip_ankle_offset"]
    ss.calib_hip_ankle_offsets            = result["calib_hip_ankle_offsets"]
    ss.calib_hip_ankle_offset             = result["calib_hip_ankle_offset"]
    ss.completed_rep_label                = result["completed_rep_label"]
    ss.feedback             = result["feedback"]
    ss.feedback_type        = result["feedback_type"]
    ss.prev_tracked_y       = result["prev_tracked_y"]
    ss.standing_tracked_y   = result["standing_tracked_y"]
    ss.standing_knee_max    = result["standing_knee_max"]
    ss.asc_min_y              = result["asc_min_y"]
    ss.descent_max_y          = result["descent_max_y"]
    ss.standing_torso_buffer  = result["standing_torso_buffer"]
    ss.standing_hao_buffer    = result["standing_hao_buffer"]
    ss.pre_rep_standing_torso = result["pre_rep_standing_torso"]
    ss.pre_rep_standing_hao   = result["pre_rep_standing_hao"]
    ss.bottom_knee_ref        = result["bottom_knee_ref"]
    ss.dl_at_bottom           = result["dl_at_bottom"]
    ss.dl_reached_lockout     = result["dl_reached_lockout"]
    ss.sq_descent_min_knee    = result["sq_descent_min_knee"]
    ss.ohp_bottom_ref         = result["ohp_bottom_ref"]
    # Append rep log entry here — fires before st.rerun() so Calib 3 is never missed
    label = result["completed_rep_label"]
    if label:
        summary = f"{label} — {' · '.join(ss.current_rep_errors)}" if ss.current_rep_errors else f"{label} — Good form"
        ss.rep_log.append(summary)
        ss.current_rep_errors = []
    if result["needs_rerun"]:
        ss.calib_just_completed = True
        if ss.voice_enabled:
            _tts.speak("Calibration complete")
        st.rerun()


# CALIBRATION COMPLETE — inline page (replaces @st.dialog to avoid Streamlit fragment close bug)
def _calib_done_dialog():
    st.markdown(
        '<div class="fb fb-ok"><i class="fa-solid fa-circle-check"></i>'
        f'CALIBRATION COMPLETE — {ss.exercise} baseline locked in</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    calib_entries = [e for e in ss.rep_log if e.startswith("Calib")][-3:]
    if calib_entries:
        st.markdown("**Calibration rep summary:**")
        for entry in calib_entries:
            is_good = "Good form" in entry
            icon, color = ("✓", "green") if is_good else ("✗", "red")
            st.markdown(f":{color}[{icon} {entry}]")
        errors_in_calib = sum(1 for e in calib_entries if "Good form" not in e)
        if errors_in_calib >= 2:
            st.warning("Form errors in 2+ calibration reps — consider re-calibrating with better form.")
    st.markdown("---")
    rep_target_val = st.number_input(
        "Target reps per set (0 = unlimited)",
        min_value=0, max_value=200, value=ss.rep_target if ss.rep_target > 0 else 3, step=1,
        help="The system will pause and prompt you when you reach this count. Set 0 for no limit.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Re-calibrate", use_container_width=True):
            ss.calib_mode                         = True
            ss.calib_reps_collected               = 0
            ss.calib_depths                       = []
            ss.calib_depth                        = None
            ss.calib_peak                         = 0.0
            ss.calib_peak_torso                   = 0.0
            ss.calib_peak_knee                    = 180.0
            ss.calib_peak_signed_torso            = 0.0
            ss.calib_torso_angles                 = []
            ss.calib_knee_angles                  = []
            ss.calib_signed_torso_angles          = []
            ss.calib_standing_signed_torso_angles = []
            ss.calib_torso_angle                  = None
            ss.calib_knee_angle                   = None
            ss.calib_signed_torso_angle           = None
            ss.calib_standing_signed_torso_angle  = None
            ss.last_signed_torso_val              = 0.0
            ss.last_hip_ankle_offset              = 0.0
            ss.calib_hip_ankle_offsets            = []
            ss.calib_hip_ankle_offset             = None
            ss.calib_just_completed               = False
            ss.rep_count            = 0
            ss.current_rep_errors   = []
            ss.rep_log              = []
            ss.rep_target           = 0
            ss.bottom_knee_ref      = 0.0
            ss.dl_reached_lockout   = False
            ss.sq_descent_min_knee  = 180.0
            ss.ohp_bottom_ref       = 0.0
            ss.phase                = _initial_phase()
            ss.is_counting_down     = True
            ss.countdown_start      = time.time()
            ss.feedback             = f"Get into position — calibration starts in {ss.countdown_duration}s"
            ss.feedback_type        = "calib"
            st.rerun()
    with col_b:
        if st.button("Begin Analysis", type="primary", use_container_width=True):
            ss.rep_target           = rep_target_val
            ss.calib_just_completed = False
            ss.rep_count            = 0
            ss.current_rep_errors   = []
            ss.rep_log              = []
            ss.phase                = _initial_phase()
            ss.is_counting_down     = True
            ss.countdown_start      = time.time()
            ss.analysis_countdown   = True
            ss.feedback             = f"Get ready — analysis starts in {ss.countdown_duration}s"
            ss.feedback_type        = "calib"
            st.rerun()


# SET COMPLETE — inline page (replaces @st.dialog to avoid Streamlit fragment close bug)
def _set_done_dialog():
    st.markdown(
        '<div class="fb fb-ok"><i class="fa-solid fa-circle-check"></i>'
        f'SET COMPLETE — {ss.rep_target} rep{"s" if ss.rep_target != 1 else ""} done. Great work!</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    analysis_entries = [e for e in ss.rep_log if e.startswith("Rep")]
    if analysis_entries:
        st.markdown("**Set rep summary:**")
        for entry in analysis_entries:
            is_good = "Good form" in entry
            icon, color = ("✓", "green") if is_good else ("✗", "red")
            st.markdown(f":{color}[{icon} {entry}]")
    st.markdown("---")
    new_target = st.number_input(
        "Reps for next set",
        min_value=1, max_value=200, value=ss.rep_target, step=1,
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Same Set", use_container_width=True):
            ss.rep_count          = 0
            ss.current_rep_errors = []
            ss.rep_log            = []
            ss.set_just_completed = False
            ss.phase              = _initial_phase()
            ss.is_counting_down   = True
            ss.countdown_start    = time.time()
            ss.analysis_countdown = True
            ss.feedback           = f"Get ready — next set starts in {ss.countdown_duration}s"
            ss.feedback_type      = "calib"
            st.rerun()
    with col_b:
        if st.button("New Count", type="primary", use_container_width=True):
            ss.rep_target         = new_target
            ss.rep_count          = 0
            ss.current_rep_errors = []
            ss.rep_log            = []
            ss.set_just_completed = False
            ss.phase              = _initial_phase()
            ss.is_counting_down   = True
            ss.countdown_start    = time.time()
            ss.analysis_countdown = True
            ss.feedback           = f"Get ready — next set starts in {ss.countdown_duration}s"
            ss.feedback_type      = "calib"
            st.rerun()
    with col_c:
        if st.button("Finish", use_container_width=True):
            ss.set_just_completed = False
            ss.rep_target         = 0
            st.rerun()


# SET STOPPED — user hit Stop mid-set
def _stopped_dialog():
    st.markdown(
        '<div class="fb fb-warn"><i class="fa-solid fa-hand"></i>'
        f'SET STOPPED — {ss.rep_count} rep{"s" if ss.rep_count != 1 else ""} completed</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    analysis_entries = [e for e in ss.rep_log if e.startswith("Rep")]
    if analysis_entries:
        st.markdown("**Reps completed:**")
        for entry in analysis_entries:
            is_good = "Good form" in entry
            icon, color = ("✓", "green") if is_good else ("✗", "red")
            st.markdown(f":{color}[{icon} {entry}]")
    st.markdown("---")
    new_target = st.number_input(
        "Target reps for next set",
        min_value=1, max_value=200, value=max(ss.rep_target, 3), step=1,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Restart Analysis", type="primary", use_container_width=True):
            ss.rep_target         = new_target
            ss.rep_count          = 0
            ss.current_rep_errors = []
            ss.rep_log            = []
            ss.set_stopped        = False
            ss.phase              = _initial_phase()
            ss.is_counting_down   = True
            ss.countdown_start    = time.time()
            ss.analysis_countdown = True
            ss.feedback           = f"Get ready — analysis starts in {ss.countdown_duration}s"
            ss.feedback_type      = "calib"
            st.rerun()
    with col_b:
        if st.button("Re-calibrate", use_container_width=True):
            ss.set_stopped                        = False
            ss.calib_mode                         = True
            ss.calib_reps_collected               = 0
            ss.calib_depths                       = []
            ss.calib_depth                        = None
            ss.calib_peak                         = 0.0
            ss.calib_peak_torso                   = 0.0
            ss.calib_peak_knee                    = 180.0
            ss.calib_peak_signed_torso            = 0.0
            ss.calib_torso_angles                 = []
            ss.calib_knee_angles                  = []
            ss.calib_signed_torso_angles          = []
            ss.calib_standing_signed_torso_angles = []
            ss.calib_torso_angle                  = None
            ss.calib_knee_angle                   = None
            ss.calib_signed_torso_angle           = None
            ss.calib_standing_signed_torso_angle  = None
            ss.last_signed_torso_val              = 0.0
            ss.last_hip_ankle_offset              = 0.0
            ss.calib_hip_ankle_offsets            = []
            ss.calib_hip_ankle_offset             = None
            ss.calib_just_completed               = False
            ss.rep_count            = 0
            ss.current_rep_errors   = []
            ss.rep_log              = []
            ss.rep_target           = 0
            ss.bottom_knee_ref      = 0.0
            ss.dl_reached_lockout   = False
            ss.sq_descent_min_knee  = 180.0
            ss.ohp_bottom_ref       = 0.0
            ss.phase                = _initial_phase()
            ss.is_counting_down     = True
            ss.countdown_start      = time.time()
            ss.feedback             = f"Get into position — calibration starts in {ss.countdown_duration}s"
            ss.feedback_type        = "calib"
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
    countdown_secs = st.slider(
        "Countdown (s)", min_value=3, max_value=10,
        value=ss.countdown_duration, step=1,
        help="Seconds before calibration begins after clicking Calibrate",
    )
    ss.countdown_duration = countdown_secs

    col_c, col_r = st.columns(2)
    with col_c:
        if st.button("Calibrate", use_container_width=True):
            ss.calib_ever_started   = True
            ss.calib_mode           = True
            ss.calib_reps_collected = 0
            ss.calib_depths         = []
            ss.calib_depth          = None
            ss.calib_peak           = 0.0
            ss.bottom_knee_ref      = 0.0
            ss.dl_reached_lockout   = False
            ss.sq_descent_min_knee  = 180.0
            ss.ohp_bottom_ref       = 0.0
            ss.rep_count            = 0
            ss.phase                = _initial_phase()
            ss.is_counting_down     = True
            ss.countdown_start      = time.time()
            ss.feedback             = f"Get into position — calibration starts in {ss.countdown_duration}s"
            ss.feedback_type        = "calib"
    with col_r:
        if st.button("Reset", use_container_width=True):
            for key in list(ss.keys()):
                del ss[key]
            _init_state()
            st.rerun()

    # STOP button — only visible during active analysis
    _analysis_active = (
        ss.calib_depth is not None
        and not ss.calib_mode
        and not ss.calib_just_completed
        and not ss.set_just_completed
        and not ss.set_stopped
    )
    if _analysis_active:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⏹ Stop Set", use_container_width=True):
            ss.set_stopped = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label"><i class="fa-solid fa-volume-high"></i> Voice</div>', unsafe_allow_html=True)
    ss.voice_enabled = st.toggle("Voice coaching", value=ss.voice_enabled, label_visibility="collapsed")

    rep_log_ph = st.empty()


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

if ss.set_just_completed:
    _set_done_dialog()
    st.stop()

if ss.set_stopped:
    _stopped_dialog()
    st.stop()

if not ss.calib_ever_started:
    feedback_ph.markdown(
        '<div class="fb fb-calib"><i class="fa-solid fa-crosshairs"></i>'
        'CALIBRATION REQUIRED — Click <strong>Calibrate</strong> in the sidebar to begin</div>',
        unsafe_allow_html=True,
    )
    webcam_active = False
else:
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
        _prev_reps          = -1
        _prev_phase         = ""
        _prev_knee          = -1.0
        _prev_depth         = "UNSET"
        _prev_fps           = -1.0
        _prev_feedback      = ""
        _prev_feedback_type = ""
        _prev_log_len       = -1
        _prev_counting      = None
        _prev_tts_phase       = ""
        _spoken_this_phase    = False
        _spoken_depth_warning = False
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

                ty  = tracked_y_for_exercise(lms, ss.exercise)
                ka  = avg_knee_angle(lms)
                _joint = avg_elbow_angle(lms) if ss.exercise == "Overhead Press" else ka
                ta  = torso_angle(lms)
                sta = signed_torso_angle(lms)
                hao = hip_ankle_offset(lms)
                ss.knee_angle_disp = round(ka, 1)

                if ss.is_counting_down:
                    time_left = ss.countdown_duration - (now - ss.countdown_start)
                    if time_left > 0:
                        frame = draw_countdown(frame, time_left, w, h)
                    else:
                        ss.is_counting_down  = False
                        # Sync prev_tracked_y to current position so delta = 0 on
                        # the first FSM call — prevents spurious hip_trigger from
                        # the 0.0 init value or stale position after the countdown.
                        # Reset standing_knee_max so a stale accumulated max from
                        # the pre-countdown standing period can't fire knee_trigger.
                        ss.prev_tracked_y    = ty
                        ss.standing_knee_max = 0.0
                        if ss.analysis_countdown:
                            ss.analysis_countdown = False
                            ss.feedback      = "Analysing form — begin your set"
                            ss.feedback_type = "ok"
                            if ss.voice_enabled:
                                _tts.speak("Ready")
                        else:
                            ss.feedback      = "Calibration active — perform 3 full reps"
                            ss.feedback_type = "calib"
                            if ss.voice_enabled:
                                _tts.speak("Calibration started")
                        _unpack_state(run_fsm(_pack_state(), ty, _joint, ta, ss.exercise, sta, hao))
                    # During countdown: draw skeleton/HUD but suppress all analysis UI
                    frame = draw_pose(frame, result, False)
                    frame = draw_hud(frame, ss.phase, ss.rep_count, ka, ta, ss.fps, w, h)
                else:
                    _unpack_state(run_fsm(_pack_state(), ty, _joint, ta, ss.exercise, sta, hao))

                    # Pause for set-complete dialog when target is reached
                    if ss.rep_target > 0 and not ss.calib_mode and ss.rep_count >= ss.rep_target:
                        ss.set_just_completed = True
                        if ss.voice_enabled:
                            _tts.speak("Set complete")
                        st.rerun()

                    fsm_feedback_type = ss.feedback_type

                    # Form analysis — Squat, Deadlift, Overhead Press.
                    # Tier 1 Red Zone checks run always; Tier 2 baseline checks only after calibration.
                    has_error      = False
                    _top_error_type = None
                    if ss.exercise in ("Squat", "Deadlift", "Overhead Press"):
                        baseline = (
                            {
                                "torso_angle":                 ss.calib_torso_angle,
                                "knee_angle":                  ss.calib_knee_angle,
                                "signed_torso_angle":          ss.calib_signed_torso_angle,
                                "signed_torso_standing_angle": ss.calib_standing_signed_torso_angle,
                                "hip_ankle_offset":            ss.calib_hip_ankle_offset,
                            }
                            if ss.calib_torso_angle is not None else None
                        )
                        if ss.exercise == "Squat":
                            # calib_peak_signed_torso resets to 0 when a rep completes;
                            # use last completed rep's bottom angle instead (persists across reps).
                            _sq_ref = (
                                ss.calib_signed_torso_angles[-1]
                                if ss.calib_mode and ss.calib_signed_torso_angles
                                else ss.calib_peak_signed_torso if ss.calib_mode else None
                            )
                            errors = _squat.analyze(lms, baseline, ss.phase, ss.rep_count,
                                                    bottom_ref_torso=_sq_ref)
                        elif ss.exercise == "Deadlift":
                            errors = _deadlift.analyze(lms, baseline, ss.phase, ss.rep_count,
                                                       bottom_ref_torso=ss.calib_peak_signed_torso if ss.calib_mode else None)
                        else:
                            errors = _ohp.analyze(lms, baseline, ss.phase, ss.rep_count)
                        if errors:
                            _top_error_type  = errors[0]["type"]
                            ss.feedback      = errors[0]["message"]
                            ss.feedback_type = "warn"
                            has_error        = True
                            ss.feedback_hold_until = now + 1.5
                            ss.held_feedback = ss.feedback
                            if ss.feedback not in ss.current_rep_errors:
                                ss.current_rep_errors.append(ss.feedback)

                    # Sticky hold: keep error visible if FSM would replace it with a neutral cue
                    if not has_error and now < ss.feedback_hold_until:
                        if fsm_feedback_type == "neutral":
                            ss.feedback      = ss.held_feedback
                            ss.feedback_type = "warn"
                            has_error        = True

                    # TTS — phase tracking + first error per phase
                    if ss.phase != _prev_tts_phase:
                        _prev_tts_phase       = ss.phase
                        _spoken_this_phase    = False
                        _spoken_depth_warning = False
                    if ss.feedback == "Insufficient depth — go lower before ascending":
                        if not _spoken_depth_warning and ss.voice_enabled:
                            _tts.speak("Go deeper")
                            _spoken_depth_warning = True
                    else:
                        _spoken_depth_warning = False
                    if has_error and not _spoken_this_phase and ss.voice_enabled:
                        cue = _TTS_ERROR_MAP.get(_top_error_type)
                        if cue:
                            _tts.speak(cue)
                            _spoken_this_phase = True

                    frame = draw_pose(frame, result, has_error)
                    frame = draw_hud(frame, ss.phase, ss.rep_count, ka, ta, ss.fps, w, h)

                    if ss.calib_mode:
                        frame = draw_calibration_bar(frame, ss.calib_reps_collected, 3, w, h)
            else:
                frame = draw_no_pose(frame)

            # Metrics — only push to browser when the displayed value changes
            if ss.rep_count != _prev_reps:
                metric_reps.metric("REPS", ss.rep_count)
                _prev_reps = ss.rep_count
            if ss.phase != _prev_phase:
                metric_phase.metric("PHASE", ss.phase.capitalize())
                _prev_phase = ss.phase
            if round(ss.knee_angle_disp) != round(_prev_knee):
                metric_knee.metric("KNEE ANGLE", f"{ss.knee_angle_disp:.1f}°")
                _prev_knee = ss.knee_angle_disp
            if ss.calib_depth != _prev_depth:
                metric_depth.metric("CALIB DEPTH", f"{ss.calib_depth:.4f}" if ss.calib_depth else "—")
                _prev_depth = ss.calib_depth
            if round(ss.fps) != round(_prev_fps):
                metric_fps.metric("FPS", f"{ss.fps:.0f}")
                _prev_fps = ss.fps

            # Rep log — only when log grows or countdown state toggles
            _log_key = (len(ss.rep_log), ss.is_counting_down)
            if _log_key != (_prev_log_len, _prev_counting):
                if ss.is_counting_down:
                    rep_log_ph.empty()
                elif ss.rep_log:
                    _lines = []
                    for _entry in ss.rep_log[-8:]:
                        _good  = "Good form" in _entry
                        _icon  = "fa-circle-check" if _good else "fa-triangle-exclamation"
                        _color = "var(--ok)" if _good else "var(--danger)"
                        _lines.append(
                            f'<div class="status-row" style="margin-bottom:6px;color:{_color}">'
                            f'<i class="fa-solid {_icon}" style="font-size:0.55rem"></i>{_entry}</div>'
                        )
                    rep_log_ph.markdown(
                        '<div class="section-label" style="margin-top:14px">'
                        '<i class="fa-solid fa-list-check"></i> Rep Log</div>' + "".join(_lines),
                        unsafe_allow_html=True,
                    )
                _prev_log_len  = len(ss.rep_log)
                _prev_counting = ss.is_counting_down

            # Feedback box — only when message or type changes
            if ss.feedback != _prev_feedback or ss.feedback_type != _prev_feedback_type:
                if not ss.is_counting_down:
                    render_feedback()
                else:
                    feedback_ph.empty()
                _prev_feedback      = ss.feedback
                _prev_feedback_type = ss.feedback_type

            # Video — JPEG base64 is 5-10x smaller than Streamlit's default PNG encode
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64   = base64.b64encode(jpg).decode()
            video_ph.markdown(
                f'<img src="data:image/jpeg;base64,{b64}"'
                ' style="width:100%;border-radius:2px;border:1px solid var(--border)">',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.error(f"Runtime error: {exc}")
    finally:
        cap.release()
else:
    st.markdown(
        '<div class="fb fb-neutral"><i class="fa-solid fa-pause"></i>Webcam Has Been Toggled Off. Please Toggle On To Resume</div>',
        unsafe_allow_html=True,
    )
