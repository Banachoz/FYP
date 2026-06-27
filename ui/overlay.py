import math
import cv2
import mediapipe as mp

_mp_drawing = mp.solutions.drawing_utils
_mp_pose    = mp.solutions.pose

_PHASE_COLOURS = {
    "STANDING":  (232, 240,   0),
    "DESCENDING":(  0, 165, 255),
    "ASCENDING": (  0, 217, 126),
}


def apply_vignette(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (9, 11, 14), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    return frame


def draw_pose(frame, result, has_error=False):
    colour = (0, 0, 255) if has_error else (0, 230, 232)
    _mp_drawing.draw_landmarks(
        frame,
        result.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=_mp_drawing.DrawingSpec(color=colour, thickness=2, circle_radius=3),
        connection_drawing_spec=_mp_drawing.DrawingSpec(color=(200, 210, 220), thickness=1),
    )
    return frame


def draw_hud(frame, phase, rep_count, knee_angle_val, fps, w, h):
    phase_col = _PHASE_COLOURS.get(phase, (180, 180, 180))
    cv2.line(frame, (0, 0), (w, 0), (232, 240, 0), 2)
    cv2.putText(frame, phase,                   (12,  28), cv2.FONT_HERSHEY_DUPLEX,  0.70, phase_col,      1, cv2.LINE_AA)
    cv2.putText(frame, f"REP {rep_count}",      (12,  54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 205, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"KNEE {knee_angle_val:.0f} deg", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 100, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} fps",        (w - 68, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 62, 78), 1, cv2.LINE_AA)
    return frame


def draw_countdown(frame, time_left, w, h):
    txt = str(math.ceil(time_left))
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.5, 5)
    cv2.putText(frame, txt, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_DUPLEX, 3.5, (232, 240, 0), 5, cv2.LINE_AA)
    sub = "GET INTO POSITION"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, sub, ((w - sw) // 2, (h + th) // 2 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 165, 180), 1, cv2.LINE_AA)
    return frame


def draw_calibration_bar(frame, reps_collected, total_reps, w, h):
    filled = int((reps_collected / total_reps) * w)
    cv2.rectangle(frame, (0, h - 4), (w, h),      (22, 26, 36),   -1)
    cv2.rectangle(frame, (0, h - 4), (filled, h), (232, 240,  0), -1)
    label = f"CALIBRATING {reps_collected}/{total_reps}"
    cv2.putText(frame, label, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (232, 240, 0), 1, cv2.LINE_AA)
    return frame


def draw_no_pose(frame):
    cv2.putText(frame, "NO POSE DETECTED", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 59, 59), 2, cv2.LINE_AA)
    return frame
