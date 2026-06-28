import math
import numpy as np
import mediapipe as mp

_LM = mp.solutions.pose.PoseLandmark

LEFT_SHOULDER  = _LM.LEFT_SHOULDER.value
RIGHT_SHOULDER = _LM.RIGHT_SHOULDER.value
LEFT_HIP       = _LM.LEFT_HIP.value
RIGHT_HIP      = _LM.RIGHT_HIP.value
LEFT_KNEE      = _LM.LEFT_KNEE.value
RIGHT_KNEE     = _LM.RIGHT_KNEE.value
LEFT_ANKLE     = _LM.LEFT_ANKLE.value
RIGHT_ANKLE    = _LM.RIGHT_ANKLE.value
LEFT_ELBOW     = _LM.LEFT_ELBOW.value
RIGHT_ELBOW    = _LM.RIGHT_ELBOW.value
LEFT_WRIST     = _LM.LEFT_WRIST.value
RIGHT_WRIST    = _LM.RIGHT_WRIST.value


def lm_xy(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def avg_hip_y(landmarks):
    return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0


def avg_wrist_y(landmarks):
    return (landmarks[LEFT_WRIST].y + landmarks[RIGHT_WRIST].y) / 2.0


def avg_knee_angle(landmarks):
    l = angle_between(lm_xy(landmarks, LEFT_HIP),  lm_xy(landmarks, LEFT_KNEE),  lm_xy(landmarks, LEFT_ANKLE))
    r = angle_between(lm_xy(landmarks, RIGHT_HIP), lm_xy(landmarks, RIGHT_KNEE), lm_xy(landmarks, RIGHT_ANKLE))
    return (l + r) / 2.0


def torso_angle(landmarks):
    shoulder     = (lm_xy(landmarks, LEFT_SHOULDER) + lm_xy(landmarks, RIGHT_SHOULDER)) / 2
    hip          = (lm_xy(landmarks, LEFT_HIP)       + lm_xy(landmarks, RIGHT_HIP))      / 2
    vertical_ref = np.array([hip[0], hip[1] - 1.0])
    return angle_between(shoulder, hip, vertical_ref)


def signed_torso_angle(landmarks):
    """Signed torso angle via atan2. 0° = upright. Sign convention:
    positive when shoulder is to the right of hip, negative when to the left.
    Comparing against a calibrated baseline (same sign convention) lets us detect
    both excessive forward lean AND backward hyperextension regardless of camera orientation."""
    sx = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    sy = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    hx = (landmarks[LEFT_HIP].x     + landmarks[RIGHT_HIP].x)      / 2
    hy = (landmarks[LEFT_HIP].y     + landmarks[RIGHT_HIP].y)      / 2
    return math.degrees(math.atan2(sx - hx, hy - sy))


def avg_elbow_angle(landmarks):
    l = angle_between(lm_xy(landmarks, LEFT_SHOULDER),  lm_xy(landmarks, LEFT_ELBOW),  lm_xy(landmarks, LEFT_WRIST))
    r = angle_between(lm_xy(landmarks, RIGHT_SHOULDER), lm_xy(landmarks, RIGHT_ELBOW), lm_xy(landmarks, RIGHT_WRIST))
    return (l + r) / 2.0


def tracked_y_for_exercise(landmarks, exercise):
    if exercise == "Overhead Press":
        return avg_wrist_y(landmarks)
    return avg_hip_y(landmarks)
