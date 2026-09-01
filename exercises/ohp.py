from core.angle_calculator import (
    torso_angle, signed_torso_angle, hip_ankle_offset,
    wrist_shoulder_offset, facing_direction, avg_knee_angle,
    LEFT_WRIST, RIGHT_WRIST,
)

RED_ZONES = {
    "torso_angle_max": 10,  # OHP: some backward lean is expected
}

DEVIATION_THRESHOLD         = 3   # degrees above calibrated peak lean — back arch during press
STANDING_HYPEREXT_THRESHOLD =  5
HIP_FORWARD_THRESHOLD       =  0.08
HIP_ABSOLUTE_THRESHOLD      =  0.11
CALIB_ARCH_THRESHOLD        =  5.5   # pre-calib absolute arch check
WRIST_FORWARD_THRESHOLD     =  0.10 # wrists in front of shoulders at lockout, raise to make it looser
WRIST_VIS_MIN               =  0.40 # MediaPipe visibility cutoff
KNEE_BEND_THRESHOLD         = 173   # degrees — knees must stay near-straight during press


def analyze(landmarks, baseline, phase):
    """
    Overhead press form analysis. Bar tracked via wrist Y.
    FSM: STANDING → ASCENDING (press) → DESCENDING (lower) → STANDING.

    Tier 1  — Spine red zone (ASCENDING / DESCENDING only)
    Fallback — Absolute checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    if phase == "STANDING":
        # Direction from foot landmarks — reliable for OHP because calib_signed_torso_angle
        # captures backward lean (OHP arch), which has the opposite sign to the forward lean
        # used by squat/deadlift. Foot-based direction has no such ambiguity.
        direction = facing_direction(landmarks)

        live_offset = hip_ankle_offset(landmarks)
        if not baseline:
            if direction is not None:
                hip_fwd_condition = direction * live_offset > HIP_ABSOLUTE_THRESHOLD
            else:
                hip_fwd_condition = abs(live_offset) > HIP_ABSOLUTE_THRESHOLD
            if hip_fwd_condition:
                return [{
                    "type":     "hip_forward_absolute",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Hips pushing forward — stand tall and brace your core",
                }]
        if baseline:
            calib_offset = baseline.get("hip_ankle_offset")
            if calib_offset is not None and direction is not None:
                if direction * (live_offset - calib_offset) > HIP_FORWARD_THRESHOLD:
                    return [{
                        "type":     "hip_forward",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Hips pushing forward — brace your core and stand tall",
                    }]

            bsst = baseline.get("signed_torso_standing_angle")
            if bsst is not None and direction is not None:
                st  = signed_torso_angle(landmarks)
                dev = direction * (st - bsst)
                if dev < -STANDING_HYPEREXT_THRESHOLD:
                    return [{
                        "type":     "hyperextension",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Excessive lower back arch — brace your core and tuck your pelvis",
                    }]

        # Wrist position check — bar should be directly overhead or slightly behind shoulders
        lw_vis = landmarks[LEFT_WRIST].visibility
        rw_vis = landmarks[RIGHT_WRIST].visibility
        if min(lw_vis, rw_vis) > WRIST_VIS_MIN and direction is not None:
            wso = wrist_shoulder_offset(landmarks)
            if direction * wso > WRIST_FORWARD_THRESHOLD:
                return [{
                    "type":     "wrist_too_forward",
                    "priority": 2,
                    "severity": "warning",
                    "message":  "Bar too far forward — press directly overhead or slightly behind",
                }]
        return []

    # ASCENDING and DESCENDING
    ta = torso_angle(landmarks)

    if ta > RED_ZONES["torso_angle_max"]:
        return [{
            "type":     "red_zone_arch",
            "priority": 0,
            "severity": "critical",
            "message":  "CRITICAL — Severe back arch. Lower the bar immediately",
        }]

    errors = []

    if phase in ("ASCENDING", "DESCENDING"):
        if not baseline:
            if ta > CALIB_ARCH_THRESHOLD:
                errors.append({
                    "type":     "calib_arch",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Excessive back arch — brace your core and keep a neutral spine",
                })
        else:
            calib_ta = baseline.get("torso_angle")
            if calib_ta is not None and ta > calib_ta + DEVIATION_THRESHOLD:
                errors.append({
                    "type":     "back_arch",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Excessive back arch — brace your core and keep a neutral spine",
                })

    if phase == "ASCENDING":
        ka = avg_knee_angle(landmarks)
        if ka < KNEE_BEND_THRESHOLD:
            errors.append({
                "type":     "knee_bend",
                "priority": 2,
                "severity": "warning",
                "message":  "Knees bending — avoid using leg drive on a strict press",
            })

    errors.sort(key=lambda x: x["priority"])
    return errors
