from core.angle_calculator import (
    torso_angle, signed_torso_angle, hip_ankle_offset,
    LEFT_HIP, RIGHT_HIP,
)

RED_ZONES = {
    "torso_angle_max": 30,  # OHP: some backward lean is expected; underreporting means ~35° physical
}

DEVIATION_THRESHOLD         = 10   # degrees above calibrated peak lean — back arch during press
STANDING_HYPEREXT_THRESHOLD =  8
HIP_FORWARD_THRESHOLD       =  0.08
HIP_ABSOLUTE_THRESHOLD      =  0.12
CALIB_ARCH_THRESHOLD        = 15   # degrees — pre-calib absolute arch check


def analyze(landmarks, baseline, phase, rep_count=0):
    """
    Overhead press form analysis. Bar tracked via wrist Y.
    FSM: STANDING → ASCENDING (press) → DESCENDING (lower) → STANDING.

    Tier 1  — Spine red zone (ASCENDING / DESCENDING only)
    Fallback — Absolute checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    if phase == "STANDING":
        live_offset = hip_ankle_offset(landmarks)
        if not baseline:
            if abs(live_offset) > HIP_ABSOLUTE_THRESHOLD:
                return [{
                    "type":     "hip_forward_absolute",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Hips pushing forward — stand tall and brace your core",
                }]
        if baseline:
            bst_dir   = baseline.get("signed_torso_angle")
            direction = 1.0 if (bst_dir or 0) >= 0 else -1.0

            calib_offset = baseline.get("hip_ankle_offset")
            if calib_offset is not None:
                if direction * (live_offset - calib_offset) > HIP_FORWARD_THRESHOLD:
                    return [{
                        "type":     "hip_forward",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Hips pushing forward — brace your core and stand tall",
                    }]

            bsst = baseline.get("signed_torso_standing_angle")
            if bsst is not None:
                st  = signed_torso_angle(landmarks)
                dev = direction * (st - bsst)
                if dev < -STANDING_HYPEREXT_THRESHOLD:
                    return [{
                        "type":     "hyperextension",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Excessive lower back arch — brace your core and tuck your pelvis",
                    }]
        return []

    # ASCENDING and DESCENDING
    ta = torso_angle(landmarks)

    if ta > RED_ZONES["torso_angle_max"]:
        return [{
            "type":     "red_zone_spine",
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

    errors.sort(key=lambda x: x["priority"])
    return errors
