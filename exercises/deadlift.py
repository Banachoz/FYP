from core.angle_calculator import (
    torso_angle, signed_torso_angle, hip_ankle_offset, shoulder_ankle_offset,
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER,
)

RED_ZONES = {
    "torso_angle_max": 80,  # higher than squat (65) — conventional DL requires significant forward lean
}

DEVIATION_THRESHOLD          = 10
STANDING_HYPEREXT_THRESHOLD  =  8
HIP_FORWARD_THRESHOLD        =  0.08
HIP_ABSOLUTE_THRESHOLD       =  0.12
SHOULDER_ANKLE_THRESHOLD     =  0.05  # torso-normalised
SETUP_TORSO_MIN              = 45     # degrees — gate for hip-shoulder height check at bottom


def analyze(landmarks, baseline, phase, rep_count=0):
    """
    Conventional deadlift form analysis. FSM starts in ASCENDING (user grips bar at bottom).
    Rep counted at lockout (knee extension). Ankle used as bar-position proxy.

    Tier 1  — Spine red zone (universal, early return)
    Fallback — Absolute checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    errors = []
    ta = torso_angle(landmarks)

    # Tier 1 — Spine red zone (all phases)
    if ta > RED_ZONES["torso_angle_max"]:
        return [{
            "type":     "red_zone_spine",
            "priority": 0,
            "severity": "critical",
            "message":  "CRITICAL — Dangerous forward lean. Stop the lift immediately",
        }]

    # STANDING — lockout checks
    if phase == "STANDING":
        live_offset = hip_ankle_offset(landmarks)
        if not baseline:
            if abs(live_offset) > HIP_ABSOLUTE_THRESHOLD:
                return [{
                    "type":     "hip_forward_absolute",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Hips pushing forward — stand tall at lockout",
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
                        "message":  "Hips pushing forward at lockout — brace your core and stand tall",
                    }]

            if rep_count > 0:
                bsst = baseline.get("signed_torso_standing_angle")
                if bsst is not None:
                    st  = signed_torso_angle(landmarks)
                    dev = direction * (st - bsst)
                    if dev < -STANDING_HYPEREXT_THRESHOLD:
                        return [{
                            "type":     "hyperextension",
                            "priority": 1,
                            "severity": "warning",
                            "message":  "Hyperextension at lockout — brace your core and stand tall",
                        }]
        return []

    # ASCENDING and DESCENDING — movement checks
    bst       = baseline.get("signed_torso_angle") if baseline else None
    direction = 1.0 if (bst or 0) >= 0 else -1.0

    # Shoulders behind bar (proxy: ankles) — both phases
    sao = shoulder_ankle_offset(landmarks)
    if direction * sao < -SHOULDER_ANKLE_THRESHOLD:
        errors.append({
            "type":     "shoulders_behind",
            "priority": 2,
            "severity": "warning",
            "message":  "Shoulders behind the bar — keep shoulders over or in front of the bar",
        })

    if phase == "ASCENDING":
        hip_y      = (landmarks[LEFT_HIP].y      + landmarks[RIGHT_HIP].y)      / 2
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2

        # Hips too high at setup: when torso is still heavily inclined, hips must be below shoulders
        if ta > SETUP_TORSO_MIN and hip_y <= shoulder_y:
            errors.append({
                "type":     "hips_too_high",
                "priority": 1,
                "severity": "warning",
                "message":  "Hips too high at setup — lower your hips before initiating the pull",
            })

        if baseline and bst is not None:
            st         = signed_torso_angle(landmarks)
            signed_dev = direction * (st - bst)
            if signed_dev > DEVIATION_THRESHOLD:
                errors.append({
                    "type":     "hips_first",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Hips rising too fast — drive chest and hips up together",
                })
        elif not baseline and ta > 55:
            errors.append({
                "type":     "calib_lean",
                "priority": 1,
                "severity": "warning",
                "message":  "Excessive forward lean during pull — maintain a neutral spine",
            })

    if phase == "DESCENDING":
        if baseline and bst is not None:
            st         = signed_torso_angle(landmarks)
            signed_dev = direction * (st - bst)
            if signed_dev > DEVIATION_THRESHOLD:
                errors.append({
                    "type":     "forward_rounding",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Back rounding on the way down — maintain a neutral spine",
                })
        elif not baseline and ta > 55:
            errors.append({
                "type":     "calib_lean",
                "priority": 1,
                "severity": "warning",
                "message":  "Back rounding on descent — keep a neutral spine",
            })

    errors.sort(key=lambda x: x["priority"])
    return errors
