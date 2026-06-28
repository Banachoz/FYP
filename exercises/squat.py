from core.angle_calculator import (
    torso_angle, signed_torso_angle, avg_knee_angle,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)

# Torso angle measured from VERTICAL (0° = upright, 90° = parallel to floor).
# Knee angle is the interior angle at the knee (180° = fully extended, ~70° = deep squat).
RED_ZONES = {
    "torso_angle_max": 65,   # torso nearly horizontal — spinal danger
}

DEVIATION_THRESHOLD          = 10   # degrees from personal baseline (descent checks)
STANDING_HYPEREXT_THRESHOLD  =  8   # degrees — tighter, standing comparison is position-matched


def analyze(landmarks, baseline, phase):
    """
    Returns a priority-sorted list of form error dicts. Caller uses errors[0] only.

    Tier 1  — Spine red zone (universal, early return)
    Fallback — Absolute reference checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    if phase == "STANDING":
        # Only hyperextension/APT check at lockout — compare standing-to-standing baseline
        if baseline:
            bsst = baseline.get("signed_torso_standing_angle")
            if bsst is not None:
                st        = signed_torso_angle(landmarks)
                direction = 1.0 if bsst >= 0 else -1.0
                dev       = direction * (st - bsst)
                if dev < -STANDING_HYPEREXT_THRESHOLD:
                    return [{
                        "type":     "hyperextension",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Hyperextension at lockout — brace your core and tuck your pelvis",
                    }]
        return []

    errors = []
    ta = torso_angle(landmarks)
    ka = avg_knee_angle(landmarks)

    # ── Tier 1: Spine Red Zone ────────────────────────────────────────────────
    if ta > RED_ZONES["torso_angle_max"]:
        return [{
            "type":     "red_zone_spine",
            "priority": 0,
            "severity": "critical",
            "message":  "CRITICAL — Excessive forward lean. Chest up immediately",
        }]

    # ── Absolute fallback checks (no personal baseline yet) ───────────────────
    if not baseline:
        if ta > 45:
            errors.append({
                "type":     "calib_lean",
                "priority": 1,
                "severity": "warning",
                "message":  "Too much forward lean — chest up, brace your core",
            })
        # Hip hinge without knee bend: valid in both phases
        if ka > 130 and ta > 20:
            errors.append({
                "type":     "calib_hip",
                "priority": 2,
                "severity": "warning",
                "message":  "Hinging at hips without bending knees — sit back and down",
            })
        # Only flag insufficient depth when the user is already coming back up
        if phase == "ASCENDING" and ka > 150:
            errors.append({
                "type":     "calib_depth",
                "priority": 2,
                "severity": "warning",
                "message":  "Bend your knees more — sit back and down",
            })
        errors.sort(key=lambda x: x["priority"])
        return errors

    # ── Tier 2: Personal baseline deviation (signed — detects both directions) ─
    bst = baseline.get("signed_torso_angle")

    if bst is not None:
        st        = signed_torso_angle(landmarks)
        direction = 1.0 if bst >= 0 else -1.0      # which way is "forward" for this person
        signed_dev = direction * (st - bst)          # +ve = more forward, -ve = more backward

        # Forward rounding: current lean exceeds calibrated bottom lean
        if signed_dev > DEVIATION_THRESHOLD:
            errors.append({
                "type":     "forward_rounding",
                "priority": 1,
                "severity": "critical" if signed_dev > 20 else "warning",
                "message":  _forward_rounding_msg(phase, signed_dev),
            })

        # Hyperextension / anterior pelvic tilt: leaning backward past calibrated position
        elif signed_dev < -DEVIATION_THRESHOLD:
            errors.append({
                "type":     "hyperextension",
                "priority": 1,
                "severity": "warning",
                "message":  "Hyperextension detected — brace your core and tuck your pelvis",
            })

        # Hip-dominant: significant forward lean while knees remain very unbent during descent
        if phase == "DESCENDING" and signed_dev > DEVIATION_THRESHOLD and ka > 150:
            errors.append({
                "type":     "hip_dominant",
                "priority": 2,
                "severity": "warning",
                "message":  "Hinging at hips without bending knees — sit back and down",
            })

    knee_x  = (landmarks[LEFT_KNEE].x  + landmarks[RIGHT_KNEE].x)  / 2
    ankle_x = (landmarks[LEFT_ANKLE].x + landmarks[RIGHT_ANKLE].x) / 2
    if abs(knee_x - ankle_x) > 0.15:
        errors.append({
            "type":     "knee_travel",
            "priority": 3,
            "severity": "warning",
            "message":  "Knees travelling too far forward — shift weight back to heels",
        })

    errors.sort(key=lambda x: x["priority"])
    return errors


def _forward_rounding_msg(phase, deviation):
    return {
        "DESCENDING": "Back rounding on the way down — chest up, brace your core",
        "ASCENDING":  "Back collapsing on the way up — drive your chest through your knees",
    }.get(phase, "Back rounding detected — chest up and brace your core")
