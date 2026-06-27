from core.angle_calculator import (
    torso_angle, avg_knee_angle,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)

# Torso angle is measured from VERTICAL (0° = fully upright, 90° = parallel to floor).
# Knee angle is the interior angle at the knee (180° = fully extended, ~80° = deep squat).
RED_ZONES = {
    "torso_angle_max": 65,   # beyond 65° from vertical = torso nearly horizontal = danger
    "knee_angle_min":  65,   # below 65° = excessive knee stress (too deep)
}

DEVIATION_THRESHOLD = 15  # degrees from personal baseline


def analyze(landmarks, baseline, phase):
    """
    Returns a priority-sorted list of form error dicts. Caller uses errors[0] only.
    Two-tier system: Red Zone (universal) → Personal baseline deviation.
    """
    if phase == "STANDING":
        return []

    errors = []
    ta = torso_angle(landmarks)
    ka = avg_knee_angle(landmarks)

    # ── Tier 1: Universal Red Zone (applies to everyone) ──────────────────────
    if ta > RED_ZONES["torso_angle_max"]:
        errors.append({
            "type":     "red_zone_spine",
            "priority": 1,
            "severity": "critical",
            "message":  "CRITICAL — Excessive forward lean. Chest up immediately",
        })
        return errors  # most critical possible — surface immediately, skip other checks

    if ka < RED_ZONES["knee_angle_min"]:
        errors.append({
            "type":     "red_zone_knee",
            "priority": 1,
            "severity": "critical",
            "message":  "CRITICAL — Squat too deep, excessive knee stress. Come up slightly",
        })
        return errors

    # ── Tier 2: Personal baseline deviation (body-proportion-aware) ───────────
    if not baseline:
        return []

    bt = baseline.get("torso_angle")
    bk = baseline.get("knee_angle")

    if bt is not None:
        torso_dev = ta - bt

        # Forward rounding / excessive forward lean
        if torso_dev > DEVIATION_THRESHOLD:
            errors.append({
                "type":     "forward_rounding",
                "priority": 1,
                "severity": "critical" if torso_dev > 25 else "warning",
                "message":  _forward_rounding_msg(phase, torso_dev),
            })

        # Backward hyperextension (leaning back past baseline)
        elif bt - ta > DEVIATION_THRESHOLD:
            errors.append({
                "type":     "hyperextension",
                "priority": 1,
                "severity": "warning",
                "message":  "Leaning too far back — bring your chest forward",
            })

        # Hip-dominant lean: torso tilting forward but knees barely bending
        if torso_dev > 10 and ka > 130:
            errors.append({
                "type":     "hip_dominant",
                "priority": 2,
                "severity": "warning",
                "message":  "Hinging at hips without bending knees — sit back and down",
            })

    # Insufficient depth (knee angle shallower than calibrated depth)
    if bk is not None and ka - bk > DEVIATION_THRESHOLD:
        errors.append({
            "type":     "insufficient_depth",
            "priority": 3,
            "severity": "warning",
            "message":  "Not deep enough — squat lower to match your calibrated depth",
        })

    # Forward knee travel (knee significantly ahead of ankle in side view)
    knee_x  = (landmarks[LEFT_KNEE].x  + landmarks[RIGHT_KNEE].x)  / 2
    ankle_x = (landmarks[LEFT_ANKLE].x + landmarks[RIGHT_ANKLE].x) / 2
    if abs(knee_x - ankle_x) > 0.12:
        errors.append({
            "type":     "knee_travel",
            "priority": 4,
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
