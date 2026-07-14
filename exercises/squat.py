from core.angle_calculator import (
    torso_angle, signed_torso_angle, avg_knee_angle,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)

# Torso angle measured from VERTICAL (0° = upright, 90° = parallel to floor).
# Knee angle is the interior angle at the knee (180° = fully extended, ~70° = deep squat).
RED_ZONES = {
    "torso_angle_max": 65,   # torso nearly horizontal — spinal danger
}

DEVIATION_THRESHOLD          = 10    # degrees from personal baseline (descent checks)
STANDING_HYPEREXT_THRESHOLD  =  5    # degrees — tighter, standing comparison is position-matched
HIP_FORWARD_THRESHOLD        =  0.05 # normalised units — hips pushed forward of ankles (relative)
HIP_ABSOLUTE_THRESHOLD       =  0.07 # normalised units — absolute hip-ankle offset (no baseline needed)


def analyze(landmarks, baseline, phase):
    """
    Returns a priority-sorted list of form error dicts.
    Caller displays errors[0] in the feedback box; all errors are logged per rep.

    Tier 1  — Spine red zone (universal, early return)
    Fallback — Absolute reference checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    if phase == "STANDING":
        # Absolute check — fires during calibration AND workout (no baseline required).
        # abs() makes it direction-agnostic (left-facing or right-facing camera angle).
        hip_x   = (landmarks[LEFT_HIP].x   + landmarks[RIGHT_HIP].x)   / 2
        ankle_x = (landmarks[LEFT_ANKLE].x + landmarks[RIGHT_ANKLE].x) / 2
        if abs(hip_x - ankle_x) > HIP_ABSOLUTE_THRESHOLD:
            return [{
                "type":     "hip_forward_absolute",
                "priority": 1,
                "severity": "warning",
                "message":  "Hips pushing forward — tuck your pelvis and stand tall",
            }]

        if baseline:
            # Use bottom signed torso for direction (clearly signed, never near zero)
            bst_dir = baseline.get("signed_torso_angle")
            direction = 1.0 if (bst_dir or 0) >= 0 else -1.0

            # Check 1 — hip-forward displacement (strongest hyperextension signal)
            calib_offset = baseline.get("hip_ankle_offset")
            if calib_offset is not None:
                hip_x   = (landmarks[LEFT_HIP].x   + landmarks[RIGHT_HIP].x)   / 2
                ankle_x = (landmarks[LEFT_ANKLE].x + landmarks[RIGHT_ANKLE].x) / 2
                live_offset = hip_x - ankle_x
                if direction * (live_offset - calib_offset) > HIP_FORWARD_THRESHOLD:
                    return [{
                        "type":     "hip_forward",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Hips pushing forward at lockout — avoid overextending your lower back",
                    }]

            # Check 2 — shoulder lean backward (signed torso standing comparison)
            bsst = baseline.get("signed_torso_standing_angle")
            if bsst is not None:
                st  = signed_torso_angle(landmarks)
                dev = direction * (st - bsst)
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
                "message":  "Back rounding on the way down — chest up, brace your core",
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
        # Hips-first during calibration: torso still inclined on the way up
        if phase == "ASCENDING" and ta > 40:
            errors.append({
                "type":     "calib_hips_first",
                "priority": 1,
                "severity": "warning",
                "message":  "Hips shooting up first — drive your chest and hips up together",
            })
        errors.sort(key=lambda x: x["priority"])
        return errors

    # ── Tier 2: Personal baseline deviation (signed — detects both directions) ─
    bst = baseline.get("signed_torso_angle")

    # DESCENDING: torso tips forward beyond calibrated bottom lean → back rounding
    if phase == "DESCENDING" and bst is not None:
        st         = signed_torso_angle(landmarks)
        direction  = 1.0 if bst >= 0 else -1.0
        signed_dev = direction * (st - bst)
        if signed_dev > DEVIATION_THRESHOLD:
            errors.append({
                "type":     "forward_rounding",
                "priority": 1,
                "severity": "critical" if signed_dev > 20 else "warning",
                "message":  "Back rounding on the way down — chest up, brace your core",
            })

    # ASCENDING: torso should become more upright; if still inclined, hips are rising first
    if phase == "ASCENDING" and bst is not None:
        st         = signed_torso_angle(landmarks)
        direction  = 1.0 if bst >= 0 else -1.0
        signed_dev = direction * (st - bst)
        if signed_dev > DEVIATION_THRESHOLD:
            errors.append({
                "type":     "hips_first",
                "priority": 1,
                "severity": "warning",
                "message":  "Hips shooting up first — drive your chest and hips up together",
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
