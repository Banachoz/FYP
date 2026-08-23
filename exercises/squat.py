from core.angle_calculator import (
    torso_angle, signed_torso_angle, avg_knee_angle, hip_ankle_offset,
    facing_direction,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)

_dbg = 0  # frame counter for debug throttle

# Torso angle measured from VERTICAL (0° = upright, 90° = parallel to floor).
# Knee angle is the interior angle at the knee (180° = fully extended, ~70° = deep squat).
RED_ZONES = {
    "torso_angle_max": 65,   # torso nearly horizontal — spinal danger
}

DEVIATION_THRESHOLD          = 10    # degrees from personal baseline (descent checks)
STANDING_HYPEREXT_THRESHOLD  =  6    # degrees — torso lean backward from calibrated standing
HIP_FORWARD_THRESHOLD        =  0.08 # torso-normalised+aspect-corrected — ~5 cm push at 2 m (relative)
HIP_ABSOLUTE_THRESHOLD       =  0.11 # torso-normalised+aspect-corrected — ~5 cm total lean (absolute)


def analyze(landmarks, baseline, phase, rep_count=0, bottom_ref_torso=None):  # noqa: C901
    """
    Returns a priority-sorted list of form error dicts.
    Caller displays errors[0] in the feedback box; all errors are logged per rep.

    Tier 1  — Spine red zone (universal, early return)
    Fallback — Absolute reference checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    if phase == "STANDING":
        global _dbg
        _dbg = (_dbg + 1) % 30

        live_offset = hip_ankle_offset(landmarks)  # torso-normalised + aspect-corrected
        # Absolute check — calibration only (no baseline). Once a baseline exists
        # the more accurate baseline-corrected hip_forward check below takes over.
        if not baseline:
            if _dbg == 0:
                print(f"[hip_abs] live={live_offset:.3f}  threshold={HIP_ABSOLUTE_THRESHOLD}")
            if bottom_ref_torso:
                dir_live = 1.0 if bottom_ref_torso >= 0 else -1.0
                hip_fwd_condition = dir_live * live_offset > HIP_ABSOLUTE_THRESHOLD
            else:
                _fd = facing_direction(landmarks)
                if _fd is not None:
                    hip_fwd_condition = _fd * live_offset > HIP_ABSOLUTE_THRESHOLD
                else:
                    hip_fwd_condition = abs(live_offset) > HIP_ABSOLUTE_THRESHOLD
            if hip_fwd_condition:
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
                dev_hip = direction * (live_offset - calib_offset)
                if _dbg == 0:
                    print(f"[hip_fwd] live={live_offset:.3f}  calib={calib_offset:.3f}  dev={dev_hip:.3f}  threshold={HIP_FORWARD_THRESHOLD}  dir={direction:.0f}")
                if dev_hip > HIP_FORWARD_THRESHOLD:
                    return [{
                        "type":     "hip_forward",
                        "priority": 1,
                        "severity": "warning",
                        "message":  "Hips pushing forward at lockout — avoid overextending your lower back",
                    }]

            # Check 2 — shoulder lean backward
            bsst = baseline.get("signed_torso_standing_angle")
            if bsst is not None:
                st      = signed_torso_angle(landmarks)
                dev_ht  = direction * (st - bsst)
                if _dbg == 0:
                    print(f"[hyperext] st={st:.1f}  bsst={bsst:.1f}  dev={dev_ht:.1f}  threshold={-STANDING_HYPEREXT_THRESHOLD}  dir={direction:.0f}")
                if dev_ht < -STANDING_HYPEREXT_THRESHOLD:
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
    # debug — investigating torso_angle underreporting vs physical angle
    print(f"[SQ torso_angle] ta={ta:.1f}  threshold={RED_ZONES['torso_angle_max']}")

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
        if phase == "DESCENDING" and ta > 45:
            errors.append({
                "type":     "calib_lean",
                "priority": 1,
                "severity": "warning",
                "message":  "Back rounding on the way down — chest up, brace your core",
            })
        # Hip hinge without knee bend: valid in both phases
        if phase == "DESCENDING" and ka > 140 and ta > 30:
            errors.append({
                "type":     "calib_hip",
                "priority": 2,
                "severity": "warning",
                "message":  "Hinging at hips without bending knees — sit back and down",
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