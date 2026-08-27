from core.angle_calculator import (
    torso_angle, signed_torso_angle, hip_ankle_offset,
    shoulder_ankle_offset, wrist_shoulder_offset, facing_direction,
    LEFT_WRIST, RIGHT_WRIST,
)

RED_ZONES = {
    "torso_angle_max": 60,
}

DEVIATION_THRESHOLD          = 8   # degrees — forward_rounding check (calibration hips rising first & analysis back rounding)
HIPS_FIRST_THRESHOLD         = 18   # degrees — hips_first needs more headroom at the bottom
STANDING_HYPEREXT_THRESHOLD  =  8
HIP_FORWARD_THRESHOLD        =  0.08
HIP_ABSOLUTE_THRESHOLD       =  0.14
SHOULDER_ANKLE_THRESHOLD     =  0.05  # torso-normalised
# WRIST_FORWARD_THRESHOLD      =  0.07  # torso-normalised — wrists must not be forward of shoulders (full side view)
WRIST_FORWARD_THRESHOLD      =  0.10  # torso-normalised — wrists must not be forward of shoulders (slightly facing cam)
WRIST_VIS_MIN                =  0.40  # MediaPipe visibility cutoff — side-view occludes wrists more often
SETUP_TORSO_MIN              = 25     # degrees — gate for hip-shoulder height check at bottom


def analyze(landmarks, baseline, phase, bottom_ref_torso=None):
    """
    Conventional deadlift form analysis. FSM starts in ASCENDING (user grips bar at bottom).
    Ankle used as bar-position proxy.

    Tier 1  — Spine red zone (universal, early return)
    Fallback — Absolute checks when no baseline exists
    Tier 2  — Personal baseline deviation (post-calibration)
    """
    errors = []
    ta = torso_angle(landmarks)
    # debug — investigating torso_angle underreporting vs physical angle
    print(f"[DL torso_angle] ta={ta:.1f}  threshold={RED_ZONES['torso_angle_max']}")

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
    bst = baseline.get("signed_torso_angle") if baseline else None
    # Use the calibrated bottom torso angle for direction when available; fall back to the
    # live torso angle otherwise. At the bottom of a deadlift the torso is heavily inclined,
    # so the live angle has a clear sign even without a calibrated baseline.
    direction_ref = bst if bst is not None else signed_torso_angle(landmarks)
    direction     = 1.0 if direction_ref >= 0 else -1.0

    if phase == "ASCENDING":
        # Starting-position checks — only while the torso is still heavily inclined
        if ta > SETUP_TORSO_MIN:
            # Shoulders must not drift behind the bar (ankles used as bar-position proxy)
            sao = shoulder_ankle_offset(landmarks)
            if direction * sao < -SHOULDER_ANKLE_THRESHOLD:
                errors.append({
                    "type":     "shoulders_behind",
                    "priority": 2,
                    "severity": "warning",
                    "message":  "Shoulders behind the bar — keep shoulders over or in front of the bar",
                })


        # Bar drifting away from the body — wrists moving forward of the shoulders.
        # Gated on MediaPipe wrist visibility: side-view occlusion makes wrist tracking unreliable.
        lw_vis = landmarks[LEFT_WRIST].visibility
        rw_vis = landmarks[RIGHT_WRIST].visibility
        # debug — investigating wrist visibility gate blocking bar_too_far in side view
        print(f"[wrist_vis] lw={lw_vis:.2f}  rw={rw_vis:.2f}  min={min(lw_vis,rw_vis):.2f}  threshold={WRIST_VIS_MIN}")
        if min(lw_vis, rw_vis) > WRIST_VIS_MIN:
            wso = wrist_shoulder_offset(landmarks)
            print(f"[wrist_offset] wso={wso:.3f}  dir*wso={direction*wso:.3f}  threshold={WRIST_FORWARD_THRESHOLD}")
            if direction * wso > WRIST_FORWARD_THRESHOLD:
                errors.append({
                    "type":     "bar_too_far",
                    "priority": 2,
                    "severity": "warning",
                    "message":  "Bar too far from body — keep the bar close to your legs throughout the lift",
                })

        # Hips rising faster than the torso — personal baseline deviation
        if baseline and bst is not None:
            st         = signed_torso_angle(landmarks)
            signed_dev = direction * (st - bst)
            if signed_dev > HIPS_FIRST_THRESHOLD:
                errors.append({
                    "type":     "hips_first",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Hips rising too fast — drive chest and hips up together",
                })
        elif not baseline and bottom_ref_torso:
            st      = signed_torso_angle(landmarks)
            dir_b   = 1.0 if bottom_ref_torso >= 0 else -1.0
            dev     = dir_b * (st - bottom_ref_torso)
            if dev > DEVIATION_THRESHOLD:
                errors.append({
                    "type":     "calib_hips_first",
                    "priority": 1,
                    "severity": "warning",
                    "message":  "Hips rising too fast — drive chest and hips up together",
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
        elif not baseline and ta > 35:
            errors.append({
                "type":     "calib_lean",
                "priority": 1,
                "severity": "warning",
                "message":  "Back rounding on descent — keep a neutral spine",
            })

    errors.sort(key=lambda x: x["priority"])
    return errors
