import numpy as np

DESCENT_THRESHOLD = 0.03
ASCENT_THRESHOLD  = 0.03
STANDING_KNEE_MIN = 150  # degrees — squat/deadlift "standing" check


def run_fsm(state: dict, tracked_y: float, knee_angle: float,
            torso_angle_val: float, exercise: str,
            signed_torso_val: float = 0.0) -> dict:
    """
    Pure FSM update — no Streamlit imports, no side effects.

    Expected state keys:
        phase, rep_count, calib_mode, calib_reps_collected, calib_depths,
        calib_depth, calib_peak, calib_peak_torso, calib_peak_knee,
        calib_peak_signed_torso, calib_torso_angles, calib_knee_angles,
        calib_signed_torso_angles, calib_standing_signed_torso_angles,
        calib_torso_angle, calib_knee_angle, calib_signed_torso_angle,
        calib_standing_signed_torso_angle, last_signed_torso_val,
        feedback, feedback_type, depth_ratio, prev_tracked_y, standing_tracked_y

    Returns a copy of state with updated values plus 'needs_rerun' bool.
    OHP flow:  STANDING → ASCENDING (press up) → DESCENDING (lower) → STANDING
    Squat/DL:  STANDING → DESCENDING (go down)  → ASCENDING (come up) → STANDING
    """
    s      = {**state, "needs_rerun": False, "completed_rep_label": ""}
    prev   = s["prev_tracked_y"]
    delta  = tracked_y - prev
    is_ohp = (exercise == "Overhead Press")

    if s["phase"] == "STANDING":
        s["feedback"]      = _msg_standing(exercise)
        s["feedback_type"] = "neutral"

        trigger = (delta < -DESCENT_THRESHOLD) if is_ohp else (delta > DESCENT_THRESHOLD)

        if trigger:
            s["phase"]              = "ASCENDING" if is_ohp else "DESCENDING"
            s["standing_tracked_y"] = prev
            if s["calib_mode"]:
                s["calib_peak"]              = tracked_y
                s["calib_peak_torso"]        = torso_angle_val
                s["calib_peak_knee"]         = knee_angle
                s["calib_peak_signed_torso"] = signed_torso_val

    elif s["phase"] == "DESCENDING":
        if is_ohp:
            s["feedback"]      = "Lower the bar with control"
            s["feedback_type"] = "neutral"
            if tracked_y >= s["standing_tracked_y"] - 0.05:
                s = _complete_rep(s)
        else:
            s["feedback"]      = _msg_descending(exercise)
            s["feedback_type"] = "neutral"
            if s["calib_mode"] and tracked_y > s["calib_peak"]:
                s["calib_peak"]              = tracked_y
                s["calib_peak_torso"]        = torso_angle_val
                s["calib_peak_knee"]         = knee_angle
                s["calib_peak_signed_torso"] = signed_torso_val
            if delta < -ASCENT_THRESHOLD:
                if s["calib_depth"] is not None:
                    required = s["standing_tracked_y"] + (
                        (s["calib_depth"] - s["standing_tracked_y"]) * s["depth_ratio"]
                    )
                    if tracked_y < required:
                        s["feedback"]       = "Insufficient depth — go lower before ascending"
                        s["feedback_type"]  = "warn"
                        s["prev_tracked_y"] = tracked_y
                        return s
                s["phase"] = "ASCENDING"

    elif s["phase"] == "ASCENDING":
        if is_ohp:
            s["feedback"]      = "Press to full lockout overhead"
            s["feedback_type"] = "neutral"
            if s["calib_mode"] and tracked_y < s["calib_peak"]:
                s["calib_peak"]       = tracked_y
                s["calib_peak_torso"] = torso_angle_val
            if delta > ASCENT_THRESHOLD:
                s["phase"] = "DESCENDING"
        else:
            s["feedback"]      = _msg_ascending(exercise)
            s["feedback_type"] = "neutral"
            if knee_angle >= STANDING_KNEE_MIN:
                s = _complete_rep(s)

    s["last_signed_torso_val"] = signed_torso_val
    s["prev_tracked_y"] = tracked_y
    return s


def _complete_rep(s: dict) -> dict:
    s["rep_count"]    += 1
    s["feedback"]      = f"Rep {s['rep_count']} complete"
    s["feedback_type"] = "ok"
    s["phase"]         = "STANDING"
    # Compute label now — calib_reps_collected not yet incremented (+1), calib_mode not yet flipped
    if s["calib_mode"]:
        s["completed_rep_label"] = f"Calib {s['calib_reps_collected'] + 1}"
    else:
        s["completed_rep_label"] = f"Rep {s['rep_count']}"
    if s["calib_mode"]:
        s["calib_depths"].append(s["calib_peak"])
        s["calib_torso_angles"].append(s["calib_peak_torso"])
        s["calib_knee_angles"].append(s["calib_peak_knee"])
        s["calib_signed_torso_angles"].append(s["calib_peak_signed_torso"])
        s["calib_standing_signed_torso_angles"].append(s["last_signed_torso_val"])
        s["calib_reps_collected"] += 1
        s["calib_peak"]              = 0.0
        s["calib_peak_torso"]        = 0.0
        s["calib_peak_knee"]         = 180.0
        s["calib_peak_signed_torso"] = 0.0
        if s["calib_reps_collected"] >= 3:
            s["calib_depth"]                      = float(np.mean(s["calib_depths"]))
            s["calib_torso_angle"]                = float(np.mean(s["calib_torso_angles"]))
            s["calib_knee_angle"]                 = float(np.mean(s["calib_knee_angles"]))
            s["calib_signed_torso_angle"]         = float(np.mean(s["calib_signed_torso_angles"]))
            s["calib_standing_signed_torso_angle"]= float(np.mean(s["calib_standing_signed_torso_angles"]))
            s["calib_mode"]               = False
            s["feedback"]                 = "Calibration complete — system locked to your ROM"
            s["feedback_type"]            = "ok"
            s["needs_rerun"]              = True
    return s


def _msg_standing(exercise):
    return {
        "Squat":          "Begin your squat — descend when ready",
        "Deadlift":       "Hinge at the hips and reach for the bar",
        "Overhead Press": "Grip the bar — press overhead when ready",
    }.get(exercise, "Begin when ready")


def _msg_descending(exercise):
    return {
        "Squat":    "Keep descending — control the movement",
        "Deadlift": "Keep the bar close — maintain a flat back",
    }.get(exercise, "Keep descending")


def _msg_ascending(exercise):
    return {
        "Squat":    "Drive through your heels — extend fully at the top",
        "Deadlift": "Drive your hips forward — stand tall",
    }.get(exercise, "Drive up")
