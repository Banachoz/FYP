import numpy as np

DESCENT_THRESHOLD    = 0.03
ASCENT_THRESHOLD     = 0.03
STANDING_KNEE_MIN    = 165  # degrees — squat/deadlift "standing" check
DEPTH_RATIO          = 0.90 # fraction of calibrated depth required before ascending is accepted, tracks hips
SQUAT_DEPTH_KNEE_MIN = 150  # degrees — knee must reach this angle for a squat rep to count
OHP_ELBOW_LOCKOUT    = 165  # degrees — elbow fully extended overhead


def run_fsm(state: dict, tracked_y: float, knee_angle: float,
            torso_angle_val: float, exercise: str,
            signed_torso_val: float = 0.0,
            hip_ankle_offset_val: float = 0.0) -> dict:
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
    OHP flow:  ASCENDING (press) → STANDING (lockout) → DESCENDING (lower) → ASCENDING
    Squat/DL:  STANDING → DESCENDING (go down) → ASCENDING (come up) → STANDING
    """
    s      = {**state, "needs_rerun": False, "completed_rep_label": ""}
    prev   = s["prev_tracked_y"]
    delta  = tracked_y - prev
    is_ohp = (exercise == "Overhead Press")

    if s["phase"] == "STANDING":
        if is_ohp:
            # OHP STANDING = top of the press (bar overhead, elbows locked out).
            # Collect torso/HAO for calibration baseline, then wait for bar to come down.
            s["feedback"]      = "Lockout — lower the bar with control"
            s["feedback_type"] = "neutral"
            if s["calib_mode"]:
                s["standing_torso_buffer"].append(signed_torso_val)
                s["standing_hao_buffer"].append(hip_ankle_offset_val)
            # Track the peak lockout position (minimum wrist Y) for position-based descent detection.
            if tracked_y < s["asc_min_y"]:
                s["asc_min_y"] = tracked_y
            # Velocity OR position: bar coming down (position check catches slow descent).
            if delta > DESCENT_THRESHOLD or tracked_y > s["asc_min_y"] + 0.03:
                if s["calib_mode"] and s["standing_torso_buffer"]:
                    s["pre_rep_standing_torso"] = float(np.mean(s["standing_torso_buffer"]))
                    s["pre_rep_standing_hao"]   = float(np.mean(s["standing_hao_buffer"]))
                s["standing_torso_buffer"] = []
                s["standing_hao_buffer"]   = []
                s["phase"]              = "DESCENDING"
                s["standing_tracked_y"] = prev
                s["asc_min_y"]          = 99.0
        else:
            s["feedback"]      = _msg_standing(exercise)
            s["feedback_type"] = "neutral"

            if knee_angle > s["standing_knee_max"]:
                s["standing_knee_max"] = knee_angle

            if s["calib_mode"]:
                s["standing_torso_buffer"].append(signed_torso_val)
                s["standing_hao_buffer"].append(hip_ankle_offset_val)

            # Knee trigger: catches knee-dominant starts where hip barely moves
            hip_trigger  = delta > DESCENT_THRESHOLD
            knee_trigger = (s["standing_knee_max"] - knee_angle) > 10

            if hip_trigger or knee_trigger:
                if s["calib_mode"] and s["standing_torso_buffer"]:
                    s["pre_rep_standing_torso"] = float(np.mean(s["standing_torso_buffer"]))
                    s["pre_rep_standing_hao"]   = float(np.mean(s["standing_hao_buffer"]))
                s["standing_torso_buffer"] = []
                s["standing_hao_buffer"]   = []
                s["phase"]              = "DESCENDING"
                s["dl_at_bottom"]       = False
                s["standing_tracked_y"] = prev
                s["standing_knee_max"]  = 0.0
                s["sq_descent_min_knee"] = 180.0
                if s["calib_mode"]:
                    s["calib_peak"]              = tracked_y
                    s["calib_peak_torso"]        = torso_angle_val
                    s["calib_peak_knee"]         = knee_angle
                    s["calib_peak_signed_torso"] = signed_torso_val

    elif s["phase"] == "DESCENDING":
        if is_ohp:
            s["feedback"]      = "Lower the bar with control"
            s["feedback_type"] = "neutral"
            # Bar has returned to shoulder level → count rep, go back to ASCENDING
            if s["ohp_bottom_ref"] > 0 and tracked_y >= s["ohp_bottom_ref"] - 0.05:
                s["ohp_bottom_ref"] = tracked_y  # refresh for next rep
                s = _complete_rep(s, exercise)
                s["phase"] = "ASCENDING"  # override STANDING set by _complete_rep
        else:
            s["feedback"]      = _msg_descending(exercise)
            s["feedback_type"] = "neutral"
            if s["calib_mode"] and tracked_y > s["calib_peak"]:
                s["calib_peak"]              = tracked_y
                s["calib_peak_torso"]        = torso_angle_val
                s["calib_peak_knee"]         = knee_angle
                s["calib_peak_signed_torso"] = signed_torso_val
            if not s["calib_mode"] and tracked_y > s["descent_max_y"]:
                s["descent_max_y"] = tracked_y
            if not is_ohp and knee_angle < s["sq_descent_min_knee"]:
                s["sq_descent_min_knee"] = knee_angle
            if exercise == "Deadlift":
                # Rep is complete when the bar returns to the floor (knee angle reaches
                # the starting reference). Only count if the user actually reached lockout
                # (dl_reached_lockout) — guards against false reps from small hip bobs.
                knee_ref = s["calib_knee_angle"] if s["calib_knee_angle"] is not None else s["bottom_knee_ref"]
                if knee_ref > 0 and knee_angle <= knee_ref + 5:
                    s["dl_at_bottom"]  = False
                    s["asc_min_y"]     = tracked_y
                    s["descent_max_y"] = 0.0
                    if s["dl_reached_lockout"]:
                        s = _complete_rep(s, exercise)  # counts the rep
                    s["phase"]              = "ASCENDING"  # override STANDING set by _complete_rep
                    s["dl_reached_lockout"] = False  # reset for the next rep
            else:
                # Position-based ascent detection: hip has risen 2% above the lowest
                # point reached during descent. Works for slow reps where per-frame
                # velocity never crosses the old ASCENT_THRESHOLD.
                bottom_ref = s["calib_peak"] if s["calib_mode"] else s["descent_max_y"]
                if bottom_ref > 0 and tracked_y < bottom_ref - 0.02:
                    if s["calib_depth"] is not None:
                        required = s["standing_tracked_y"] + (
                            (s["calib_depth"] - s["standing_tracked_y"]) * DEPTH_RATIO
                        )
                        if s["descent_max_y"] < required:
                            s["feedback"]       = "Insufficient depth — go lower before ascending"
                            s["feedback_type"]  = "warn"
                            s["prev_tracked_y"] = tracked_y
                            return s
                    s["phase"]         = "ASCENDING"
                    s["asc_min_y"]     = tracked_y
                    s["descent_max_y"] = 0.0

    elif s["phase"] == "ASCENDING":
        if is_ohp:
            s["feedback"]      = "Press to full lockout overhead"
            s["feedback_type"] = "neutral"
            # Set shoulder-level reference on very first frame
            if s["ohp_bottom_ref"] == 0.0:
                s["ohp_bottom_ref"] = tracked_y
            # Track worst back arch during the press for calibration
            if s["calib_mode"] and torso_angle_val > s["calib_peak_torso"]:
                s["calib_peak_torso"]        = torso_angle_val
                s["calib_peak_signed_torso"] = signed_torso_val
                s["calib_peak_knee"]         = knee_angle
            # Elbow lockout detected → bar is overhead
            if knee_angle >= OHP_ELBOW_LOCKOUT:
                s["phase"] = "STANDING"
        else:
            s["feedback"]      = _msg_ascending(exercise)
            s["feedback_type"] = "neutral"
            # Capture the starting knee angle for deadlift on the very first frame of the
            # first calibration rep — the user is still at the bottom position at this point,
            # before any upward movement. Used later to detect when they've returned to the
            # bottom after descending, so the phase can switch back to ASCENDING.
            if (exercise == "Deadlift" and s["calib_mode"]
                    and s["calib_reps_collected"] == 0 and s["bottom_knee_ref"] == 0.0):
                s["bottom_knee_ref"] = knee_angle
            # Capture bottom position for deadlift starting in ASCENDING (calib_peak = 0.0 initially).
            # Safe for squat: calib_peak is already set at the true bottom during DESCENDING,
            # so tracked_y (decreasing on the way up) never exceeds it here.
            if s["calib_mode"] and tracked_y > s["calib_peak"]:
                s["calib_peak"]              = tracked_y
                s["calib_peak_torso"]        = torso_angle_val
                s["calib_peak_knee"]         = knee_angle
                s["calib_peak_signed_torso"] = signed_torso_val
            if tracked_y < s["asc_min_y"]:
                s["asc_min_y"] = tracked_y
            if knee_angle >= STANDING_KNEE_MIN:
                if exercise == "Deadlift":
                    # Lockout reached — enter STANDING but don't count the rep yet.
                    # The rep is counted when the bar returns to the floor (DESCENDING→ASCENDING).
                    s["phase"]              = "STANDING"
                    s["asc_min_y"]          = 99.0
                    s["standing_knee_max"]  = 0.0
                    s["dl_reached_lockout"] = True
                else:
                    if exercise == "Squat" and s["sq_descent_min_knee"] > SQUAT_DEPTH_KNEE_MIN:
                        s["feedback"]            = "Go deeper — bend your knees further"
                        s["feedback_type"]       = "warn"
                        s["phase"]               = "STANDING"
                        s["asc_min_y"]           = 99.0
                        s["sq_descent_min_knee"] = 180.0
                        s["standing_knee_max"]   = 0.0
                    else:
                        s = _complete_rep(s, exercise)
                        s["sq_descent_min_knee"] = 180.0
            elif tracked_y > s["asc_min_y"] + 0.04:
                # Hip has dropped 4 % of frame from peak — genuine re-descent.
                # Position-based so noise (~1 %) cannot trigger it.
                s["phase"]              = "DESCENDING"
                s["dl_at_bottom"]       = False
                s["dl_reached_lockout"] = False  # never reached lockout this attempt
                s["asc_min_y"]          = 99.0

    s["last_signed_torso_val"]   = signed_torso_val
    s["last_hip_ankle_offset"]   = hip_ankle_offset_val
    s["prev_tracked_y"] = tracked_y
    return s


def _complete_rep(s: dict, exercise: str = "") -> dict:
    s["rep_count"]         += 1
    s["feedback"]           = f"Rep {s['rep_count']} complete"
    s["feedback_type"]      = "ok"
    s["phase"]              = "STANDING"
    s["asc_min_y"]          = 99.0
    s["standing_knee_max"]  = 0.0
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
        s["calib_standing_signed_torso_angles"].append(s["pre_rep_standing_torso"])
        s["calib_hip_ankle_offsets"].append(s["pre_rep_standing_hao"])
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
            s["calib_standing_signed_torso_angle"] = float(np.mean(s["calib_standing_signed_torso_angles"]))
            # Deadlift starts in ASCENDING so pre_rep_standing_hao is 0.0 for rep 1
            # (no prior STANDING phase). Use only reps 2 and 3 to avoid a biased baseline.
            hao_vals = s["calib_hip_ankle_offsets"]
            if exercise == "Deadlift" and len(hao_vals) >= 2:
                hao_vals = hao_vals[-2:]
            s["calib_hip_ankle_offset"]            = float(np.mean(hao_vals))
            s["calib_mode"]               = False
            s["feedback"]                 = "Calibration complete — system locked to your ROM"
            s["feedback_type"]            = "ok"
            s["needs_rerun"]              = True
    return s


def _msg_standing(exercise):
    return {
        "Squat":          "Begin your squat — descend when ready",
        "Deadlift":       "Hinge at the hips and reach for the bar",
        "Overhead Press": "Grip the bar then press overhead when ready",
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
