RED_ZONES = {
    "torso_angle_min": 20,
    "torso_angle_max": 80,
}

DEVIATION_THRESHOLD = 15


def analyze(landmarks, baseline, phase):
    """Returns a priority-sorted list of form error dicts. Empty list = good form."""
    return []  # Phase 2: back flatness + hip hinge ratio detection
