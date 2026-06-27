RED_ZONES = {
    "lumbar_deviation_max": 20,
}

DEVIATION_THRESHOLD = 15


def analyze(landmarks, baseline, phase):
    """Returns a priority-sorted list of form error dicts. Empty list = good form."""
    return []  # Phase 2: lumbar hyperextension + elbow angle detection
