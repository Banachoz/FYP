# FYP Development Guide
## AI Exercise Form Analysis System
### Real-Time Biomechanical Feedback for Injury Prevention

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Technical Stack](#2-technical-stack)
3. [System Architecture](#3-system-architecture)
4. [What the System Measures](#4-what-the-system-measures)
5. [The Calibration System](#5-the-calibration-system)
6. [Phase Detection Logic](#6-phase-detection-logic)
7. [Form Analysis & Red Zones](#7-form-analysis--red-zones)
8. [Feedback System](#8-feedback-system)
9. [Feature Roadmap](#9-feature-roadmap)
10. [File & Folder Structure](#10-file--folder-structure)
11. [Implementation Guide (Phase by Phase)](#11-implementation-guide-phase-by-phase)
12. [Evaluation Plan](#12-evaluation-plan)
13. [Known Limitations](#13-known-limitations)
14. [Viva Defense Notes](#14-viva-defense-notes)

---

## 1. Project Identity

| Field | Details |
|-------|---------|
| **Title** | Developing an AI System for Real-Time Exercise Form Analysis to Prevent Injuries Among Gym Users |
| **SDG** | SDG 3 — Good Health and Well-Being |
| **Type** | Applied AI / Computer Vision System |
| **Platform** | Web-based (Streamlit), runs locally on standard laptop |
| **Supervisor** | TS. Dr. Law Foong Li |

### Aim

To develop a real-time computer vision system that analyzes exercise form during compound barbell movements and provides personalized corrective feedback to reduce the risk of workout-related injuries among gym users.

### Objectives

1. To implement pose estimation using computer vision to detect body keypoints and calculate joint angles for assessing exercise form correctness
2. To design a calibration system that personalizes form analysis based on individual body mechanics while maintaining biomechanically safe thresholds
3. To develop phase detection and feedback systems that provide context-specific corrections prioritized by injury prevention importance
4. To evaluate system performance through angle measurement validation and user studies with gym participants

### Target Users

**Primary:**
- Beginner to intermediate gym-goers learning compound movements
- Home gym users without access to trainers or training partners
- Budget-conscious lifters who cannot afford regular personal training

**Secondary:**
- Fitness enthusiasts seeking objective form feedback
- Rehabilitation patients cleared for strength training
- Fitness coaches using the system as a supplementary tool

---

## 2. Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI Engine** | MediaPipe Pose | 33 keypoint skeletal tracking |
| **Math/Vision** | OpenCV + NumPy | Angle calculations, frame processing |
| **Backend Logic** | Python | Rule-based heuristic analysis |
| **Web Interface** | Streamlit | Rapid UI development, webcam access |
| **Audio Feedback** | pyttsx3 (TTS) | Hands-free voice coaching |
| **Data Storage** | Local JSON + In-Memory | Session logs, no database needed |

### Why This Stack

- **Streamlit** removes the need for a full MERN web stack (massive scope reduction)
- **MediaPipe** is pre-trained — no GPU, no training data needed
- **Local processing** = zero latency, zero cloud dependency, full privacy
- **pyttsx3** works offline, no API key required
- **JSON logging** is simple, lightweight, sufficient for FYP scope

### Hardware Requirements

- Standard laptop with built-in or USB webcam (720p minimum, 30 FPS)
- No GPU required (MediaPipe CPU version handles this)
- Any modern laptop (4GB RAM minimum)
- Camera positioned to user's **side view** (5–6 feet away, full body visible)

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                      │
│   Exercise Selector | Live Feed | Feedback Panel         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   PYTHON BACKEND                         │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐  │
│  │  MediaPipe  │───▶│    Angle     │──▶│  Form       │  │
│  │  Pose Est.  │    │  Calculator  │   │  Analyzer   │  │
│  │ (33 points) │    │ (Trig Math)  │   │ (Your Rules)│  │
│  └─────────────┘    └──────────────┘   └──────┬──────┘  │
│                                               │          │
│  ┌─────────────┐    ┌──────────────┐   ┌──────▼──────┐  │
│  │  Calibration│    │   Phase      │   │  Priority   │  │
│  │  System     │    │  Detector    │   │  Logic      │  │
│  │ (Baselines) │    │ (Up/Down)    │   │ (Triage)    │  │
│  └─────────────┘    └──────────────┘   └──────┬──────┘  │
│                                               │          │
│  ┌─────────────┐    ┌──────────────┐   ┌──────▼──────┐  │
│  │  Session    │    │  Snapshot    │   │  Feedback   │  │
│  │  Logger     │    │  Capture     │   │  Engine     │  │
│  │ (JSON)      │    │ (Peak Error) │   │ (TTS+Text)  │  │
│  └─────────────┘    └──────────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               LOCAL STORAGE (JSON)                       │
│          session_log.json | snapshots/                   │
└─────────────────────────────────────────────────────────┘
```

### System Workflow (Happy Path)

1. **User opens Streamlit app** → Selects exercise from dropdown (Squat / Deadlift / Overhead Press)
2. **Calibration phase** → User performs 3–5 reps → System records personal angle baselines
3. **Safety gate** → System validates calibration against Universal Red Zones → Accepts or rejects
4. **Live analysis begins** → Webcam feed processed at 30 FPS
5. **Phase detection** → System identifies Descending / Bottom / Ascending
6. **Form analysis** → Joint angles compared against user baseline + Red Zones
7. **Priority triage** → If multiple errors, spine/back error surfaced first
8. **Feedback delivered** → TTS voice cue + on-screen text overlay
9. **Snapshot captured** → Peak error frame saved (max 5 per set)
10. **Post-set review** → Dashboard shows session summary, errors ranked by injury risk

---

## 4. What the System Measures

### Key MediaPipe Landmarks Used

```
Landmark ID  |  Body Part
─────────────────────────
    11        |  Left Shoulder
    12        |  Right Shoulder
    23        |  Left Hip
    24        |  Right Hip
    25        |  Left Knee
    26        |  Right Knee
    27        |  Left Ankle
    28        |  Right Ankle
    15        |  Left Wrist   (OHP only)
    16        |  Right Wrist  (OHP only)
    13        |  Left Elbow   (OHP only)
    14        |  Right Elbow  (OHP only)
```

### Measurement Table

| # | Measurement | Landmarks Used | Exercise | What It Catches |
|---|------------|---------------|----------|-----------------|
| 1 | **Torso Angle** | Shoulder → Hip (vs. vertical) | ALL | Back rounding, forward collapse, hyperextension |
| 2 | **Phase Detection** | Hip y-coord (squat/DL), Wrist y-coord (OHP) | ALL | Movement timing for context-aware feedback |
| 3 | **Knee Angle** | Hip → Knee → Ankle | Squat | Squat depth |
| 4 | **Hip Depth** | Hip y vs. Knee y | Squat | Parallel depth check |
| 5 | **Hip Hinge Ratio** | Hip rise rate vs. Shoulder rise rate | Deadlift | Hips shooting up before shoulders |
| 6 | **Back Flatness** | Shoulder–Hip angle consistency | Deadlift | Back rounding at bottom of pull |
| 7 | **Lumbar Hyperextension** | Shoulder → Hip vs. vertical | OHP | Excessive backward arch |
| 8 | **Elbow Angle** | Shoulder → Elbow → Wrist | OHP | Elbow flare, bar path |

### The Core Math (Angle Calculation)

Every measurement is either an **angle** or a **y-coordinate comparison**.

```python
import numpy as np

def calculate_angle(A, B, C):
    """
    Calculate angle at point B formed by points A-B-C.
    A, B, C are (x, y) coordinate tuples.
    Returns angle in degrees.
    """
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent floating point errors
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

# Example: Knee angle during squat
hip   = (landmarks[23].x, landmarks[23].y)
knee  = (landmarks[25].x, landmarks[25].y)
ankle = (landmarks[27].x, landmarks[27].y)

knee_angle = calculate_angle(hip, knee, ankle)
```

```python
def calculate_torso_angle(shoulder, hip):
    """
    Calculate torso angle relative to vertical axis.
    Returns angle in degrees from vertical.
    """
    # Vertical reference point (directly above hip)
    vertical_ref = (hip[0], hip[1] - 1)

    return calculate_angle(shoulder, hip, vertical_ref)
```

---

## 5. The Calibration System

### Why Calibration Exists

Different people have different natural mechanics:
- Long femurs → more forward lean required in squat
- Short torso → different torso angle at squat depth
- Hip anatomy → determines natural range of motion

**Fixed universal thresholds fail most people.**
**Solution: Two-tier system (Universal Safety + Personal Baseline)**

### Tier 1: Universal Red Zones (Non-Negotiable)

Research-based safety limits that apply to everyone regardless of body type. Derived from sports medicine and biomechanics literature.

```python
UNIVERSAL_RED_ZONES = {
    "squat": {
        "torso_angle_min": 30,    # More forward than this = dangerous rounding
        "torso_angle_max": 85,    # More backward than this = falling over
        "knee_angle_min":  65,    # Deeper = excessive knee stress
        "knee_angle_max":  110,   # Shallower = insufficient depth
    },
    "deadlift": {
        "torso_angle_min": 20,    # More rounded = spine at critical risk
        "torso_angle_max": 80,    # Too upright = not hinging properly
    },
    "overhead_press": {
        "lumbar_deviation_max": 20,  # More than 20° arch = hyperextension risk
    }
}
```

### Tier 2: Personal Baseline (User-Specific)

```python
def run_calibration(exercise, n_reps=5):
    """
    Records user's natural angles over calibration reps.
    Returns accepted baseline or prompts tutorial if unsafe.
    """
    calibration_angles = []

    for rep in range(n_reps):
        # Record the peak angle at the bottom of each rep
        peak_angle = detect_peak_angle(exercise)
        calibration_angles.append(peak_angle)

    # Calculate personal average
    user_baseline = np.mean(calibration_angles)

    # SAFETY GATE: Check against Red Zones
    red_zone = UNIVERSAL_RED_ZONES[exercise]

    if red_zone["torso_angle_min"] <= user_baseline <= red_zone["torso_angle_max"]:
        print(f"Baseline accepted: {user_baseline:.1f}°")
        return user_baseline
    else:
        trigger_tutorial(exercise)
        return None  # Force recalibration
```

### Calibration Flow

```
User performs 3–5 calibration reps
            │
            ▼
System calculates average angles
            │
            ▼
     SAFETY GATE CHECK
    ┌───────┴───────┐
    │               │
  PASS            FAIL
    │               │
    ▼               ▼
Accept baseline  Show tutorial
(proceed to      Ask to recalibrate
  workout)       (if still fails →
                  suggest trainer)
```

### What Gets Stored as Baseline

```python
user_session = {
    "exercise":          "squat",
    "torso_baseline":    43.2,    # Their natural torso angle
    "knee_baseline":     88.5,    # Their natural knee angle at depth
    "hip_depth_ratio":   1.05,    # Hip y vs Knee y at bottom
    "calibration_reps":  5,
    "session_date":      "2025-06-09"
}
```

---

## 6. Phase Detection Logic

### Why Phases Matter

Different phases of a lift require different feedback:
- Telling someone "drive up" while they're still descending is wrong
- Spine alert at the bottom is more critical than at the top
- Context-aware feedback is more useful and actionable

### Phase States

```
STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING
   │                                              │
   └──────────────── REP COMPLETE ───────────────┘
```

### Detection Logic

```python
def detect_phase(landmarks, exercise, prev_hip_y):
    """
    Determines current movement phase based on joint y-coordinate movement.
    Note: In MediaPipe, y increases downward on screen.
    """
    if exercise in ["squat", "deadlift"]:
        current_y = landmarks[23].y  # Left hip y-coordinate
    elif exercise == "overhead_press":
        current_y = landmarks[15].y  # Left wrist y-coordinate

    y_change = current_y - prev_hip_y

    THRESHOLD = 0.005  # Minimum movement to register phase change

    if y_change > THRESHOLD:
        return "descending"    # y increasing = moving down
    elif y_change < -THRESHOLD:
        return "ascending"     # y decreasing = moving up
    else:
        return "bottom"        # Stationary = at peak/bottom
```

### Phase-Specific Feedback Rules

```python
PHASE_FEEDBACK = {
    "squat": {
        "descending": {
            "back_rounding": "Keep your chest up as you go down",
        },
        "bottom": {
            "back_rounding": "STOP — Back is rounding. Chest up, brace core",
            "too_shallow":   "Go deeper — hips not below parallel"
        },
        "ascending": {
            "back_rounding": "Back collapsing — drive through your heels",
        }
    },
    "deadlift": {
        "bottom": {
            "back_rounding": "CRITICAL — Back severely rounded. Do not lift",
        },
        "ascending": {
            "hips_shooting": "Hips rising too fast — drive chest up simultaneously"
        }
    },
    "overhead_press": {
        "ascending": {
            "hyperextension": "Back arching — brace your core and tuck your hips"
        },
        "descending": {
            "elbow_flare": "Keep elbows slightly forward — protect your shoulders"
        }
    }
}
```

---

## 7. Form Analysis & Red Zones

### The Analysis Pipeline

```python
def analyze_form(landmarks, exercise, user_baseline, current_phase):
    """
    Main form analysis function.
    Returns list of errors sorted by priority (highest risk first).
    """
    errors = []

    torso_angle = calculate_torso_angle(
        (landmarks[11].x, landmarks[11].y),  # Shoulder
        (landmarks[23].x, landmarks[23].y)   # Hip
    )

    if exercise == "squat":
        knee_angle = calculate_angle(
            (landmarks[23].x, landmarks[23].y),  # Hip
            (landmarks[25].x, landmarks[25].y),  # Knee
            (landmarks[27].x, landmarks[27].y)   # Ankle
        )

        torso_deviation = torso_angle - user_baseline["torso_baseline"]
        if torso_deviation > 15:
            errors.append({
                "type":     "back_rounding",
                "priority": 1,
                "severity": "critical" if torso_deviation > 25 else "warning",
                "message":  PHASE_FEEDBACK["squat"][current_phase]["back_rounding"]
            })

        knee_deviation = knee_angle - user_baseline["knee_baseline"]
        if knee_deviation > 15:
            errors.append({
                "type":     "insufficient_depth",
                "priority": 3,
                "severity": "warning",
                "message":  "Go deeper — you are shallower than your calibration depth"
            })

    # Sort by priority (1 = most dangerous first)
    errors.sort(key=lambda x: x["priority"])
    return errors
```

### Priority Triage System

```python
PRIORITY_LEVELS = {
    1: "SPINE / BACK ROUNDING",       # Always surfaced first
    2: "HIP HINGE FAILURE",           # Deadlift-specific
    3: "DEPTH / RANGE OF MOTION",     # Squat depth, press lockout
    4: "HYPEREXTENSION",              # OHP arch
    5: "MINOR FORM DEVIATION"         # Small elbow position, etc.
}

def get_priority_feedback(errors):
    """
    Returns only the highest-priority error to avoid overwhelming user.
    """
    if not errors:
        return None
    return errors[0]
```

### Fatigue Detection (Pattern Recognition)

```python
def detect_fatigue(rep_history):
    """
    Compares current rep form to earlier reps.
    Detects progressive form breakdown = fatigue signal.
    """
    if len(rep_history) < 3:
        return None

    first_rep_torso  = rep_history[0]["torso_angle"]
    latest_rep_torso = rep_history[-1]["torso_angle"]
    drift = latest_rep_torso - first_rep_torso

    if drift > 10:
        return {
            "type":    "fatigue_detected",
            "message": f"Form degrading over {len(rep_history)} reps — consider ending the set",
            "drift":   round(drift, 1)
        }
    return None
```

---

## 8. Feedback System

### Multi-Modal Feedback

| Mode | Method | When |
|------|--------|------|
| **Text overlay** | OpenCV text on webcam feed | Every frame with error |
| **Voice cue** | pyttsx3 TTS | Once per rep (not every frame) |
| **Color overlay** | Green/Red skeleton | Continuous visual indicator |
| **Snapshot** | Saved frame with annotation | When Red Zone breached |

### Preventing Feedback Spam

```python
last_tts_rep = 0

def deliver_feedback(error, current_rep, tts_engine):
    # Always show text overlay
    display_text_overlay(error["message"])

    # Only speak once per rep (not 30x per second)
    global last_tts_rep
    if current_rep != last_tts_rep:
        tts_engine.say(error["message"])
        tts_engine.runAndWait()
        last_tts_rep = current_rep
```

### TTS Setup

```python
import pyttsx3

def init_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate",   150)   # Speed (words per minute)
    engine.setProperty("volume", 0.9)   # Volume (0.0 to 1.0)
    return engine
```

### Skeleton Color Coding

```python
def draw_skeleton(frame, landmarks, has_error, frame_width, frame_height):
    """
    Draw skeleton overlay. Green = good form, Red = error detected.
    """
    color = (0, 0, 255) if has_error else (0, 255, 0)  # BGR

    connections = mp.solutions.pose.POSE_CONNECTIONS

    for connection in connections:
        start_idx = connection[0]
        end_idx   = connection[1]

        start_point = (
            int(landmarks[start_idx].x * frame_width),
            int(landmarks[start_idx].y * frame_height)
        )
        end_point = (
            int(landmarks[end_idx].x * frame_width),
            int(landmarks[end_idx].y * frame_height)
        )

        cv2.line(frame, start_point, end_point, color, 2)
    return frame
```

---

## 9. Feature Roadmap

### Phase 1: Working Demo (Target: April 7)

| Feature | Difficulty | Description |
|---------|-----------|-------------|
| F1: Pose Estimation | Easy | MediaPipe skeleton overlay on webcam feed |
| F2: Phase Detection | Easy | y-axis tracking: Standing → Descending → Ascending |
| F3: Basic Calibration | Intermediate | Record min/max joint angles over 3–5 reps |
| F6a: Simple Text Feedback | Intermediate | On-screen "Go Lower" / "Rep Complete" overlays |

**April 7 Demo Success Criteria:**
- Webcam opens and streams
- MediaPipe skeleton drawn on live feed
- Hip movement tracked (up/down)
- Rep counter increments correctly
- Current phase displayed on screen

### Phase 2: The Expert Brain (April – May)

| Feature | Difficulty | Description |
|---------|-----------|-------------|
| F5: Safety Validation | Intermediate | Universal Red Zone checking during calibration |
| F7: Priority Logic | Intermediate | Triage system — spine errors surfaced before depth errors |
| F6b: Auditory TTS | Intermediate | Voice coaching via pyttsx3 |

### Phase 3: Final System Integration (May – June)

| Feature | Difficulty | Description |
|---------|-----------|-------------|
| F4: Multi-Exercise Support | Intermediate | Squat + Deadlift + OHP with exercise-specific rules |
| F8: Event-Triggered Snapshots | Intermediate | Auto-capture peak error frames (max 5 per set) |
| F9: Post-Set Diagnostic Review | Intermediate | Session summary dashboard with errors ranked by risk |

### Stretch Goals (Time-Permitting)

| Feature | Difficulty | Notes |
|---------|-----------|-------|
| Enhanced calibration | Difficult | Only if core features are complete and stable |
| Fatigue tracking refinement | Intermediate | Extend pattern recognition across longer sets |

### Acknowledged Future Work (Not Implementing)

- Multi-camera angle support (front view for knee valgus)
- 3D perspective correction (parallax mitigation)
- Mobile application deployment

---

## 10. File & Folder Structure

```
fyp_exercise_analysis/
│
├── app.py                        # Main Streamlit entry point
│
├── core/
│   ├── pose_estimator.py         # MediaPipe setup and landmark extraction
│   ├── angle_calculator.py       # All trigonometry functions
│   ├── phase_detector.py         # Movement phase detection logic
│   ├── form_analyzer.py          # Exercise-specific form rules
│   ├── calibration.py            # Calibration + Red Zone validation
│   ├── feedback_engine.py        # TTS + text overlay delivery
│   └── priority_triage.py        # Error prioritization logic
│
├── exercises/
│   ├── squat.py                  # Squat-specific rules + thresholds
│   ├── deadlift.py               # Deadlift-specific rules + thresholds
│   └── overhead_press.py         # OHP-specific rules + thresholds
│
├── data/
│   ├── red_zones.json            # Universal Red Zone threshold values
│   ├── feedback_messages.json    # All feedback text strings
│   └── sessions/                 # Auto-generated session logs
│       └── session_YYYY-MM-DD.json
│
├── snapshots/                    # Auto-captured peak error frames
│   └── error_TIMESTAMP.jpg
│
├── ui/
│   ├── dashboard.py              # Post-set Streamlit dashboard component
│   └── overlay.py                # OpenCV frame annotation functions
│
├── utils/
│   ├── logger.py                 # Session logging utilities
│   └── helpers.py                # Miscellaneous utility functions
│
├── requirements.txt
└── README.md
```

---

## 11. Implementation Guide (Phase by Phase)

### Phase 1 Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv fyp_env
source fyp_env/bin/activate        # Mac/Linux
fyp_env\Scripts\activate           # Windows

# Install dependencies
pip install mediapipe opencv-python streamlit numpy pyttsx3
```

**requirements.txt:**
```
mediapipe==0.10.9
opencv-python==4.9.0.80
streamlit==1.32.0
numpy==1.26.4
pyttsx3==2.90
```

---

### Phase 1 Step 2: Basic Pose Estimation

```python
# core/pose_estimator.py

import cv2
import mediapipe as mp

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def init_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,           # 0=fast, 1=balanced, 2=accurate
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_frame(frame, pose):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(rgb_frame)
    return results

def get_landmarks(results):
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    return frame
```

---

### Phase 1 Step 3: April 7 Demo (Minimal Working Version)

```python
# app.py — April 7 Demo

import cv2
import streamlit as st
from core.pose_estimator import init_pose, process_frame, get_landmarks, draw_landmarks
from core.phase_detector import detect_phase

st.title("AI Exercise Form Analyzer — Demo")

exercise   = st.selectbox("Select Exercise", ["Squat", "Deadlift", "Overhead Press"])
start      = st.button("Start Analysis")

if start:
    cap        = cv2.VideoCapture(0)
    pose       = init_pose()
    frame_slot = st.empty()

    rep_count  = 0
    prev_hip_y = None
    phase      = "standing"
    prev_phase = "standing"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results   = process_frame(frame, pose)
        landmarks = get_landmarks(results)
        frame     = draw_landmarks(frame, results)

        if landmarks:
            hip_y = landmarks[23].y

            if prev_hip_y is not None:
                phase = detect_phase(hip_y, prev_hip_y)

            if prev_phase == "bottom" and phase == "ascending":
                rep_count += 1

            prev_hip_y = hip_y
            prev_phase = phase

            cv2.putText(frame, f"Phase: {phase.upper()}",
                        (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {rep_count}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame_rgb, channels="RGB")

    cap.release()
```

---

### Phase 1 Step 4: Phase Detector

```python
# core/phase_detector.py

MOVEMENT_THRESHOLD = 0.005

def detect_phase(current_y, previous_y):
    """
    MediaPipe: y increases downward.
    """
    change = current_y - previous_y

    if change > MOVEMENT_THRESHOLD:
        return "descending"
    elif change < -MOVEMENT_THRESHOLD:
        return "ascending"
    else:
        return "bottom"
```

---

### Phase 2 Step 1: Calibration System

```python
# core/calibration.py

import numpy as np

UNIVERSAL_RED_ZONES = {
    "squat": {
        "torso_angle_min": 30,
        "torso_angle_max": 85,
        "knee_angle_min":  65,
        "knee_angle_max":  110,
    },
    "deadlift": {
        "torso_angle_min": 20,
        "torso_angle_max": 80,
    },
    "overhead_press": {
        "lumbar_deviation_max": 20,
    }
}

def run_calibration(angle_history, exercise):
    """
    angle_history: list of dicts with torso_angle, knee_angle per rep
    Returns (baseline_dict, status_string)
    """
    if len(angle_history) < 3:
        return None, "not_enough_reps"

    baseline = {
        "torso_angle": np.mean([r["torso_angle"] for r in angle_history]),
        "knee_angle":  np.mean([r["knee_angle"]  for r in angle_history]),
        "reps":        len(angle_history)
    }

    red_zone  = UNIVERSAL_RED_ZONES.get(exercise, {})
    torso_min = red_zone.get("torso_angle_min", 0)
    torso_max = red_zone.get("torso_angle_max", 180)

    if baseline["torso_angle"] < torso_min:
        return None, "back_too_rounded"
    elif baseline["torso_angle"] > torso_max:
        return None, "excessive_backward_lean"
    else:
        return baseline, "accepted"
```

---

### Phase 2 Step 2: Form Analyzer

```python
# core/form_analyzer.py

DEVIATION_THRESHOLD = 15  # degrees

def analyze_form(landmarks, exercise, baseline, phase):
    errors = []

    shoulder    = (landmarks[11].x, landmarks[11].y)
    hip         = (landmarks[23].x, landmarks[23].y)
    knee        = (landmarks[25].x, landmarks[25].y)
    ankle       = (landmarks[27].x, landmarks[27].y)

    torso_angle = calculate_torso_angle(shoulder, hip)
    knee_angle  = calculate_angle(hip, knee, ankle)

    torso_dev = torso_angle - baseline["torso_angle"]
    knee_dev  = knee_angle  - baseline["knee_angle"]

    if torso_dev > DEVIATION_THRESHOLD:
        errors.append({
            "type":     "back_rounding",
            "priority": 1,
            "severity": "critical" if torso_dev > 25 else "warning",
            "message":  f"Back rounding detected — {torso_dev:.0f}° from your baseline"
        })

    if exercise == "squat" and knee_dev > DEVIATION_THRESHOLD:
        errors.append({
            "type":     "insufficient_depth",
            "priority": 3,
            "severity": "warning",
            "message":  "Go deeper — shallower than your calibration depth"
        })

    errors.sort(key=lambda x: x["priority"])
    return errors
```

---

## 12. Evaluation Plan

### Quantitative Metrics

| Metric | Method | Target |
|--------|--------|--------|
| **Angle accuracy** | System vs. manual protractor measurement | ±5°, 85%+ of frames |
| **Phase detection accuracy** | System labels vs. manual video annotation | 90%+ correct |
| **Form error detection rate** | Test on known good/bad form examples | 80%+ detection |
| **False positive rate** | Incorrect alerts on correct form | Below 15% |
| **Processing speed** | Frame processing time | Under 33ms (≥30 FPS) |

### User Study (Qualitative)

**Participants:** 20+ gym users (beginner/intermediate)

**Protocol:**
1. Participant performs 3 sets of their chosen exercise
2. System provides real-time feedback throughout
3. Post-session survey administered

**Survey Questions (1–5 Likert scale):**
- "Was the feedback accurate?"
- "Was the feedback helpful?"
- "Did it improve your form awareness?"
- "Would you use this regularly?" (Yes/No + reason)
- Open comment: "What would you change?"

### Comparative Analysis

| Method | Description |
|--------|-------------|
| **A (Baseline)** | Single fixed threshold, no calibration, no phases |
| **B (Your System)** | Full system with calibration + phases + priority logic |
| **C (Control)** | No feedback — user lifts without assistance |

**Hypothesis:** Method B significantly outperforms A and C on error detection accuracy and user satisfaction.

### Validation Dataset

Record 30–50 test videos covering:
- Known correct form (5 reps × 3 exercises)
- Known incorrect form (specific errors deliberately introduced)
- Different body types (tall/short, different builds)

Manual frame-by-frame annotation provides ground truth for accuracy measurement.

---

## 13. Known Limitations

| Limitation | Reason | Mitigation |
|-----------|--------|------------|
| Cannot detect knee valgus | Requires front-view camera | Documented as future work |
| Cannot detect stance width | Requires front-view camera | Same as above |
| Cannot detect left-right asymmetry | Requires front-view camera | Same as above |
| Camera angle sensitivity | Must be true side view | Clear setup instructions for users |
| Barbell not tracked | System focuses on body mechanics only | Documented; body position is primary injury factor |
| Lighting-dependent accuracy | Poor lighting degrades landmark detection | Recommend adequate lighting in user guide |

---

## 14. Viva Defense Notes

### "What's AI about this project?"

> "The project uses two layers of AI. First, MediaPipe's deep neural networks detect and track 33 body keypoints in real-time — this is the perception layer. Second, I built an intelligent analysis system on top: it applies biomechanical reasoning, makes adaptive decisions through personalized calibration, recognizes temporal patterns across reps, and generates context-aware feedback based on movement phase. This is applied AI — combining deep learning for perception with knowledge-based expert system logic for intelligent decision-making."

### "You're just using MediaPipe — where's your contribution?"

> "MediaPipe gives me raw coordinates — it's a sensor. My contribution is the intelligence layer: the biomechanical angle algorithms, the two-tier calibration system that personalizes to each user's body mechanics, the state machine detecting exercise phases, the priority triage surfacing the most dangerous error first, and the pattern recognition tracking form degradation across reps. Around 80–90% of the backend is my custom code. MediaPipe is the tool; I built the brain."

### "Why not train your own model?"

> "Training a pose model from scratch requires large labeled datasets and significant compute — and it's a solved problem. Reinventing it adds no research value for an undergraduate FYP. My contribution is in the intelligent analysis layer: the biomechanical rules, calibration logic, and feedback system that turn raw pose data into actionable injury prevention coaching. This is applied AI system development, which is the appropriate scope for this project."

### "How does the system handle different body types?"

> "Two-tier approach. First, Universal Red Zones set non-negotiable safety limits for everyone based on sports science research — no one proceeds with a dangerously rounded back regardless of body type. Second, calibration records each user's natural movement over 3–5 reps and establishes their personal baseline within those safe limits. During the workout, the system flags deviations from the user's own baseline — not deviations from a universal ideal. A tall person's 42° forward lean is their baseline; a shorter person's 68° upright posture is theirs. Both are only flagged when their own form breaks down."

### "What are your limitations?"

> "The main limitation is single-camera side view, which prevents detecting frontal plane errors like knee valgus and lateral asymmetry. These require a front-facing camera and are documented as future work. The system also requires adequate lighting and correct side-view positioning, addressed in the user guide. These are accepted trade-offs given the FYP scope."

---

*Document Version: 1.0 | Last Updated: June 2025*
*FYP — CS (AI) | Supervisor: TS. Dr. Law Foong Li*
