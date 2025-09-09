# ai_workout_coach_manual_select_smallhud.py
# Manual selection (S=Squat, P=Push-up, B=Bicep)
# Keeps: set+rest timer (squats & pushups), voice+text coaching, camera angle feedback
# New: smaller semi-transparent HUD, Fullscreen toggle (F/N), Hide HUD (H)

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import random
from collections import deque

# ---------------------- Configuration ----------------------
MIN_DETECT_CONF = 0.6
MIN_TRACK_CONF  = 0.6
ANGLE_SMOOTH_WINDOW = 5
HOLD_FRAMES = 3
FEEDBACK_THROTTLE = 5.0
RECOMMEND_DISPLAY_SEC = 7.0
MOTIVATION_COOLDOWN = (12.0, 22.0)
SET_SIZE = 10
REST_SECONDS = 30

# ---------------------- TTS (non-blocking) ----------------------
_engine_lock = threading.Lock()
engine = pyttsx3.init()
engine.setProperty("rate", 165)
def speak_async(text):
    def _run(txt):
        try:
            with _engine_lock:
                engine.say(txt)
                engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, args=(text,), daemon=True).start()

# ---------------------- Mediapipe ----------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose     = mp.solutions.pose

# ---------------------- Helpers ----------------------
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(angle))

def avg_point(lm, idx1, idx2, w, h):
    return [((lm[idx1].x + lm[idx2].x) / 2.0) * w,
            ((lm[idx1].y + lm[idx2].y) / 2.0) * h]

def safe_point(lm, idx, w, h):
    return [lm[idx].x * w, lm[idx].y * h]

def has_landmarks(lm, indices, min_vis=0.5):
    try:
        for idx in indices:
            if lm[idx].visibility < min_vis:
                return False
        return True
    except Exception:
        return False

# ---------------------- Coaching texts ----------------------
SUGGESTIONS = {
    "back_too_bent":     "Keep chest up and straighten your back.",
    "knees_too_forward": "Push hips back; keep knees above toes.",
    "hips_sagging":      "Tighten core and lift hips to a straight line.",
    "arms_too_straight": "Bend elbows more to engage the muscles.",
    "curl_too_short":    "Curl higher—bring hand closer to shoulder.",
    "curl_fully_up":     "Nice full curl. Keep elbows close.",
    "keep_elbows_fixed": "Keep elbows fixed to your sides.",
    "pushup_body_drop":  "Keep a straight line head to heels.",
    "pushup_shallow":    "Lower till your chest is near the floor."
}
MOTIVATION = [
    "Great form, keep going!",
    "Halfway there, you got this!",
    "Nice pace, stay strong!",
    "Final reps, give your best!",
    "Breathe, control, and push!"
]

# ---------------------- Exercise defs ----------------------
EXERCISES = {
    "squat":       {"down_angle": 90,  "up_angle": 160},
    "pushup":      {"down_angle": 60,  "up_angle": 160},
    "bicep_curl":  {"down_angle": 160, "up_angle": 40 }
}

# ---------------------- Tracker ----------------------
class ExerciseTracker:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.stage = None
        self._hold_up = 0
        self._hold_down = 0
        self.in_rest = False
        self.rest_ends_at = 0.0

    def reset(self):
        self.counter = 0
        self.stage = None
        self._hold_up = 0
        self._hold_down = 0
        self.in_rest = False
        self.rest_ends_at = 0.0

    def start_rest(self, seconds):
        self.in_rest = True
        self.rest_ends_at = time.time() + seconds

    def update_rest(self):
        if self.in_rest and time.time() >= self.rest_ends_at:
            self.in_rest = False
            return True
        return False

    def update(self, angle, down_th, up_th, is_bicep=False):
        if self.in_rest:
            return False
        counted = False
        if is_bicep:
            if angle >= down_th:
                self._hold_down += 1; self._hold_up = 0
                if self._hold_down >= HOLD_FRAMES:
                    self.stage = "down"
            elif angle <= up_th:
                self._hold_up += 1; self._hold_down = 0
                if self._hold_up >= HOLD_FRAMES and self.stage == "down":
                    self.counter += 1; counted = True; self.stage = "up"
            else:
                self._hold_up = self._hold_down = 0
        else:
            if angle > up_th:
                self._hold_up += 1; self._hold_down = 0
                if self._hold_up >= HOLD_FRAMES:
                    if self.stage == "down":
                        self.counter += 1; counted = True
                    self.stage = "up"
            elif angle < down_th:
                self._hold_down += 1; self._hold_up = 0
                if self._hold_down >= HOLD_FRAMES:
                    self.stage = "down"
            else:
                self._hold_up = self._hold_down = 0
        return counted

# ---------------------- Buffers ----------------------
angle_buffers = {
    "squat": deque(maxlen=ANGLE_SMOOTH_WINDOW),
    "pushup": deque(maxlen=ANGLE_SMOOTH_WINDOW),
    "bicep_curl": deque(maxlen=ANGLE_SMOOTH_WINDOW),
}
def smooth_angle(name, a):
    angle_buffers[name].append(a)
    return float(sum(angle_buffers[name]) / len(angle_buffers[name]))

# ---------------------- Form checks ----------------------
def check_squat(lm, w, h):
    hip = avg_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
    knee = avg_point(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h)
    ankle = avg_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)
    shoulder = avg_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)

    knee_angle = calculate_angle(hip, knee, ankle)
    back_angle = calculate_angle(shoulder, hip, knee)

    errors = []
    if back_angle < 140: errors.append("back_too_bent")
    if (knee[0] - ankle[0]) > (0.08 * w): errors.append("knees_too_forward")
    return knee_angle, errors

def check_pushup(lm, w, h):
    shoulder = avg_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
    hip      = avg_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value,       mp_pose.PoseLandmark.RIGHT_HIP.value,       w, h)
    ankle    = avg_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value,     mp_pose.PoseLandmark.RIGHT_ANKLE.value,     w, h)
    elbow    = avg_point(lm, mp_pose.PoseLandmark.LEFT_ELBOW.value,     mp_pose.PoseLandmark.RIGHT_ELBOW.value,     w, h)
    wrist    = avg_point(lm, mp_pose.PoseLandmark.LEFT_WRIST.value,     mp_pose.PoseLandmark.RIGHT_WRIST.value,     w, h)

    body_angle = calculate_angle(shoulder, hip, ankle)
    arm_angle  = calculate_angle(shoulder, elbow, wrist)

    errors = []
    if body_angle < 165: errors.append("pushup_body_drop")
    if arm_angle > 160:  errors.append("arms_too_straight")
    if 100 < arm_angle < 160: errors.append("pushup_shallow")
    return arm_angle, errors

def check_bicep(lm, w, h):
    shoulder = avg_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
    elbow    = avg_point(lm, mp_pose.PoseLandmark.LEFT_ELBOW.value,     mp_pose.PoseLandmark.RIGHT_ELBOW.value,     w, h)
    wrist    = avg_point(lm, mp_pose.PoseLandmark.LEFT_WRIST.value,     mp_pose.PoseLandmark.RIGHT_WRIST.value,     w, h)

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    errors = []
    if elbow_angle > 160: errors += ["keep_elbows_fixed", "curl_too_short"]
    elif elbow_angle < 40: errors.append("curl_fully_up")
    return elbow_angle, errors

# ---------------------- Camera guidance ----------------------
UPPER_BODY = [
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,    mp_pose.PoseLandmark.RIGHT_WRIST.value,
]
LOWER_BODY = [
    mp_pose.PoseLandmark.LEFT_HIP.value,   mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,  mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
]

def camera_feedback(lm):
    msgs = []
    if not has_landmarks(lm, UPPER_BODY, 0.5) and not has_landmarks(lm, LOWER_BODY, 0.5):
        msgs.append("Please step back for full-body tracking.")
    elif not has_landmarks(lm, LOWER_BODY, 0.5):
        msgs.append("Your lower body isn’t visible.")
    elif not has_landmarks(lm, UPPER_BODY, 0.5):
        msgs.append("Move to center of the camera.")
    return msgs

# ---------------------- Main ----------------------
def main():
    window_name = "AI Workout Coach"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)   # allow resizing / fullscreen
    cv2.resizeWindow(window_name, 960, 720)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    trackers = {n: ExerciseTracker(n) for n in EXERCISES.keys()}
    last_feedback_time = {n: 0.0 for n in EXERCISES.keys()}
    last_feedback_time["camera"] = 0.0

    last_recommend = ("", 0.0)
    current_ex = None  # manual selection
    show_hud = True
    fullscreen = False

    next_motivation_time = time.time() + random.uniform(*MOTIVATION_COOLDOWN)

    with mp_pose.Pose(min_detection_confidence=MIN_DETECT_CONF, min_tracking_confidence=MIN_TRACK_CONF) as pose:
        speak_async("Started. Press S for squats, P for push-ups, B for bicep curls, F for fullscreen, H to hide panel, Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # process pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            angle = None
            errors = []
            counted = False
            camera_msgs = []

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Camera guidance (throttled)
                camera_msgs = camera_feedback(lm)
                if camera_msgs:
                    now = time.time()
                    if now - last_feedback_time["camera"] > 6.0:
                        speak_async(camera_msgs[0])
                        last_feedback_time["camera"] = now

                # Track only the chosen exercise
                if current_ex:
                    if current_ex == "squat":
                        raw_angle, errors = check_squat(lm, w, h)
                        angle = smooth_angle("squat", raw_angle)
                        counted = trackers["squat"].update(
                            angle, EXERCISES["squat"]["down_angle"], EXERCISES["squat"]["up_angle"]
                        )
                    elif current_ex == "pushup":
                        raw_arm, errors = check_pushup(lm, w, h)
                        angle = smooth_angle("pushup", raw_arm)
                        counted = trackers["pushup"].update(
                            angle, EXERCISES["pushup"]["down_angle"], EXERCISES["pushup"]["up_angle"]
                        )
                    else:  # bicep_curl
                        raw_angle, errors = check_bicep(lm, w, h)
                        angle = smooth_angle("bicep_curl", raw_angle)
                        counted = trackers["bicep_curl"].update(
                            angle, EXERCISES["bicep_curl"]["down_angle"], EXERCISES["bicep_curl"]["up_angle"], is_bicep=True
                        )

                    # Set & Rest logic (squat/pushup only)
                    if counted:
                        speak_async(f"{current_ex.replace('_',' ')} {trackers[current_ex].counter}")
                        last_recommend = (f"Counted: {trackers[current_ex].counter}", time.time())
                        if current_ex in ("squat", "pushup"):
                            if trackers[current_ex].counter % SET_SIZE == 0:
                                trackers[current_ex].start_rest(REST_SECONDS)
                                speak_async("Set complete! Take 30 seconds rest.")
                                last_recommend = ("Set complete! Take 30 sec rest.", time.time())

                # Coaching (throttled)
                now = time.time()
                if errors and (now - last_feedback_time.get(current_ex or "none", 0.0) > FEEDBACK_THROTTLE):
                    spoken = []
                    for e in errors:
                        msg = SUGGESTIONS.get(e)
                        if msg and msg not in spoken:
                            spoken.append(msg)
                    if spoken:
                        speak_async(". ".join(spoken))
                        last_feedback_time[current_ex or "none"] = now
                        last_recommend = (" / ".join(spoken), now)

                # Motivational cues (not during rest)
                if current_ex:
                    tr = trackers[current_ex]
                    if (not tr.in_rest) and time.time() >= next_motivation_time and tr.counter > 0:
                        cue = random.choice(MOTIVATION)
                        speak_async(cue)
                        last_recommend = (cue, time.time())
                        next_motivation_time = time.time() + random.uniform(*MOTIVATION_COOLDOWN)

                # Landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ---------------------- UI ----------------------
            # Small semi-transparent HUD (top-left)
            if show_hud:
                panel_w, panel_h = 300, 110   # smaller panel
                x0, y0 = 10, 10
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

                # Exercise title
                ex_display = (current_ex or "Select exercise").replace("_"," ").upper()
                cv2.putText(frame, ex_display, (x0+10, y0+28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

                # Reps & stage
                if current_ex:
                    count = trackers[current_ex].counter
                    stage = trackers[current_ex].stage or "—"
                    cv2.putText(frame, f"REPS: {count}", (x0+10, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                    cv2.putText(frame, f"STAGE: {stage.upper()}", (x0+10, y0+92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,255,180), 2)

            # Angle/progress (placed just below HUD)
            if current_ex and angle is not None:
                if current_ex == "bicep_curl":
                    down, up = EXERCISES["bicep_curl"]["down_angle"], EXERCISES["bicep_curl"]["up_angle"]
                    pct = max(0.0, min(1.0, (down - angle) / max(1.0, down - up)))
                    bar_x, bar_y = 16, (10 + (110 if show_hud else 0) + 20)
                    bar_w, bar_h = 280, 16
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*pct), bar_y+bar_h), (0,200,255), -1)
                    cv2.putText(frame, f"{int(pct*100)}% curl", (bar_x, bar_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)
                else:
                    y_text = (10 + (110 if show_hud else 0) + 30)
                    cv2.putText(frame, f"Angle: {int(angle)}\N{DEGREE SIGN}", (16, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Tips / feedback
            rec_msg, rec_time = last_recommend
            if rec_msg and (time.time() - rec_time) <= RECOMMEND_DISPLAY_SEC:
                lines = rec_msg.split(" / ")
                # print up to 3 lines near lower-left
                base_y = int(h*0.55)
                for i, line in enumerate(lines[:3]):
                    y = base_y + i*32
                    cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,165,255), 2)

            # Rest overlay (squat/pushup)
            if current_ex in ("squat", "pushup"):
                tr = trackers[current_ex]
                if tr.in_rest:
                    remaining = int(max(0, tr.rest_ends_at - time.time()))
                    overlay2 = frame.copy()
                    cv2.rectangle(overlay2, (0,0), (w,h), (0,0,0), -1)
                    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
                    cv2.putText(frame, "REST", (int(w*0.35), int(h*0.35)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,255,255), 6)
                    cv2.putText(frame, f"{remaining}", (int(w*0.45), int(h*0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,215,255), 8)
                    if tr.update_rest():
                        speak_async("Rest over. Start the next set.")

            # Bottom instructions
            cv2.putText(frame, "S=Squat  P=Pushup  B=Bicep  H=HideHUD  F=Fullscreen  N=Normal  Q=Quit",
                        (16, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)

            # Camera guidance
            if camera_msgs:
                for i, m in enumerate(camera_msgs[:2]):
                    cv2.putText(frame, m, (16, h-50 - 28*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow(window_name, frame)

            # ---------------------- Key handling ----------------------
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord('q')):  # Esc or q
                speak_async("Exiting. Good job!")
                break
            elif key == ord('s'):
                current_ex = "squat"
                trackers[current_ex].reset()
                angle_buffers["squat"].clear()
                last_recommend = ("Now tracking squats. Keep chest up, hips back.", time.time())
                speak_async("Now tracking squats")
            elif key == ord('p'):
                current_ex = "pushup"
                trackers[current_ex].reset()
                angle_buffers["pushup"].clear()
                last_recommend = ("Now tracking pushups. Keep a straight line head to heels.", time.time())
                speak_async("Now tracking pushups")
            elif key == ord('b'):
                current_ex = "bicep_curl"
                trackers[current_ex].reset()
                angle_buffers["bicep_curl"].clear()
                last_recommend = ("Now tracking bicep curls. Keep elbows fixed to your sides.", time.time())
                speak_async("Now tracking bicep curls")
            elif key == ord('h'):
                show_hud = not show_hud
            elif key == ord('f'):
                fullscreen = True
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('n'):
                fullscreen = False
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 720)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()