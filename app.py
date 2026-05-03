import os
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from google import genai
from google.genai import types
from gtts import gTTS
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent
API_KEY  = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY"):
                API_KEY = line.split("=", 1)[-1].strip().strip('"').strip("'")
                break

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found.\n"
        "Add it to your .env file as: GEMINI_API_KEY=your_key_here\n"
        "Get a free key at https://aistudio.google.com"
    )

print(f"API KEY LOADED: {bool(API_KEY)} ({API_KEY[:8]}...)")

# =========================
# FLASK APP
# =========================
app = Flask(__name__, template_folder="templates", static_folder="static")

# =========================
# GEMINI CLIENT  (google-genai package)
# =========================
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# =========================
# MEDIAPIPE SETUP  (Tasks API — replaces deprecated mp.solutions.hands)
# =========================
HAND_MODEL_PATH = str(BASE_DIR / "models" / "hand_landmarker.task")

# 21-landmark hand skeleton (index pairs to draw as line connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

_base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

# Streaming detector (VIDEO mode — needs monotonic timestamps)
hands_detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=_base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_tracking_confidence=0.5,
    )
)
_video_ts_lock = threading.Lock()
_last_video_ts = 0  # ms; must be strictly increasing per detect_for_video call

# Static detector (IMAGE mode — for snapshot analysis)
hands_static = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=_base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
)


def _draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 220, 255), -1)

# =========================
# GLOBALS
# =========================
camera        = None
camera_lock   = threading.Lock()
is_running    = False
hand_detected = False
last_result   = {"text": "", "timestamp": 0}

# =========================
# CAMERA HELPERS
# =========================
def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            for index in [0, 1, 2]:
                for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                    cam = cv2.VideoCapture(index, backend)
                    if cam.isOpened():
                        camera = cam
                        break
                if camera and camera.isOpened():
                    break
            if camera is None or not camera.isOpened():
                raise RuntimeError("Camera not accessible. Check if another app is using it.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def release_camera():
    global camera
    with camera_lock:
        if camera:
            camera.release()
            camera = None
        cv2.destroyAllWindows()


# =========================
# MEDIAPIPE HELPERS
# =========================
def process_with_mediapipe(frame):
    """Draw landmarks on a live frame and update hand_detected flag."""
    global hand_detected, _last_video_ts
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with _video_ts_lock:
        ts_ms = max(int(time.monotonic() * 1000), _last_video_ts + 1)
        _last_video_ts = ts_ms

    result = hands_detector.detect_for_video(mp_image, ts_ms)
    hand_detected = bool(result.hand_landmarks)
    for lms in result.hand_landmarks:
        _draw_hand(frame, lms)
    return frame, hand_detected


def get_finger_description(frame):
    """
    Run the static (accurate) MediaPipe detector on a single frame
    and return a text description of which fingers are extended.
    """
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = hands_static.detect(mp_image)
    if not result.hand_landmarks:
        return None, False

    descriptions = []
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        lm = hand_landmarks
        # Thumb direction depends on which side of the frame the wrist is
        thumb_up  = lm[4].x < lm[3].x if lm[0].x < 0.5 else lm[4].x > lm[3].x
        index_up  = lm[8].y  < lm[5].y
        middle_up = lm[12].y < lm[9].y
        ring_up   = lm[16].y < lm[13].y
        pinky_up  = lm[20].y < lm[17].y

        fingers = []
        if thumb_up:  fingers.append("thumb")
        if index_up:  fingers.append("index")
        if middle_up: fingers.append("middle")
        if ring_up:   fingers.append("ring")
        if pinky_up:  fingers.append("pinky")

        count = len(fingers)
        label = (
            f"Hand {idx+1}: [{', '.join(fingers) if fingers else 'fist - no fingers extended'}] "
            f"({count} finger{'s' if count != 1 else ''} up)"
        )
        descriptions.append(label)

    return "\n".join(descriptions), True


# =========================
# FRAME CAPTURE
# =========================
def capture_best_frame(num_frames=15, delay=0.04):
    """Capture several frames and return the sharpest one."""
    try:
        cap    = get_camera()
        frames = []
        for _ in range(num_frames):
            ok, frame = cap.read()
            if ok and frame is not None:
                f     = cv2.flip(frame, 1)
                gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                frames.append((score, f))
            time.sleep(delay)
        if not frames:
            return None
        frames.sort(key=lambda x: x[0], reverse=True)
        return frames[0][1]
    except Exception as e:
        print("Capture error:", e)
        return None


# =========================
# GEMINI INTERPRETATION
# =========================
def interpret_gesture():
    # Try up to 4 times to get a frame with a hand
    frame       = None
    finger_desc = None

    for _ in range(4):
        candidate = capture_best_frame(num_frames=12, delay=0.04)
        if candidate is None:
            time.sleep(0.1)
            continue
        desc, has_hand = get_finger_description(candidate)
        if has_hand:
            frame       = candidate
            finger_desc = desc
            break
        time.sleep(0.15)

    if frame is None:
        return None, "Could not capture a frame from the camera."
    if not finger_desc:
        return "No hand detected. Show your hand clearly in the frame.", None

    # ---- Prepare images ----
    h, w  = frame.shape[:2]
    pad   = 20
    y1, y2 = max(0, h//4 - pad), min(h, 3*h//4 + pad)
    x1, x2 = max(0, w//4 - pad), min(w, 3*w//4 + pad)
    crop  = cv2.GaussianBlur(frame[y1:y2, x1:x2], (3, 3), 0)

    def to_jpeg_bytes(f):
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None

    full_bytes = to_jpeg_bytes(frame)
    crop_bytes = to_jpeg_bytes(crop)

    prompt = f"""You are a professional sign language interpreter specializing in ASL (American Sign Language).

MediaPipe hand tracking detected:
{finger_desc}

Two images are attached: the full camera frame and a cropped close-up of the hand.
Use BOTH the images AND the hand data to identify the gesture.

Common ASL reference:
- Fist (no fingers) = Stop / No / A
- Thumb only = Good / Thumbs Up / Yes
- Index only = One / Point / D
- Index + Middle = Peace / Two / V
- Index + Middle + Ring = Three / W
- Index + Middle + Ring + Pinky = Four / B
- All five fingers open = Five / Hello / Wave
- Thumb + Pinky = Call Me / Y
- Thumb + Index = L / Gun shape
- Thumb + Index + Pinky = I Love You / ILY
- Index + Pinky = Rock / Horn
- OK sign (index curled to thumb) = OK / Zero / O
- All fingers spread wide = Stop / Five / Open Hand

Respond with ONLY the gesture name or meaning.
Maximum 4 words. No punctuation. No explanation. No markdown."""

    try:
        # ---- google-genai SDK call ----
        contents = [prompt]
        if full_bytes:
            contents.append(types.Part.from_bytes(data=full_bytes, mime_type="image/jpeg"))
        if crop_bytes:
            contents.append(types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
        )

        text = ""
        if hasattr(response, "text") and response.text:
            text = response.text.strip()
        elif getattr(response, "candidates", None):
            try:
                text = response.candidates[0].content.parts[0].text.strip()
            except Exception:
                text = ""

        text = text.replace("*", "").replace('"', "").replace("'", "").strip()
        if not text:
            text = "Unclear gesture"

        print(f"[Gemini] Result: '{text}' | Fingers: {finger_desc}")
        return text, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Gemini API error: {str(e)}"


# =========================
# TEXT TO SPEECH
# =========================
def text_to_audio(text, speed="normal"):
    try:
        audio_dir = BASE_DIR / "static" / "audio"
        os.makedirs(str(audio_dir), exist_ok=True)
        path = str(audio_dir / "output.mp3")
        tts  = gTTS(text=text, lang="en", slow=(speed == "slow"))
        tts.save(path)
        return f"/static/audio/output.mp3?t={int(time.time())}"
    except Exception as e:
        print("TTS error:", e)
        return None


# =========================
# VIDEO STREAM
# =========================
def generate_frames():
    while True:
        if not is_running:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera OFF", (220, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
            cv2.putText(blank, "Press Start Camera", (160, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
            _, buf = cv2.imencode(".jpg", blank)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.2)
            continue

        try:
            cap       = get_camera()
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            frame, detected = process_with_mediapipe(frame)

            h, w        = frame.shape[:2]
            box_color   = (0, 255, 100) if detected else (0, 140, 255)
            status_text = "Hand Detected  Click Translate!" if detected else "Show your hand in the box"

            cv2.rectangle(frame, (0, 0), (w, 40), (8, 12, 24), -1)
            cv2.putText(frame, status_text, (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, box_color, 2)
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), box_color, 2)
            for x, y in [(w//4, h//4), (3*w//4, h//4), (w//4, 3*h//4), (3*w//4, 3*h//4)]:
                cv2.circle(frame, (x, y), 5, box_color, -1)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        except Exception as e:
            print("Stream error:", e)
            time.sleep(0.2)


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
def start():
    global is_running
    try:
        get_camera()
        is_running = True
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stop", methods=["POST"])
def stop():
    global is_running
    is_running = False
    release_camera()
    return jsonify({"status": "stopped"})

@app.route("/hand_status")
def hand_status():
    return jsonify({"hand_detected": hand_detected})

@app.route("/translate", methods=["POST"])
def translate():
    global last_result
    if not is_running:
        return jsonify({"error": "Camera not started. Press Start Camera first."}), 400

    data  = request.get_json(silent=True) or {}
    speed = data.get("speed", "normal")

    text, error = interpret_gesture()
    if error:
        return jsonify({"error": error}), 500

    audio_path = text_to_audio(text, speed)
    last_result["text"]      = text
    last_result["timestamp"] = time.time()

    return jsonify({"text": text, "audio": audio_path})

@app.route("/last_result")
def get_last_result():
    return jsonify(last_result)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(str(BASE_DIR / "templates"), exist_ok=True)
    os.makedirs(str(BASE_DIR / "static" / "audio"), exist_ok=True)
    print("=" * 50)
    print("  SignSpeak — Sign Language Translator")
    print("  Open http://127.0.0.1:5055 in browser")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5055, debug=False, threaded=True)