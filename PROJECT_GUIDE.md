# SignSpeak — Project Guide

A walkthrough of what your junior built, how the pieces fit together, how to install and run it with `uv`, and what to look at first when reviewing.

---

## 1. What this project actually does

**SignSpeak** is a small Flask web app that turns hand gestures (American Sign Language–style) into spoken English in the browser.

The flow, end to end:

1. The browser opens `index.html` and shows a live webcam feed.
2. The Flask backend (`app.py`) reads frames from the webcam with **OpenCV**.
3. **MediaPipe Hands** runs on every frame to draw hand landmarks and detect whether a hand is visible.
4. When the user clicks **Translate**, the backend:
   - grabs the sharpest of ~12 recent frames,
   - asks MediaPipe which fingers are extended,
   - sends the image + finger description to **Google Gemini** (`gemini-2.0-flash`) and asks it to name the gesture,
   - converts that text to an MP3 with **gTTS** (Google Text-to-Speech),
   - returns `{ text, audio }` to the frontend so it can show the word and play the audio.

So it’s a thin pipeline: **camera → MediaPipe → Gemini → gTTS → browser**.

---

## 2. Project structure

```
SignSpeak/
├── app.py                  ← the real application (Flask + camera + ML)
├── main.py                 ← junk / leftover from `uv init`, just prints "Hello"
├── pyproject.toml          ← uv project file (currently INCOMPLETE — see §4)
├── uv.lock                 ← uv's pinned lockfile
├── .python-version         ← tells uv to use Python 3.12
├── .env                    ← holds GEMINI_API_KEY (do NOT commit this)
├── .gitignore
├── README.md               ← empty
├── templates/
│   └── index.html          ← entire UI: HTML + inline CSS + inline JS
├── static/
│   ├── css/                ← empty (CSS is inlined into index.html)
│   ├── js/                 ← empty (JS is inlined into index.html)
│   └── audio/
│       └── output.mp3      ← regenerated every time /translate runs
├── .venv/                  ← uv's virtualenv (auto-created)
└── venv/                   ← OLD virtualenv from before uv was introduced; safe to delete
```

Important things to know about the layout:

- **`main.py` is not the entrypoint.** `uv init` created it as a stub. The actual app is `app.py`. You can delete `main.py` or ignore it.
- **There are TWO virtualenvs** (`venv/` and `.venv/`). `venv/` is leftover from before you ran `uv init`. Only `.venv/` matters now. You can delete `venv/`.
- **`templates/index.html` contains everything for the UI** — markup, styling, and the JavaScript that calls `/start`, `/stop`, `/translate`, etc. The `static/css/` and `static/js/` folders are empty placeholders.
- **`static/audio/output.mp3`** is rewritten on every translate request. It’s a runtime artefact; consider gitignoring it.

---

## 3. How the code is wired (file-by-file map)

### `app.py` — the whole backend

Sections (the file is already commented with banner-style headers):

| Section | What it does |
|---|---|
| **CONFIGURATION** | Loads `GEMINI_API_KEY` from env, falls back to parsing `.env` manually. Raises if missing. |
| **FLASK APP** | Standard Flask app pointing at `templates/` and `static/`. |
| **GEMINI CLIENT** | `genai.configure(...)` then builds a `GenerativeModel("gemini-2.0-flash")`. |
| **MEDIAPIPE SETUP** | Two `Hands` detectors: `hands_detector` (streaming, faster) and `hands_static` (one-shot, accurate). |
| **GLOBALS** | `camera`, `camera_lock` (a `threading.Lock`), `is_running`, `hand_detected`, `last_result`. These are **shared mutable state** — relevant during review. |
| **CAMERA HELPERS** | `get_camera()` opens the first working webcam (tries indices 0/1/2 with two backends), `release_camera()` shuts it down. |
| **MEDIAPIPE HELPERS** | `process_with_mediapipe(frame)` draws landmarks on a live frame; `get_finger_description(frame)` returns a string like `"Hand 1: [thumb, index] (2 fingers up)"`. |
| **FRAME CAPTURE** | `capture_best_frame()` grabs N frames and returns the sharpest using Laplacian variance (a focus metric). |
| **GEMINI INTERPRETATION** | `interpret_gesture()` retries up to 4× to get a frame with a hand, crops it, builds a prompt that lists common ASL letters/signs, and calls `model.generate_content([prompt, full_image, crop])`. |
| **TEXT TO SPEECH** | `text_to_audio(text, speed)` writes `static/audio/output.mp3` via `gTTS`. |
| **VIDEO STREAM** | `generate_frames()` is a Python generator that yields multipart-JPEG chunks for an MJPEG stream. |
| **ROUTES** | The HTTP endpoints (see table below). |
| **MAIN** | Runs Flask on `127.0.0.1:5055` with `threaded=True`. |

#### HTTP routes (the API the frontend talks to)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Renders `index.html`. |
| GET | `/video_feed` | MJPEG stream consumed by an `<img>` tag in the page. |
| POST | `/start` | Opens the camera, sets `is_running = True`. |
| POST | `/stop` | Sets `is_running = False`, releases camera. |
| GET | `/hand_status` | Returns `{hand_detected: bool}` so the UI can light up the Translate button. |
| POST | `/translate` | Body: `{speed: "normal"\|"slow"}`. Returns `{text, audio}`. The main action. |
| GET | `/last_result` | Returns the last successful translation. |

### `templates/index.html` — the entire frontend

A single HTML file with:
- inline CSS (custom properties, grid background, animated glow orb),
- markup for header / video panel / result panel / controls,
- inline JavaScript that:
  - calls `POST /start` and `POST /stop`,
  - polls `GET /hand_status` to enable/disable the Translate button,
  - calls `POST /translate` and updates the result text + plays `output.mp3`.

If you want to refactor, splitting the `<style>` block into `static/css/main.css` and the `<script>` block into `static/js/main.js` is a clean first step — those folders already exist for that reason.

---

## 4. Why `uv sync` doesn’t install everything

Open `pyproject.toml`. It currently lists **only** `opencv-python`. So `uv sync` only installs OpenCV, even though `app.py` imports a bunch of other packages. That’s why you’ve been adding them one at a time with `uv add`.

The full set of runtime imports `app.py` actually uses:

| Import in code | Package to install |
|---|---|
| `cv2` | `opencv-python` |
| `numpy` | `numpy` |
| `flask` | `flask` |
| `google.generativeai` | `google-generativeai` |
| `gtts` | `gtts` |
| `PIL` | `pillow` |
| `mediapipe` | `mediapipe` |

`os`, `io`, `time`, `threading`, `pathlib` are stdlib — no install needed.

### Fix it once with one command

From the project root:

```bash
uv add flask numpy google-generativeai gtts pillow mediapipe
```

`opencv-python` is already there, so you don’t need to re-add it. After this, `pyproject.toml` will list every dependency, `uv.lock` will be regenerated, and from now on a fresh clone just needs:

```bash
uv sync
```

…and everything installs in one shot.

> Note on Python version: `pyproject.toml` requires `>=3.12` and `.python-version` pins `3.12`. MediaPipe wheels do exist for 3.12 on Linux, so this should be fine. If `uv add mediapipe` ever fails, drop to `3.11` by editing `.python-version` and `requires-python` and re-running `uv sync`.

---

## 5. How to install and run (clean steps)

```bash
# 1. From the project root
cd "/media/fuzail/Work Data/SignSpeak"

# 2. (one-time) Make pyproject complete:
uv add flask numpy google-generativeai gtts pillow mediapipe

# 3. Make sure .env has your Gemini key (already present in your case):
#    GEMINI_API_KEY=...

# 4. Run the app
uv run app.py
```

Then open <http://127.0.0.1:5055> in a browser. Click **Start Camera**, wave a hand at the box, click **Translate**.

If the camera doesn’t open: another app (Zoom, browser tab, etc.) is probably holding `/dev/video0`. Close it and retry.

### Useful one-offs

```bash
uv run python -c "import cv2, mediapipe, flask; print('ok')"   # smoke-test deps
uv lock --upgrade                                              # bump lockfile
uv tree                                                        # see resolved deps
```

---

## 6. What to focus on during review

These are the things most likely to need attention. None of them block running the app — they’re review notes.

1. **Secret in `.env` is committed-readable.** `.env` is not in `.gitignore`. Add `.env` to `.gitignore` and rotate the Gemini key (assume it’s leaked the moment it touches a repo). Also gitignore `static/audio/output.mp3` and `venv/`.
2. **Two virtualenvs.** Delete `venv/` — only `.venv/` is used by uv.
3. **`main.py` is dead code.** It’s a leftover from `uv init`. Delete it (and consider setting `app.py` as a script entrypoint in `pyproject.toml`).
4. **Global mutable state + threads.** `camera`, `is_running`, `hand_detected`, `last_result` are module-level globals mutated from the request handlers AND from the streaming generator running in another thread. `camera` is protected by `camera_lock`; the others are not. With `threaded=True` Flask, `last_result` writes can race. Low risk in practice, worth noting.
5. **`hand_detected` is updated by the *streaming* generator.** If `/video_feed` isn’t actively being consumed by a browser, `hand_detected` never updates — but `/translate` doesn’t depend on it, so this is mostly cosmetic for the Translate button enable/disable.
6. **Camera backend list.** `cv2.CAP_DSHOW` is a Windows-only backend. On your Linux box only the `CAP_ANY` fallback ever does anything. Harmless but misleading; on Linux you’d use `cv2.CAP_V4L2`.
7. **Manual `.env` parser.** Lines 20–27 hand-roll dotenv parsing. Fine, but `python-dotenv` (or `uv`’s native `dotenv` support) is one line and handles edge cases.
8. **Gemini prompt is doing a lot of work.** The list of ASL signs in the prompt overlaps and is sometimes wrong (e.g. “Five / Hello / Wave” conflates three different gestures). Recognition quality will live or die by this prompt.
9. **No timeout on Gemini call.** A slow API response will hang the request thread. Consider `request_options={"timeout": 10}` or similar.
10. **Frontend is one 600+ line HTML file.** Splitting CSS/JS out is the obvious cleanup; the empty `static/css/` and `static/js/` folders show that was the original intent.
11. **README.md is empty.** This file (`PROJECT_GUIDE.md`) is the substitute for now; once you’re comfortable, fold a short version into `README.md`.

---

## 7. Mental model in one paragraph

A Flask server holds a single OpenCV `VideoCapture` behind a lock. One generator thread continuously reads frames, runs MediaPipe on them, draws landmarks, and streams JPEGs to the browser as MJPEG. When the user clicks Translate, the request handler grabs a few extra frames itself, picks the sharpest, asks MediaPipe which fingers are up, then sends the image + that text description to Gemini and asks for a one-phrase answer. Gemini’s answer is piped through gTTS into an MP3 saved under `static/audio/`, and both the text and the MP3 URL are returned as JSON. The frontend (a single inline-everything `index.html`) shows the text and plays the MP3.

That’s the whole project.

---

## 8. Recent Migrations (what changed and why)

Three things changed since the original walkthrough above. The older sections still describe how the app *behaves*, but the wiring underneath has been modernised. Here’s the plain-English version of each migration.

### 8.1 Frontend split — CSS and JS pulled out of the HTML

**Why:** Section §3 noted that `templates/index.html` was one 600+ line file with inline `<style>` and `<script>` blocks. That’s fine to ship but painful to read, diff, and cache. The empty `static/css/` and `static/js/` folders showed someone always meant to split them out.

**Where:**
- `templates/index.html` → trimmed from ~605 lines down to ~104. Inline `<style>…</style>` and `<script>…</script>` blocks are gone.
- `static/css/style.css` → **new file**, contains every CSS rule from the old `<style>` block, byte-for-byte.
- `static/js/app.js` → **new file**, contains every function from the old `<script>` block.

**What:** The HTML now just links to them with Flask’s `url_for`:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}"/>
...
<script src="{{ url_for('static', filename='js/app.js') }}"></script>
```

No behaviour change — the page looks and acts identically. The browser can now cache CSS and JS separately, and you can edit one without scrolling past the other.

### 8.2 Gemini SDK migration — `google-generativeai` → `google-genai`

**Why:** The old `google-generativeai` package is **deprecated and being removed**. Google replaced it with a brand-new SDK called `google-genai` that talks to the same API but uses a cleaner, client-based shape (matches their other Cloud SDKs). Pinning the dead one would break in the near future.

**Where:** `app.py` only. `pyproject.toml` will pick up the new package the next time you run `uv add google-genai` (or you can swap the line manually).

**What changed in `app.py`:**

1. **Imports** (top of file):
   ```python
   # OLD
   import google.generativeai as genai

   # NEW
   from google import genai
   from google.genai import types
   ```
   Also dropped `from PIL import Image` because we no longer convert frames into PIL images — the new SDK takes raw JPEG bytes directly.

2. **Client setup** (the `GEMINI CLIENT` banner section):
   ```python
   # OLD
   genai.configure(api_key=API_KEY)
   model = genai.GenerativeModel("gemini-2.0-flash")

   # NEW
   client     = genai.Client(api_key=API_KEY)
   MODEL_NAME = "gemini-2.0-flash"
   ```
   The old SDK had a global `configure()` + a `GenerativeModel` object. The new SDK has one `Client` object and you pass the model name on each call — much closer to how OpenAI / Anthropic SDKs work.

3. **The actual call** (inside `interpret_gesture()`):
   ```python
   # OLD — pass PIL Images directly
   response = model.generate_content([prompt, full_pil, crop_pil])

   # NEW — encode to JPEG bytes, wrap as Parts
   contents = [prompt]
   if full_bytes:
       contents.append(types.Part.from_bytes(data=full_bytes, mime_type="image/jpeg"))
   if crop_bytes:
       contents.append(types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"))

   response = client.models.generate_content(model=MODEL_NAME, contents=contents)
   ```
   Images are now `types.Part.from_bytes(data=..., mime_type="image/jpeg")`. We use `cv2.imencode(".jpg", frame)` to turn the OpenCV frame into JPEG bytes (one less library — Pillow is no longer needed for this path).

The output extraction (`response.text` with a fallback to `response.candidates[0].content.parts[0].text`) is the same shape, so downstream code didn’t change.

### 8.3 MediaPipe migration — `mp.solutions.hands` → Tasks API (`HandLandmarker`)

**Why:** The old `mp.solutions.hands` / `mp.solutions.drawing_utils` / `mp.solutions.drawing_styles` modules were the “legacy” path. In recent MediaPipe versions (the one you have, 0.10.35) **`mp.solutions` simply doesn’t exist anymore** — you can verify with `python -c "import mediapipe as mp; print(hasattr(mp,'solutions'))"` → `False`. Google replaced it with the **Tasks API**, which is a unified surface for all their on-device ML models (hands, face, pose, gestures, text, audio…). Same model under the hood, different (and more flexible) Python interface.

**Where:**
- `app.py` — imports, the `MEDIAPIPE SETUP` section, `process_with_mediapipe()`, and `get_finger_description()`.
- `models/hand_landmarker.task` — **new file (~7.5 MB)**, the hand-tracking model weights. The Tasks API does **not** ship the model bundled inside the pip package; you have to download the `.task` file separately and point the loader at it. Source: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.

**What changed in `app.py`:**

1. **Imports**:
   ```python
   # NEW (added)
   from mediapipe.tasks import python as mp_python
   from mediapipe.tasks.python import vision as mp_vision
   ```
   `import mediapipe as mp` stays — we still need `mp.Image` and `mp.ImageFormat`.

2. **Detector setup** — we now build **two** detectors with different *running modes*:

   ```python
   _base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

   # Streaming detector — used by the live MJPEG generator
   hands_detector = mp_vision.HandLandmarker.create_from_options(
       mp_vision.HandLandmarkerOptions(
           base_options=_base_opts,
           running_mode=mp_vision.RunningMode.VIDEO,
           num_hands=2,
           min_hand_detection_confidence=0.65,
           min_tracking_confidence=0.5,
       )
   )

   # Static detector — used for the snapshot we send to Gemini
   hands_static = mp_vision.HandLandmarker.create_from_options(
       mp_vision.HandLandmarkerOptions(
           base_options=_base_opts,
           running_mode=mp_vision.RunningMode.IMAGE,
           num_hands=2,
           min_hand_detection_confidence=0.5,
       )
   )
   ```
   Two modes matter:
   - `VIDEO` mode (`detect_for_video(image, timestamp_ms)`) is for streaming. **It demands strictly increasing timestamps** — we keep `_last_video_ts` under a `threading.Lock` and bump it by at least 1 ms each call.
   - `IMAGE` mode (`detect(image)`) is for one-off frames; no timestamp needed.

3. **Drawing helper** — `mp.solutions.drawing_utils.draw_landmarks(...)` is gone, so we draw the 21-landmark skeleton ourselves:

   ```python
   HAND_CONNECTIONS = [
       (0,1),(1,2),(2,3),(3,4),     # thumb
       (0,5),(5,6),(6,7),(7,8),     # index
       (5,9),(9,10),(10,11),(11,12),# middle
       (9,13),(13,14),(14,15),(15,16),  # ring
       (13,17),(17,18),(18,19),(19,20), # pinky
       (0,17),                      # palm base
   ]

   def _draw_hand(frame, landmarks):
       # connect dots with lines, then draw circles on each landmark
       ...
   ```
   Same skeleton, just our own ~10-line function instead of MediaPipe’s helper.

4. **`process_with_mediapipe()` — the live frame path:**
   ```python
   # OLD
   results = hands_detector.process(rgb)
   if results.multi_hand_landmarks:
       for lms in results.multi_hand_landmarks:
           mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS, ...)

   # NEW
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
   with _video_ts_lock:
       ts_ms = max(int(time.monotonic() * 1000), _last_video_ts + 1)
       _last_video_ts = ts_ms
   result = hands_detector.detect_for_video(mp_image, ts_ms)
   for lms in result.hand_landmarks:
       _draw_hand(frame, lms)
   ```
   Three differences worth knowing:
   - The frame is now wrapped as `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)` instead of being passed as a raw NumPy array.
   - We pass a monotonic timestamp.
   - The result attribute is `hand_landmarks` (list of hands), not `multi_hand_landmarks`.

5. **`get_finger_description()` — the snapshot path:**
   ```python
   # OLD
   results = hands_static.process(rgb)
   for hand_landmarks in results.multi_hand_landmarks:
       lm = hand_landmarks.landmark   # had a .landmark attribute

   # NEW
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
   result = hands_static.detect(mp_image)
   for hand_landmarks in result.hand_landmarks:
       lm = hand_landmarks            # the list of 21 landmarks IS hand_landmarks
   ```
   Subtle but important: in the legacy API, `hand_landmarks.landmark` was a list of points. In the Tasks API, `hand_landmarks` *is* that list directly — there’s no `.landmark` wrapper anymore. The finger-up logic (comparing `lm[8].y < lm[5].y` etc.) is unchanged.

**Net effect:** the app behaves the same — landmarks still draw on the live feed, fingers are still detected the same way — but it now runs against the supported, modern MediaPipe API and will keep working as the package continues to evolve.

### 8.4 New file you should know about

```
SignSpeak/
└── models/
    └── hand_landmarker.task   ← 7.5 MB, REQUIRED for the app to start
```

If this file is missing, `HandLandmarker.create_from_options(...)` will throw on import and the app will fail before serving its first request. It’s essentially a runtime dependency, just delivered as a file rather than a Python package. If you ever set up a fresh clone, re-download it from the URL noted in §8.3.

