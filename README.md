# Webcam (simple GUI)

Live preview, **Take photo** (JPEG), **Start / Stop recording** (MP4), and a **microphone picker** with an **input level** bar to confirm the right device.

## Run

Double-click **`Run-Webcam.bat`**, or:

```bat
cd /d C:\Users\inque\OneDrive\Desktop\Work\tools\webcam
python -m pip install -r requirements.txt
python app.py
```

1. Click **Open** (change **Camera** to `1` or `2` if needed).
2. Choose a **Microphone**, click **Listen**, and talk — the **Input level** bar should move.
3. **Take photo** — saves to `captures\photo_YYYYMMDD_HHMMSS.jpg`.
4. **Start recording** / **Stop recording** — saves `captures\video_*.mp4` with **video and the selected microphone** (requires **ffmpeg** on PATH).

Pick the correct mic in the dropdown before recording; the level bar updates while recording.

Close the window or click **Close** to release the camera.
