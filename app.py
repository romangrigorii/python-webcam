"""Simple webcam GUI: live preview, photo capture, video + mic recording."""

from __future__ import annotations

import datetime as dt
import subprocess
import threading
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
import sounddevice as sd
from PIL import Image, ImageTk

AUDIO_RATE = 44100
VIDEO_FPS = 20.0


def input_devices() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            name = dev["name"].strip()
            out.append((index, f"{index}: {name}"))
    return out


def write_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())


def mux_video_audio(
    video: Path,
    audio: Path,
    output: Path,
    *,
    video_fps: float,
    video_frames: int,
    audio_trim_sec: float,
) -> None:
    if video_frames < 1:
        raise ValueError("no video frames recorded")

    video_dur = video_frames / video_fps
    trim = max(0.0, audio_trim_sec)
    # Drop audio recorded before the first video frame; fit length to video track.
    fc = (
        f"[1:a]atrim=start={trim:.6f},asetpts=PTS-STARTPTS,"
        f"atrim=duration={video_dur:.6f},apad=whole_dur={video_dur:.6f}[a]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-filter_complex",
        fc,
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_dur:.6f}",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


class WebcamApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Webcam")
        self.root.minsize(640, 560)

        self.output_dir = Path(__file__).resolve().parent / "captures"
        self.output_dir.mkdir(exist_ok=True)

        self.cap: cv2.VideoCapture | None = None
        self.preview_job: str | None = None
        self.recording = False
        self.writer: cv2.VideoWriter | None = None
        self.record_path: Path | None = None
        self.record_video_temp: Path | None = None
        self.record_audio_path: Path | None = None
        self.frame_size = (640, 480)
        self._photo: ImageTk.PhotoImage | None = None

        self.mic_devices = input_devices()
        self.mic_stream: sd.InputStream | None = None
        self.mic_monitoring = False
        self.mic_level_job: str | None = None
        self.mic_level = 0.0
        self.mic_peak = 0.0
        self.audio_chunks: list[np.ndarray] = []
        self.mic_for_recording = False
        self.video_frames_written = 0
        self.audio_start_time = 0.0
        self.audio_trim_sec = 0.0
        self.record_video_anchor = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        ttk.Combobox(
            top,
            textvariable=self.camera_var,
            values=("0", "1", "2"),
            width=4,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(top, text="Open", command=self.open_camera).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Close", command=self.close_camera).pack(side=tk.LEFT, padx=2)

        mic_row = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        mic_row.pack(fill=tk.X)

        ttk.Label(mic_row, text="Microphone:").pack(side=tk.LEFT)
        mic_labels = [label for _, label in self.mic_devices] or ["(no inputs found)"]
        default_mic = mic_labels[0] if self.mic_devices else ""
        self.mic_var = tk.StringVar(value=default_mic)
        self.mic_combo = ttk.Combobox(
            mic_row,
            textvariable=self.mic_var,
            values=mic_labels,
            width=48,
            state="readonly" if self.mic_devices else "disabled",
        )
        self.mic_combo.pack(side=tk.LEFT, padx=(4, 8), fill=tk.X, expand=True)

        self.mic_btn = ttk.Button(
            mic_row,
            text="Listen",
            command=self.toggle_mic_monitor,
            state=tk.NORMAL if self.mic_devices else tk.DISABLED,
        )
        self.mic_btn.pack(side=tk.LEFT, padx=2)

        level_row = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        level_row.pack(fill=tk.X)

        ttk.Label(level_row, text="Input level:").pack(side=tk.LEFT)
        self.level_bar = ttk.Progressbar(level_row, maximum=100, length=320, mode="determinate")
        self.level_bar.pack(side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True)
        self.level_text = tk.StringVar(value="—")
        ttk.Label(level_row, textvariable=self.level_text, width=8).pack(side=tk.LEFT)

        self.preview = ttk.Label(self.root, text="Open a camera to start", anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill=tk.X)

        self.photo_btn = ttk.Button(bar, text="Take photo", command=self.take_photo, state=tk.DISABLED)
        self.photo_btn.pack(side=tk.LEFT, padx=4)

        self.record_btn = ttk.Button(bar, text="Start recording", command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(value=f"Saves to: {self.output_dir}")
        ttk.Label(self.root, textvariable=self.status, padding=(8, 0)).pack(fill=tk.X)

    def _mic_device_index(self) -> int | None:
        label = self.mic_var.get()
        for index, text in self.mic_devices:
            if text == label:
                return index
        return self.mic_devices[0][0] if self.mic_devices else None

    def _update_level_from_audio(self, indata: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(np.square(indata))))
        level = min(1.0, rms * 12.0)
        self.mic_level = max(self.mic_level * 0.5, level)

    def _mic_monitor_callback(self, indata, _frames, _time, status) -> None:
        if status:
            return
        self._update_level_from_audio(indata)

    def _mic_record_callback(self, indata, _frames, _time, status) -> None:
        if status:
            return
        self.audio_chunks.append(indata.copy())
        self._update_level_from_audio(indata)

    def _close_mic_stream(self) -> None:
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None

    def toggle_mic_monitor(self) -> None:
        if self.recording:
            return
        if self.mic_monitoring:
            self.stop_mic_monitor()
        else:
            self.start_mic_monitor()

    def start_mic_monitor(self) -> None:
        device = self._mic_device_index()
        if device is None:
            messagebox.showerror("Microphone", "No input devices found.")
            return
        self.stop_mic_monitor(silent=True)
        try:
            self.mic_stream = sd.InputStream(
                device=device,
                channels=1,
                dtype="float32",
                samplerate=AUDIO_RATE,
                blocksize=1024,
                callback=self._mic_monitor_callback,
            )
            self.mic_stream.start()
        except Exception as exc:
            messagebox.showerror("Microphone", f"Could not open mic:\n{exc}")
            self.mic_stream = None
            return

        self.mic_monitoring = True
        self.mic_for_recording = False
        self.mic_btn.config(text="Stop listen")
        self.mic_combo.config(state="disabled")
        self._schedule_level_update()

    def stop_mic_monitor(self, silent: bool = False) -> None:
        if self.mic_for_recording:
            return
        self.mic_monitoring = False
        if self.mic_level_job:
            self.root.after_cancel(self.mic_level_job)
            self.mic_level_job = None
        self._close_mic_stream()
        self.mic_level = 0.0
        self.mic_peak = 0.0
        self.level_bar["value"] = 0
        self.level_text.set("—")
        self.mic_btn.config(text="Listen")
        if self.mic_devices and not self.recording:
            self.mic_combo.config(state="readonly")
        if not silent and not self.recording:
            self.status.set(f"Saves to: {self.output_dir}")

    def _start_record_audio(self) -> bool:
        device = self._mic_device_index()
        if device is None:
            return False
        self.stop_mic_monitor(silent=True)
        self.audio_chunks = []
        try:
            self.mic_stream = sd.InputStream(
                device=device,
                channels=1,
                dtype="float32",
                samplerate=AUDIO_RATE,
                blocksize=1024,
                callback=self._mic_record_callback,
            )
            self.mic_stream.start()
            self.audio_start_time = time.perf_counter()
        except Exception as exc:
            messagebox.showwarning(
                "Microphone",
                f"Recording video only — mic failed:\n{exc}",
            )
            self.mic_stream = None
            return False

        self.mic_for_recording = True
        self.mic_monitoring = True
        self.mic_btn.config(state=tk.DISABLED)
        self.mic_combo.config(state="disabled")
        self._schedule_level_update()
        return True

    def _stop_record_audio(self) -> Path | None:
        if self.mic_level_job:
            self.root.after_cancel(self.mic_level_job)
            self.mic_level_job = None
        self._close_mic_stream()
        self.mic_for_recording = False
        self.mic_monitoring = False
        self.mic_btn.config(state=tk.NORMAL, text="Listen")
        if self.mic_devices:
            self.mic_combo.config(state="readonly")

        if not self.audio_chunks or self.record_audio_path is None:
            self.audio_chunks = []
            return None

        audio = np.concatenate(self.audio_chunks, axis=0)
        self.audio_chunks = []
        write_wav(self.record_audio_path, audio, AUDIO_RATE)
        return self.record_audio_path

    def _schedule_level_update(self) -> None:
        if not self.mic_monitoring:
            return
        pct = int(round(self.mic_level * 100))
        self.level_bar["value"] = pct
        self.mic_peak = max(self.mic_peak * 0.98, self.mic_level)
        peak_pct = int(round(self.mic_peak * 100))
        self.level_text.set(f"{pct}%")
        if self.recording:
            self.status.set(f"Recording video + mic… peak ~{peak_pct}%")
        elif self.mic_level > 0.02:
            self.status.set(f"Mic input detected (peak ~{peak_pct}%)")
        self.mic_level *= 0.85
        self.mic_level_job = self.root.after(50, self._schedule_level_update)

    def open_camera(self) -> None:
        self.close_camera(silent=True)
        index = int(self.camera_var.get())
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Webcam", f"Could not open camera {index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (w, h)

        self.cap = cap
        self.photo_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.status.set(f"Camera {index} — {w}×{h}")
        self._schedule_preview()

    def close_camera(self, silent: bool = False) -> None:
        if self.recording:
            self._stop_recording()
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.photo_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        if not silent:
            self.preview.config(image="", text="Camera closed")
            self.status.set(f"Saves to: {self.output_dir}")

    def _schedule_preview(self) -> None:
        self.preview_job = self.root.after(15, self._update_preview)

    def _read_frame(self):
        if not self.cap:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def _update_preview(self) -> None:
        frame = self._read_frame()
        if frame is None:
            self.status.set("Camera lost — reopen.")
            self.close_camera(silent=True)
            return

        if self.recording and self.writer is not None:
            now = time.perf_counter()
            if self.video_frames_written == 0:
                self.record_video_anchor = now
                if self.audio_start_time > 0:
                    self.audio_trim_sec = max(0.0, now - self.audio_start_time)
                self.writer.write(frame)
                self.video_frames_written += 1
            else:
                target = self.record_video_anchor + self.video_frames_written / VIDEO_FPS
                if now >= target:
                    self.writer.write(frame)
                    self.video_frames_written += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((960, 720), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.preview.config(image=self._photo, text="")

        self.preview_job = self.root.after(15, self._update_preview)

    def take_photo(self) -> None:
        frame = self._read_frame()
        if frame is None:
            return
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"photo_{stamp}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.status.set(f"Photo saved: {path.name}")

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        w, h = self.frame_size
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_path = self.output_dir / f"video_{stamp}.mp4"
        self.record_video_temp = self.output_dir / f"video_{stamp}_video.mp4"
        self.record_audio_path = self.output_dir / f"video_{stamp}_audio.wav"
        self.video_frames_written = 0
        self.audio_trim_sec = 0.0
        self.record_video_anchor = 0.0

        has_audio = self._start_record_audio()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.record_video_temp), fourcc, VIDEO_FPS, (w, h))
        if not writer.isOpened():
            self._stop_record_audio()
            messagebox.showerror("Webcam", "Could not start video writer.")
            self.record_path = None
            self.record_video_temp = None
            self.record_audio_path = None
            return

        self.writer = writer
        self.recording = True
        self.record_btn.config(text="Stop recording")
        if has_audio:
            self.status.set(f"Recording video + mic → {self.record_path.name}")
        else:
            self.status.set(f"Recording video only → {self.record_path.name}")

    def _finish_mux(
        self,
        final: Path,
        video_temp: Path,
        audio_path: Path | None,
        video_frames: int,
        audio_trim_sec: float,
    ) -> None:
        try:
            if audio_path and audio_path.exists():
                mux_video_audio(
                    video_temp,
                    audio_path,
                    final,
                    video_fps=VIDEO_FPS,
                    video_frames=video_frames,
                    audio_trim_sec=audio_trim_sec,
                )
                video_temp.unlink(missing_ok=True)
                audio_path.unlink(missing_ok=True)
                msg = f"Saved (video + audio): {final.name}"
            else:
                video_temp.replace(final)
                msg = f"Saved (video only): {final.name}"
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"").decode(errors="replace")[-400:]
            video_temp.replace(final)
            msg = f"Saved video only (audio mux failed): {final.name}"
            self.root.after(
                0,
                lambda: messagebox.showwarning(
                    "Recording",
                    f"Could not merge audio.\n\n{err}",
                ),
            )
        self.root.after(0, lambda: self.status.set(msg))

    def _stop_recording(self) -> None:
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        self.record_btn.config(text="Start recording")

        final = self.record_path
        video_temp = self.record_video_temp
        video_frames = self.video_frames_written
        audio_trim = self.audio_trim_sec
        audio_path = self._stop_record_audio()
        self.level_bar["value"] = 0
        self.level_text.set("—")

        if not final or not video_temp or not video_temp.exists():
            self.status.set("Recording stopped.")
            self.record_path = None
            self.record_video_temp = None
            self.record_audio_path = None
            return

        self.status.set("Saving recording…")
        threading.Thread(
            target=self._finish_mux,
            args=(final, video_temp, audio_path, video_frames, audio_trim),
            daemon=True,
        ).start()

        self.record_path = None
        self.record_video_temp = None
        self.record_audio_path = None

    def on_close(self) -> None:
        if self.recording:
            self._stop_recording()
        self.stop_mic_monitor(silent=True)
        self.close_camera(silent=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    WebcamApp().run()


if __name__ == "__main__":
    main()
