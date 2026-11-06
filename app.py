"""Web interface (FastAPI) to detect people and objects with yolo26n, on a video or an image.

Run:  python app.py      then open http://127.0.0.1:8000

The stress test and the benchmark are terminal scripts
(python stress_test.py / python benchmark.py).
"""
from __future__ import annotations

import asyncio
import base64
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from common import (MODEL_NAME, PERSON_CLASS_ID, ROOT, VIDEO_EXTS, draw_detections, draw_hud,
                    list_videos, load_model, process_video)

INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"
STATIC_DIR = ROOT / "static"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STREAM_MAX_W = 960  # live preview width (the saved video keeps the original size)

app = FastAPI(title=f"{MODEL_NAME} – Person detection")


# ----------------------------------------------------------------------------- models
class _Models:
    """Two instances: one for the video thread, one for image requests (predict is not thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._video = None
        self._image = None
        self.image_lock = threading.Lock()

    def video(self):
        with self._lock:
            if self._video is None:
                self._video = load_model()
            return self._video

    def image(self):
        with self._lock:
            if self._image is None:
                self._image = load_model()
            return self._image


models = _Models()


# ----------------------------------------------------------------------------- video detection state
class VideoJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.running = False
        self.status = "Ready."
        self.info: dict = {}
        self.output: str | None = None
        self.jpeg: bytes | None = None
        self.frame_id = 0

    def snapshot(self) -> dict:
        with self.lock:
            return {"running": self.running, "status": self.status, "info": self.info,
                    "output": self.output, "frame_id": self.frame_id}


job = VideoJob()


class VideoRequest(BaseModel):
    video: str
    conf: float = Field(0.25, ge=0.01, le=0.99)
    imgsz: int = Field(640, ge=160, le=1920)
    persons_only: bool = False
    save: bool = True


def _find_video(name: str) -> Path:
    for v in list_videos(INPUT_DIR):
        if v.name == name:
            return v
    raise HTTPException(404, f"Video not found: {name}")


def _video_worker(video: Path, req: VideoRequest):
    out = OUTPUT_DIR / f"{video.stem}_detected.mp4" if req.save else None
    try:
        with job.lock:
            job.status = f"Loading model {MODEL_NAME} …"
        model = models.video()
        with job.lock:
            job.status = f"Detecting on {video.name} …"

        def on_frame(frame, info):
            h, w = frame.shape[:2]
            if w > STREAM_MAX_W:
                frame = cv2.resize(frame, (STREAM_MAX_W, int(h * STREAM_MAX_W / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with job.lock:
                if ok:
                    job.jpeg = buf.tobytes()
                    job.frame_id += 1
                job.info = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in info.items()}
            return not job.stop_flag.is_set()

        st = process_video(model, video, out, conf=req.conf, imgsz=req.imgsz,
                           classes=[PERSON_CLASS_ID] if req.persons_only else None,
                           progress_every=0, on_frame=on_frame)
        msg = f"Done: {st['frames']} frames, {st['fps']} FPS, max {st['max_persons_in_frame']} persons/frame"
        if job.stop_flag.is_set():
            msg = "Stopped. " + msg
        with job.lock:
            job.status = msg
            job.output = Path(st["output"]).name if st["output"] else None
    except Exception as e:  # report any error to the UI
        with job.lock:
            job.status = f"Error: {e}"
    finally:
        with job.lock:
            job.running = False


# ----------------------------------------------------------------------------- routes: page + videos
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info():
    return {"model": MODEL_NAME}


@app.get("/api/videos")
def videos():
    return [{"name": v.name, "size_mb": round(v.stat().st_size / 1e6, 1)} for v in list_videos(INPUT_DIR)]


@app.post("/api/videos")
def upload_video(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, "Unsupported video format.")
    INPUT_DIR.mkdir(exist_ok=True)
    with open(INPUT_DIR / name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"name": name}


@app.get("/api/outputs/{name}")
def download_output(name: str):
    path = OUTPUT_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(path, filename=path.name)


# ----------------------------------------------------------------------------- routes: video detection
@app.post("/api/video/start")
def video_start(req: VideoRequest):
    video = _find_video(req.video)
    with job.lock:
        if job.running:
            raise HTTPException(409, "A detection is already running.")
        job.running = True
        job.stop_flag.clear()
        job.info, job.output, job.jpeg = {}, None, None
        job.status = "Starting …"
    job.thread = threading.Thread(target=_video_worker, args=(video, req), daemon=True)
    job.thread.start()
    return job.snapshot()


@app.post("/api/video/stop")
def video_stop():
    job.stop_flag.set()
    return job.snapshot()


@app.get("/api/video/status")
def video_status():
    return job.snapshot()


@app.get("/api/video/stream")
async def video_stream():
    """MJPEG stream of the annotated frames (use it as the src of an <img>)."""

    async def frames():
        last = -1
        while True:
            with job.lock:
                fid, jpeg, running = job.frame_id, job.jpeg, job.running
            if jpeg is not None and fid != last:
                last = fid
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            elif not running:
                break
            await asyncio.sleep(0.02)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ----------------------------------------------------------------------------- routes: image detection
@app.post("/api/image")
def detect_image(file: UploadFile = File(...), conf: float = Form(0.25), imgsz: int = Form(640),
                 persons_only: bool = Form(False)):
    if Path(file.filename or "").suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(400, "Unsupported image format.")
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot read the image.")

    model = models.image()
    with models.image_lock:
        result = model.predict(img, conf=conf, imgsz=imgsz, device="cpu", verbose=False,
                               classes=[PERSON_CLASS_ID] if persons_only else None)[0]
    infer_ms = sum(result.speed.values())

    detections = []
    if result.boxes is not None:
        for (x1, y1, x2, y2), c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                          result.boxes.cls.tolist()):
            detections.append({"label": model.names[int(k)], "conf": round(c, 3),
                               "box": [round(x1), round(y1), round(x2), round(y2)]})

    persons, objects = draw_detections(img, result, model.names)
    draw_hud(img, [f"Persons: {persons}", f"Objects: {objects}", f"Infer: {infer_ms:.0f} ms"])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {
        "persons": persons,
        "objects": objects,
        "infer_ms": round(infer_ms, 1),
        "width": img.shape[1],
        "height": img.shape[0],
        "detections": sorted(detections, key=lambda d: -d["conf"]),
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
    }


if __name__ == "__main__":
    print("Interface: http://127.0.0.1:8000   (API docs: http://127.0.0.1:8000/docs)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
