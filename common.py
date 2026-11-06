"""Shared helpers: the yolo26n model, video listing, drawing boxes, processing a video."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "yolo26n"
MODEL_PATH = ROOT / "models" / f"{MODEL_NAME}.pt"  # downloaded automatically on first use

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg"}
PERSON_CLASS_ID = 0  # COCO class 0 = person

# BGR colors: person always green, other classes get a stable (non-green) color
PERSON_COLOR = (0, 220, 0)
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (0, 0, 255),
    (255, 0, 255), (0, 128, 255), (203, 56, 255), (0, 212, 255), (128, 0, 128),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (180, 105, 255), (255, 149, 200), (255, 55, 199),
]


def load_model():
    """Load yolo26n (downloads it to models/ the first time)."""
    from ultralytics import YOLO
    MODEL_PATH.parent.mkdir(exist_ok=True)
    return YOLO(str(MODEL_PATH))


def color_for(cls_id: int) -> tuple[int, int, int]:
    if cls_id == PERSON_CLASS_ID:
        return PERSON_COLOR
    return _PALETTE[cls_id % len(_PALETTE)]


def list_videos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def draw_detections(frame: np.ndarray, result, names: dict) -> tuple[int, int]:
    """Draw boxes + labels on frame. Returns (person_count, total_objects)."""
    persons = 0
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    thickness = max(1, round(sum(frame.shape[:2]) / 600))
    font_scale = thickness / 3

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
        if cls_id == PERSON_CLASS_ID:
            persons += 1
        color = color_for(cls_id)
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
        y_text = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 4
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + base - 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
    return persons, len(clss)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Small info panel in the top-left corner."""
    scale = max(0.5, frame.shape[0] / 1000)
    th = int(28 * scale)
    w = int(max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for l in lines)) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, th * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, th * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)


def process_video(model, in_path: Path, out_path: Path | None, *, conf: float = 0.25,
                  imgsz: int = 640, classes: list[int] | None = None, deadline: float | None = None,
                  tag: str = "", progress_every: int = 100, on_frame=None) -> dict:
    """Run YOLO on every frame of `in_path`. Writes annotated video to `out_path` (None = don't save).

    Stops early if time.perf_counter() exceeds `deadline`.
    `on_frame(frame, info)` is called for every annotated frame; return False to stop.
    """
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    writer = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    names = model.names
    frames = 0
    infer_time = 0.0
    total_persons = 0
    total_objects = 0
    max_persons = 0
    t_start = time.perf_counter()

    try:
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            result = model.predict(frame, conf=conf, imgsz=imgsz, classes=classes,
                                   device="cpu", verbose=False)[0]
            dt = time.perf_counter() - t0
            infer_time += dt
            frames += 1

            persons, objects = draw_detections(frame, result, names)
            total_persons += persons
            total_objects += objects
            max_persons = max(max_persons, persons)

            if writer is not None or on_frame is not None:
                hud = [f"Persons: {persons}", f"Objects: {objects}", f"Infer: {dt * 1000:.0f} ms ({1 / dt:.1f} FPS)"]
                if tag:
                    hud.append(tag)
                draw_hud(frame, hud)
            if writer is not None:
                writer.write(frame)
            if on_frame is not None:
                info = {"frame": frames, "total": total, "persons": persons, "objects": objects,
                        "infer_ms": dt * 1000, "avg_fps": frames / infer_time}
                if on_frame(frame, info) is False:
                    break

            if progress_every and frames % progress_every == 0:
                pct = f"{frames / total * 100:5.1f}%" if total > 0 else ""
                print(f"  {tag} {in_path.name}: {frames}/{total} {pct} | "
                      f"{frames / infer_time:.1f} FPS", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    wall = time.perf_counter() - t_start
    return {
        "video": in_path.name,
        "output": str(out_path) if out_path else None,
        "frames": frames,
        "wall_time_s": round(wall, 3),
        "infer_time_s": round(infer_time, 3),
        "fps": round(frames / wall, 2) if wall else 0.0,
        "avg_infer_ms": round(infer_time / frames * 1000, 2) if frames else 0.0,
        "total_person_detections": total_persons,
        "total_object_detections": total_objects,
        "max_persons_in_frame": max_persons,
    }


def make_mosaic(paths: list[Path], out_path: Path, tile_w: int = 640, tile_h: int = 360) -> Path:
    """Combine several (annotated) videos into one grid video, all playing at the same time."""
    import math

    caps = [cv2.VideoCapture(str(p)) for p in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    n = len(caps)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cols * tile_w, rows * tile_h))
    names = [Path(p).stem.replace("_detected", "") for p in paths]
    alive = [True] * n
    try:
        while any(alive):
            canvas = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
            for i, cap in enumerate(caps):
                r, c = divmod(i, cols)
                x0, y0 = c * tile_w, r * tile_h
                frame = None
                if alive[i]:
                    ok, frame = cap.read()
                    if not ok:
                        alive[i] = False
                        frame = None
                if frame is not None:
                    h, w = frame.shape[:2]
                    s = min(tile_w / w, tile_h / h)
                    fw, fh = int(w * s), int(h * s)
                    ox, oy = x0 + (tile_w - fw) // 2, y0 + (tile_h - fh) // 2
                    canvas[oy:oy + fh, ox:ox + fw] = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
                else:
                    cv2.putText(canvas, "termine", (x0 + tile_w // 2 - 50, y0 + tile_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
                # stream name at the bottom of each tile + thin border
                cv2.rectangle(canvas, (x0, y0 + tile_h - 26), (x0 + tile_w, y0 + tile_h), (0, 0, 0), -1)
                cv2.putText(canvas, names[i], (x0 + 8, y0 + tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (60, 60, 60), 1)
            if any(alive):
                writer.write(canvas)
    finally:
        for cap in caps:
            cap.release()
        writer.release()
    return out_path
