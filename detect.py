"""Simple detection with yolo26n: process every video in input_videos/ and save annotated videos to output_videos/.

Usage:
    python detect.py
    python detect.py --persons-only --conf 0.4
"""
import argparse
import time
from pathlib import Path

from common import MODEL_NAME, PERSON_CLASS_ID, ROOT, list_videos, load_model, process_video


def main():
    ap = argparse.ArgumentParser(description="yolo26n person/object detection on a folder of videos")
    ap.add_argument("--input", default=ROOT / "input_videos", type=Path)
    ap.add_argument("--output", default=ROOT / "output_videos", type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--persons-only", action="store_true", help="detect only persons (class 0)")
    args = ap.parse_args()

    videos = list_videos(args.input)
    if not videos:
        print(f"No videos found in {args.input}. Put .mp4/.avi/... files there.")
        return

    model = load_model()
    classes = [PERSON_CLASS_ID] if args.persons_only else None

    print(f"Model: {MODEL_NAME} | {len(videos)} video(s)")
    t0 = time.perf_counter()
    for v in videos:
        out = args.output / f"{v.stem}_detected.mp4"
        print(f"\n> {v.name} -> {out.name}")
        stats = process_video(model, v, out, conf=args.conf, imgsz=args.imgsz, classes=classes)
        print(f"  done: {stats['frames']} frames, {stats['fps']} FPS, "
              f"max {stats['max_persons_in_frame']} persons/frame")
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s. Results in {args.output}")


if __name__ == "__main__":
    main()
