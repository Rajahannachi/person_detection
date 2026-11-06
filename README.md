# Person Detection – yolo26n

Person and object detection with **yolo26n** (Ultralytics) on videos or images, with a FastAPI web interface. Two scripts also measure CPU performance.

## Installation

```bash
pip install -r requirements.txt
```

The model `models/yolo26n.pt` is downloaded automatically on first run.

## Web interface

```bash
python app.py
```

Open **http://127.0.0.1:8000**.

- **Video**: pick or upload a video, then start the detection. The result is shown live and the annotated video is saved to `output_videos/`.
- **Image**: upload an image and get the annotated image with the list of detections.

## Scripts (terminal)

Videos to process go in `input_videos/`.

| Command | Purpose | Output |
|---|---|---|
| `python detect.py` | runs detection on every video | `output_videos/` |
| `python stress_test.py` | processes all videos at the same time (1 process per video) | `output_videos/`, `reports/` |
| `python benchmark.py` | measures how many videos can run at the same time | `reports/` |

Useful options:

```bash
python detect.py --persons-only --conf 0.4

python stress_test.py --streams 8                  # simulate 8 cameras
python stress_test.py --duration 120 --no-save     # stop after 2 min, no output videos

python benchmark.py --target-fps 5
```

Run any script with `--help` to see all options.

> Each stress test and benchmark process uses about 440 MB of RAM.
