"""CPU stress test with yolo26n.

- Reads the number of CPU cores (logical + physical).
- Starts ONE process per core, pins each process to its own core (CPU affinity),
  limits it to 1 thread, and runs yolo26n inference in parallel on all cores.
- Reads videos from input_videos/, writes annotated videos (boxes) to output_videos/.
- Monitors per-core CPU usage + RAM live, and saves a JSON/CSV report to reports/.

Modes (a video is NEVER cut into pieces: one worker always processes a whole video):
  parallel   : (default) 1 worker per video, ALL videos start at the same time.
               Workers are spread over the cores (2 videos per core if videos > cores).
  split      : N workers (= cores) take videos from a queue; extra videos wait their turn.
  replicate  : every worker processes every video -> max load, good for benchmarking.

Examples:
  python stress_test.py                              # all videos at the same time (1 worker / video)
  python stress_test.py --mode split                 # 1 worker per core, videos queued
  python stress_test.py --mode replicate             # every core runs every video
  python stress_test.py --mode replicate --duration 120 --no-save   # pure 2-min stress
  python stress_test.py --workers 2 --persons-only
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import queue
import time
from datetime import datetime
from pathlib import Path

import psutil

from common import MODEL_NAME, PERSON_CLASS_ID, ROOT, list_videos, load_model


# ----------------------------------------------------------------------------- worker
def worker(worker_id: int, core_id: int | None, args: dict, jobs, results, start_event, ready_queue):
    # Limit threads BEFORE importing torch so each process really uses a single core
    n = str(args["threads"])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = n

    if core_id is not None:
        try:
            psutil.Process().cpu_affinity([core_id])
        except Exception as e:  # not supported on some OS (e.g. macOS)
            print(f"[W{worker_id}] cannot pin to core {core_id}: {e}", flush=True)

    import cv2
    import numpy as np
    import torch

    from common import process_video

    torch.set_num_threads(args["threads"])
    cv2.setNumThreads(1)

    model = load_model()
    # warm-up so model init is not counted in the benchmark
    model.predict(np.zeros((args["imgsz"], args["imgsz"], 3), np.uint8), imgsz=args["imgsz"],
                  device="cpu", verbose=False)

    tag = f"W{worker_id}" + (f"/core{core_id}" if core_id is not None else "")
    ready_queue.put(worker_id)
    start_event.wait()

    deadline = time.perf_counter() + args["duration"] if args["duration"] else None
    output_dir = Path(args["output"])
    my_stats = []
    seen = set()

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        try:
            # get_nowait() is unreliable on mp.Queue (can return Empty while items exist)
            item = jobs.get(timeout=1.0)
        except queue.Empty:
            break
        # a job = (video path, stream name). Duplicated videos get different names (cam1, cam2 ...)
        video, label = Path(item[0]), item[1]

        out_path = None
        if args["save"] and label not in seen:  # save only first pass when looping
            suffix = "detected" if args["mode"] != "replicate" else f"core{core_id if core_id is not None else worker_id}"
            out_path = output_dir / f"{label}_{suffix}.mp4"
        seen.add(label)

        job_tag = f"{label} | {tag}"
        print(f"[{tag}] start {label} ({video.name})", flush=True)
        try:
            st = process_video(model, video, out_path, conf=args["conf"], imgsz=args["imgsz"],
                               classes=args["classes"], deadline=deadline, tag=job_tag,
                               progress_every=args["progress_every"])
        except Exception as e:
            print(f"[{tag}] ERROR on {label}: {e}", flush=True)
            continue
        st.update(worker=worker_id, core=core_id, stream=label)
        my_stats.append(st)
        print(f"[{tag}] done {label}: {st['frames']} frames @ {st['fps']} FPS", flush=True)

        # in duration mode keep looping over the same video(s)
        if deadline is not None and args["mode"] == "replicate":
            jobs.put(item)

    results.put({"worker": worker_id, "core": core_id, "runs": my_stats})


# ----------------------------------------------------------------------------- helpers
def cpu_info() -> dict:
    freq = psutil.cpu_freq()
    return {
        "processor": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "max_freq_mhz": round(freq.max, 0) if freq else None,
        "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
    }


def bar(pct: float, width: int = 10) -> str:
    filled = int(round(pct / 100 * width))
    return "#" * filled + "." * (width - filled)


# ----------------------------------------------------------------------------- main
def main():
    info = cpu_info()
    ap = argparse.ArgumentParser(description="yolo26n multi-core CPU stress test")
    ap.add_argument("--input", default=ROOT / "input_videos", type=Path)
    ap.add_argument("--output", default=ROOT / "output_videos", type=Path)
    ap.add_argument("--reports", default=ROOT / "reports", type=Path)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default: parallel = 1 per video, split/replicate = 1 per core)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--mode", choices=["parallel", "split", "replicate"], default="parallel",
                    help="parallel = all videos at the same time, 1 worker per video (default)")
    ap.add_argument("--streams", type=int, default=0,
                    help="total number of videos to run, duplicating the input videos "
                         "(ex: 2 videos + --streams 6 -> each video 3 times, named _cam1.._cam6). 0 = no duplication")
    ap.add_argument("--no-mosaic", action="store_true",
                    help="don't build the mosaic video (all outputs side by side) at the end")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N seconds (0 = process videos to the end)")
    ap.add_argument("--no-save", action="store_true", help="don't write output videos (pure benchmark)")
    ap.add_argument("--no-pin", action="store_true", help="don't pin workers to specific cores")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--persons-only", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    a = ap.parse_args()

    print("=" * 70)
    print(" yolo26n CPU STRESS TEST")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k:<15}: {v}")

    videos = list_videos(a.input)
    if not videos:
        print(f"\nNo videos in {a.input}. Add some .mp4/.avi files and retry.")
        return

    # jobs = (video path, stream name). --streams N duplicates the videos (round-robin) to get N streams
    if a.streams > 0:
        items = [(str(videos[i % len(videos)]), f"{videos[i % len(videos)].stem}_cam{i + 1}")
                 for i in range(a.streams)]
    else:
        items = [(str(v), v.stem) for v in videos]

    if a.mode == "parallel":
        if a.workers and a.workers < len(items):
            print(f"\n  NOTE: mode parallel -> only the first {a.workers} videos are used (--workers {a.workers})")
            items = items[:a.workers]
        n_workers = len(items)
    else:
        n_workers = max(1, a.workers or info["logical_cores"])
    pin = not a.no_pin
    cores = list(range(info["logical_cores"]))
    print(f"\n  input videos   : {len(videos)} -> {', '.join(v.name for v in videos)}")
    if a.streams > 0:
        print(f"  streams        : {len(items)} (duplicated) -> {', '.join(label for _, label in items)}")
    print(f"  workers        : {n_workers} x {a.threads} thread(s), pinned={pin}, mode={a.mode}")
    if n_workers > len(cores):
        print(f"  NOTE           : {n_workers} workers > {len(cores)} cores -> some cores run 2+ videos "
              f"(each one slower). Expect ~{440 * n_workers / 1024:.1f} GB of RAM.")
    print(f"  model          : {MODEL_NAME}  imgsz={a.imgsz} conf={a.conf}")
    if a.duration:
        print(f"  duration       : {a.duration}s")

    print("\nPreparing model ...")
    load_model()  # download once here, not in every worker at the same time

    ctx = mp.get_context("spawn")  # same behaviour on Windows / Linux
    results = ctx.Queue()
    ready = ctx.Queue()
    start_event = ctx.Event()

    # job queues: parallel = 1 video per worker, split = one shared queue, replicate = all videos per worker
    if a.mode == "parallel":
        job_queues = []
        for it in items:
            q = ctx.Queue()
            q.put(it)
            job_queues.append(q)
    elif a.mode == "split":
        shared = ctx.Queue()
        for it in items:
            shared.put(it)
        job_queues = [shared] * n_workers
    else:
        job_queues = []
        for _ in range(n_workers):
            q = ctx.Queue()
            for it in items:
                q.put(it)
            job_queues.append(q)

    args = {
        "threads": a.threads, "imgsz": a.imgsz, "conf": a.conf,
        "classes": [PERSON_CLASS_ID] if a.persons_only else None, "duration": a.duration,
        "output": str(a.output), "save": not a.no_save, "mode": a.mode,
        "progress_every": a.progress_every,
    }

    procs = []
    for i in range(n_workers):
        core = cores[i % len(cores)] if pin else None
        p = ctx.Process(target=worker, args=(i, core, args, job_queues[i], results, start_event, ready),
                        name=f"yolo-worker-{i}")
        p.start()
        procs.append(p)

    print(f"Loading model in {n_workers} workers ...")
    for _ in range(n_workers):
        ready.get()
    print("All workers ready -> START\n")

    # ---- monitoring loop
    psutil.cpu_percent(percpu=True)  # prime
    samples: list[list[float]] = []
    ram_samples: list[float] = []
    collected = []
    t0 = time.perf_counter()
    start_event.set()

    while any(p.is_alive() for p in procs) or len(collected) < n_workers:
        per_core = psutil.cpu_percent(interval=1.0, percpu=True)
        samples.append(per_core)
        ram_samples.append(psutil.virtual_memory().percent)
        while True:
            try:
                collected.append(results.get_nowait())
            except queue.Empty:
                break
        elapsed = time.perf_counter() - t0
        cores_str = " ".join(f"C{i}:{c:5.1f}%" for i, c in enumerate(per_core))
        print(f"[MON {elapsed:6.1f}s] CPU {sum(per_core) / len(per_core):5.1f}% | "
              f"RAM {ram_samples[-1]:4.1f}% | {cores_str}", flush=True)
        if not any(p.is_alive() for p in procs) and len(collected) < n_workers:
            try:
                collected.append(results.get(timeout=2))
            except queue.Empty:
                break  # a worker crashed without reporting

    for p in procs:
        p.join()
    total_time = time.perf_counter() - t0

    # ---- report
    runs = [r for w in collected for r in w["runs"]]
    total_frames = sum(r["frames"] for r in runs)
    n_samples = max(1, len(samples))
    core_avg = [sum(s[i] for s in samples) / n_samples for i in range(len(samples[0]))] if samples else []
    core_max = [max(s[i] for s in samples) for i in range(len(samples[0]))] if samples else []

    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    print(f"{'worker':>6} {'core':>5} {'videos':>6} {'frames':>7} {'FPS':>7} {'ms/frame':>9}  video(s)")
    per_worker = []
    for w in sorted(collected, key=lambda x: x["worker"]):
        fr = sum(r["frames"] for r in w["runs"])
        wall = sum(r["wall_time_s"] for r in w["runs"])
        inf = sum(r["infer_time_s"] for r in w["runs"])
        fps = fr / wall if wall else 0
        ms = inf / fr * 1000 if fr else 0
        names = ", ".join(sorted({r.get("stream", r["video"]) for r in w["runs"]}))
        per_worker.append({"worker": w["worker"], "core": w["core"], "videos": len(w["runs"]),
                           "video_names": names, "frames": fr, "fps": round(fps, 2), "avg_infer_ms": round(ms, 2)})
        print(f"{w['worker']:>6} {str(w['core']):>5} {len(w['runs']):>6} {fr:>7} {fps:>7.2f} {ms:>9.1f}  {names}")

    print("\nCPU usage per core (avg / max):")
    for i, (av, mx) in enumerate(zip(core_avg, core_max)):
        print(f"  core {i:>2}: [{bar(av)}] avg {av:5.1f}%  max {mx:5.1f}%")

    agg_fps = total_frames / total_time if total_time else 0
    print(f"\nTotal frames      : {total_frames}")
    print(f"Total time        : {total_time:.1f} s")
    print(f"Aggregate FPS     : {agg_fps:.2f}  (all cores together)")
    print(f"Avg CPU usage     : {sum(core_avg) / max(1, len(core_avg)):.1f}%")
    print(f"Peak RAM usage    : {max(ram_samples, default=0):.1f}%")
    if len(collected) < n_workers:
        print(f"WARNING: only {len(collected)}/{n_workers} workers reported (crash?)")

    a.reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": stamp, "model": MODEL_NAME, "cpu": info,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "summary": {"total_frames": total_frames, "total_time_s": round(total_time, 2),
                    "aggregate_fps": round(agg_fps, 2),
                    "avg_cpu_percent": round(sum(core_avg) / max(1, len(core_avg)), 1),
                    "peak_ram_percent": max(ram_samples, default=0)},
        "per_core_cpu": [{"core": i, "avg": round(av, 1), "max": mx} for i, (av, mx) in enumerate(zip(core_avg, core_max))],
        "per_worker": per_worker,
        "runs": runs,
    }
    json_path = a.reports / f"stress_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = a.reports / f"stress_{stamp}.csv"
    if runs:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
            wr.writeheader()
            wr.writerows(runs)
    print(f"\nReport: {json_path}")
    if not a.no_save:
        print(f"Videos: {a.output}")
        outputs = sorted({Path(r["output"]) for r in runs if r.get("output") and Path(r["output"]).exists()})
        if len(outputs) > 1 and not a.no_mosaic:
            from common import make_mosaic
            mosaic = a.output / f"mosaic_{stamp}.mp4"
            print(f"Building mosaic of {len(outputs)} videos ...", flush=True)
            make_mosaic(outputs, mosaic)
            print(f"Mosaic: {mosaic}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
