"""yolo26n benchmark: how many videos (streams) can this PC process in parallel?

For N = 1, 2, 3 ... parallel streams (1 process per stream), measures for `--duration` seconds:
  - FPS per stream (average / minimum) and total FPS
  - inference latency (average, p95)
  - CPU (average / max, per core)
  - RAM: memory of the YOLO processes (total + per stream) and system RAM

Output:
  - max number of REAL-TIME streams (every stream FPS >= target FPS, default: the video FPS)
  - N giving the best total throughput
  - JSON + CSV report and PNG chart in reports/

Examples:
  python benchmark.py                                  # yolo26n, 640, N = 1..2x cores
  python benchmark.py --target-fps 5                   # "real time" = 5 FPS per stream
  python benchmark.py --imgsz 320 640                  # compare two image sizes
  python benchmark.py --streams 1 2 4 6 8 --duration 30
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import queue
import statistics
import time
from datetime import datetime
from pathlib import Path

import psutil

from common import MODEL_NAME, PERSON_CLASS_ID, ROOT, list_videos, load_model

MB = 1024 ** 2


# ----------------------------------------------------------------------------- worker (1 video stream)
def stream_worker(wid, cores, threads, cfg, ready_q, start_ev, result_q):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    if cores:
        try:
            psutil.Process().cpu_affinity(cores)
        except Exception:
            pass

    import cv2
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    model = load_model()
    model.predict(np.zeros((cfg["imgsz"], cfg["imgsz"], 3), np.uint8), imgsz=cfg["imgsz"], device="cpu", verbose=False)

    cap = cv2.VideoCapture(cfg["video"])
    # each stream starts at a different position in the video (more realistic)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, (wid * 97) % n_frames)

    ready_q.put((wid, os.getpid()))
    start_ev.wait()

    lat = []
    persons = 0
    t0 = time.perf_counter()
    end = t0 + cfg["duration"]
    while time.perf_counter() < end:
        ok, frame = cap.read()
        if not ok:  # loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        t = time.perf_counter()
        r = model.predict(frame, conf=cfg["conf"], imgsz=cfg["imgsz"], classes=cfg["classes"],
                          device="cpu", verbose=False)[0]
        lat.append(time.perf_counter() - t)
        if r.boxes is not None and len(r.boxes):
            persons += int((r.boxes.cls == PERSON_CLASS_ID).sum())
    elapsed = time.perf_counter() - t0
    cap.release()
    result_q.put({"wid": wid, "frames": len(lat), "elapsed": elapsed, "lat": lat, "persons": persons})


# ----------------------------------------------------------------------------- helpers
def plan_cores(n: int, n_cores: int, threads_opt: str, pin: bool):
    """Return (threads per stream, list of cores for each stream)."""
    if threads_opt == "auto":
        threads = max(1, n_cores // n)
    else:
        threads = int(threads_opt)
    if not pin:
        return threads, [None] * n
    plans = []
    for i in range(n):
        start = (i * threads) % n_cores
        plans.append(sorted({(start + k) % n_cores for k in range(threads)}))
    return threads, plans


def rss_of(pids) -> int:
    total = 0
    for pid in pids:
        try:
            total += psutil.Process(pid).memory_info().rss
        except psutil.Error:
            pass
    return total


def percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return vals[k]


def run_level(n, cfg, args, n_cores, ctx):
    """Run n streams in parallel and measure. Returns a dict of metrics."""
    threads, core_plan = plan_cores(n, n_cores, args.threads, not args.no_pin)
    ready_q, result_q = ctx.Queue(), ctx.Queue()
    start_ev = ctx.Event()
    procs = [ctx.Process(target=stream_worker, args=(i, core_plan[i], threads, cfg, ready_q, start_ev, result_q))
             for i in range(n)]
    for p in procs:
        p.start()

    pids = []
    aborted = None
    t_load = time.perf_counter()
    try:
        while len(pids) < n:
            try:
                pids.append(ready_q.get(timeout=5)[1])
            except queue.Empty:
                if not all(p.is_alive() for p in procs):
                    aborted = "a process crashed while loading (RAM?)"
                    break
                if time.perf_counter() - t_load > 300:
                    aborted = "loading timeout"
                    break
                if psutil.virtual_memory().percent > args.max_ram:
                    aborted = f"RAM > {args.max_ram}% while loading"
                    break
        load_time = time.perf_counter() - t_load
        ram_loaded = rss_of(pids)

        cpu_samples, core_samples, rss_samples, sysram_samples = [], [], [], []
        results = []
        if not aborted:
            psutil.cpu_percent(percpu=True)
            start_ev.set()
            t0 = time.perf_counter()
            while len(results) < n:
                time.sleep(0.5)
                per_core = psutil.cpu_percent(percpu=True)
                core_samples.append(per_core)
                cpu_samples.append(sum(per_core) / len(per_core))
                rss_samples.append(rss_of(pids))
                vm = psutil.virtual_memory()
                sysram_samples.append(vm.percent)
                while True:
                    try:
                        results.append(result_q.get_nowait())
                    except queue.Empty:
                        break
                if vm.percent > args.max_ram:
                    aborted = f"system RAM > {args.max_ram}%"
                    break
                if time.perf_counter() - t0 > cfg["duration"] + 120:
                    aborted = "measurement timeout"
                    break
                if len(results) < n and not any(p.is_alive() for p in procs):
                    try:
                        results.append(result_q.get(timeout=2))
                    except queue.Empty:
                        aborted = "a process crashed"
                        break
    finally:
        for p in procs:
            if aborted and p.is_alive():
                p.kill()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

    m = {"streams": n, "threads_per_stream": threads, "aborted": aborted,
         "load_time_s": round(load_time, 1),
         "ram_loaded_mb": round(ram_loaded / MB),
         "ram_per_stream_mb": round(ram_loaded / MB / max(1, len(pids)))}
    if aborted or not results:
        return m

    # Skip the first ~15% CPU samples (ramp-up)
    skip = len(cpu_samples) // 7
    cpu_s = cpu_samples[skip:] or cpu_samples
    fps_list = [r["frames"] / r["elapsed"] for r in results if r["elapsed"] > 0]
    all_lat = [x for r in results for x in r["lat"]]
    total_frames = sum(r["frames"] for r in results)
    wall = max(r["elapsed"] for r in results)
    peak_rss = max(rss_samples) if rss_samples else ram_loaded
    core_avg = [round(statistics.mean(s[i] for s in core_samples[skip:] or core_samples), 1)
                for i in range(n_cores)] if core_samples else []
    m.update({
        "fps_per_stream_avg": round(statistics.mean(fps_list), 2),
        "fps_per_stream_min": round(min(fps_list), 2),
        "fps_total": round(total_frames / wall, 2),
        "latency_avg_ms": round(statistics.mean(all_lat) * 1000, 1),
        "latency_p95_ms": round(percentile(all_lat, 95) * 1000, 1),
        "cpu_avg_pct": round(statistics.mean(cpu_s), 1),
        "cpu_max_pct": round(max(cpu_samples), 1),
        "cpu_per_core_avg": core_avg,
        "ram_peak_mb": round(peak_rss / MB),
        "ram_peak_per_stream_mb": round(peak_rss / MB / n),
        "sys_ram_peak_pct": round(max(sysram_samples), 1),
        "realtime": min(fps_list) >= cfg["target_fps"],
        "frames": total_frames,
    })
    return m


# ----------------------------------------------------------------------------- chart
def make_chart(configs: list[dict], target_fps: float, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

    panels = [("fps_per_stream_min", "FPS per stream (slowest)", True),
              ("fps_total", "Total FPS (all streams)", False),
              ("cpu_avg_pct", "Average CPU (%)", False),
              ("ram_peak_mb", "YOLO processes RAM (MB)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURF)
    for ax, (key, label, show_target) in zip(axes.flat, panels):
        ax.set_facecolor(SURF)
        for i, c in enumerate(configs):
            pts = [(r["streams"], r[key]) for r in c["levels"] if key in r]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=8,
                    markeredgecolor=SURF, markeredgewidth=2, label=c["name"], zorder=3)
            if len(configs) == 1:  # direct label on the last point
                ax.annotate(f"{ys[-1]:g}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 6),
                            color=INK2, fontsize=9)
        if show_target:
            ax.axhline(target_fps, color=INK2, lw=1, ls="--", zorder=2)
            ax.annotate(f"real time = {target_fps:g} FPS", xy=(0.01, target_fps), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=9)
        if key == "cpu_avg_pct":
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(label, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of parallel videos", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.xaxis.get_major_locator().set_params(integer=True)
    if len(configs) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), frameon=False,
                   ncol=len(configs), fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91 if len(configs) > 1 else 0.95))
    fig.savefig(path, dpi=120, facecolor=SURF)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    n_cores = psutil.cpu_count(logical=True)
    ap = argparse.ArgumentParser(description="yolo26n: how many videos in parallel?")
    ap.add_argument("--video", type=Path, help="test video (default: first video in input_videos)")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640], help="ex: 320 640")
    ap.add_argument("--streams", nargs="+", type=int, help=f"values of N (default: 1..{n_cores * 2})")
    ap.add_argument("--duration", type=float, default=20, help="measurement seconds per level")
    ap.add_argument("--threads", default="auto", help="threads per stream: 'auto' (= cores / N) or a number")
    ap.add_argument("--target-fps", type=float, help="'real time' FPS per stream (default: video FPS)")
    ap.add_argument("--max-ram", type=float, default=92, help="stop if system RAM exceeds this %%")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--persons-only", action="store_true")
    ap.add_argument("--no-pin", action="store_true", help="do not pin processes to cores")
    ap.add_argument("--keep-going", action="store_true",
                    help="keep going even when total throughput drops (default: stop after 2 drops)")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    a = ap.parse_args()

    video = a.video or next(iter(list_videos(ROOT / "input_videos")), None)
    if not video or not Path(video).exists():
        print("No video: put a video in input_videos/ or use --video")
        return

    import cv2
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25
    vres = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    cap.release()
    target = a.target_fps or vfps
    levels = a.streams or list(range(1, n_cores * 2 + 1))

    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    sysinfo = {"processor": platform.processor(), "logical_cores": n_cores,
               "physical_cores": psutil.cpu_count(logical=False),
               "max_freq_mhz": freq.max if freq else None,
               "ram_total_gb": round(vm.total / 1024 ** 3, 1), "ram_available_gb": round(vm.available / 1024 ** 3, 1)}

    print("=" * 78)
    print(" BENCHMARK yolo26n: how many videos in parallel?")
    print("=" * 78)
    print(f"  CPU   : {sysinfo['processor']}  ({n_cores} logical / {sysinfo['physical_cores']} physical)")
    print(f"  RAM   : {sysinfo['ram_total_gb']} GB total, {sysinfo['ram_available_gb']} GB free")
    print(f"  Video : {Path(video).name}  {vres} @ {vfps:.1f} FPS")
    print(f"  Real time = {target:g} FPS per stream | levels N = {levels} | {a.duration:g}s per level")

    ctx = mp.get_context("spawn")
    all_configs = []

    load_model()  # download once, before starting the processes
    for imgsz in a.imgsz:
        name = f"{MODEL_NAME} @ {imgsz}"
        cfg = {"imgsz": imgsz, "conf": a.conf, "video": str(video),
               "classes": [PERSON_CLASS_ID] if a.persons_only else None,
               "duration": a.duration, "target_fps": target}
        print(f"\n--- {name} " + "-" * (70 - len(name)))
        print(f"{'N':>3} {'thr':>3} {'FPS/strm':>9} {'min':>6} {'FPS tot':>8} {'lat ms':>7} {'p95':>6} "
              f"{'CPU%':>5} {'RAM MB':>7} {'MB/strm':>7} {'RAM sys':>7}  real time")
        res = []
        per_stream_mb = 440  # estimate before the first measurement
        best_total, drops = 0.0, 0
        for n in levels:
            avail_mb = psutil.virtual_memory().available / MB
            if n * per_stream_mb > avail_mb * 0.9:
                print(f"{n:>3}  stop: not enough RAM (need ~{n * per_stream_mb:.0f} MB, "
                      f"free {avail_mb:.0f} MB)")
                res.append({"streams": n, "aborted": "not enough RAM (estimate)"})
                break
            m = run_level(n, cfg, a, n_cores, ctx)
            res.append(m)
            if m.get("aborted"):
                print(f"{n:>3}  stop: {m['aborted']}")
                break
            per_stream_mb = m["ram_peak_per_stream_mb"] or per_stream_mb
            print(f"{n:>3} {m['threads_per_stream']:>3} {m['fps_per_stream_avg']:>9.2f} "
                  f"{m['fps_per_stream_min']:>6.2f} {m['fps_total']:>8.2f} {m['latency_avg_ms']:>7.0f} "
                  f"{m['latency_p95_ms']:>6.0f} {m['cpu_avg_pct']:>5.0f} {m['ram_peak_mb']:>7} "
                  f"{m['ram_peak_per_stream_mb']:>7} {m['sys_ram_peak_pct']:>6.0f}%  "
                  f"{'YES' if m['realtime'] else 'no'}", flush=True)
            if m["fps_total"] > best_total:
                best_total, drops = m["fps_total"], 0
            else:
                drops += 1
                if drops >= 2 and not a.keep_going:
                    print("     stop: total throughput no longer increases (CPU saturated). Use --keep-going to continue")
                    break

        ok = [r for r in res if "fps_total" in r]
        rt = [r for r in ok if r["realtime"]]
        best = max(ok, key=lambda r: r["fps_total"]) if ok else None
        summary = {
            "max_realtime_streams": max((r["streams"] for r in rt), default=0),
            "best_throughput_streams": best["streams"] if best else None,
            "best_fps_total": best["fps_total"] if best else None,
            "theoretical_streams_at_target": round(best["fps_total"] / target, 2) if best else 0,
            "ram_per_stream_mb": best["ram_peak_per_stream_mb"] if best else None,
            "max_streams_tested": max((r["streams"] for r in ok), default=0),
        }
        all_configs.append({"name": name, "model": MODEL_NAME, "imgsz": imgsz, "levels": res, "summary": summary})

    # ----- summary
    print("\n" + "=" * 78)
    print(f" SUMMARY  (real time = {target:g} FPS per stream)")
    print("=" * 78)
    for c in all_configs:
        s = c["summary"]
        print(f"\n  {c['name']}")
        print(f"    Real-time streams (max)         : {s['max_realtime_streams']}")
        print(f"    Best total throughput           : {s['best_fps_total']} FPS with {s['best_throughput_streams']} streams")
        print(f"    Theoretical capacity at {target:g} FPS : {s['theoretical_streams_at_target']} streams "
              f"(= total throughput / target FPS)")
        print(f"    RAM per stream                  : ~{s['ram_per_stream_mb']} MB")
        if s["max_realtime_streams"] == 0 and s["best_fps_total"]:
            print(f"    -> Even 1 stream does not reach {target:g} FPS. Try --imgsz 320, "
                  f"or a lower --target-fps (e.g. process 1 frame out of 2).")

    a.reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = a.reports / f"benchmark_{stamp}"
    report = {"timestamp": stamp, "model": MODEL_NAME, "system": sysinfo,
              "video": {"name": Path(video).name, "resolution": vres, "fps": vfps},
              "target_fps": target, "duration_s": a.duration, "configs": all_configs}
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["config", "streams", "threads_per_stream", "fps_per_stream_avg", "fps_per_stream_min", "fps_total",
            "latency_avg_ms", "latency_p95_ms", "cpu_avg_pct", "cpu_max_pct", "ram_peak_mb",
            "ram_peak_per_stream_mb", "sys_ram_peak_pct", "realtime", "aborted"]
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in all_configs:
            for r in c["levels"]:
                w.writerow({"config": c["name"], **r})

    try:
        make_chart(all_configs, target, base.with_suffix(".png"),
                   f"{MODEL_NAME} on {n_cores} cores: parallel videos ({Path(video).name}, {vres})")
        print(f"\nChart  : {base.with_suffix('.png')}")
    except Exception as e:
        print(f"\n(chart not generated: {e})")
    print(f"Report : {base.with_suffix('.json')}")
    print(f"CSV    : {base.with_suffix('.csv')}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
