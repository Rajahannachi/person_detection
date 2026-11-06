"""Petite interface graphique (Tkinter) pour tester yolo26n sur les vidéos.

Lancer :  python app.py

Onglet "Détection"   : choisir une vidéo, voir la détection en direct, sauvegarder la vidéo annotée.
Onglet "Stress test" : lancer stress_test.py avec les options choisies et voir les logs en direct.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import psutil
from PIL import Image, ImageTk

from common import MODEL_NAME, PERSON_CLASS_ID, ROOT, list_videos, load_model, process_video

INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"


def open_path(path: Path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{MODEL_NAME} – Détection de personnes")
        self.geometry("1200x760")
        self.minsize(950, 600)

        self.model = None  # yolo26n, loaded on first detection
        self.frame_q: queue.Queue = queue.Queue(maxsize=2)
        self.log_q: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.det_thread: threading.Thread | None = None
        self.stress_proc: subprocess.Popen | None = None
        self._photo = None

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Big.TLabel", font=("Segoe UI", 11, "bold"))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_det = ttk.Frame(nb)
        self.tab_stress = ttk.Frame(nb)
        nb.add(self.tab_det, text="  Détection  ")
        nb.add(self.tab_stress, text="  Stress test CPU  ")

        self._build_detection_tab()
        self._build_stress_tab()
        self.refresh_videos()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self._poll)

    # ------------------------------------------------------------------ UI: detection tab
    def _build_detection_tab(self):
        left = ttk.Frame(self.tab_det, width=300)
        left.pack(side="left", fill="y", padx=(4, 8), pady=4)
        left.pack_propagate(False)

        ttk.Label(left, text="Vidéos (input_videos)", style="Big.TLabel").pack(anchor="w")
        lb_frame = ttk.Frame(left)
        lb_frame.pack(fill="both", expand=True, pady=4)
        self.video_list = tk.Listbox(lb_frame, height=10, activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(lb_frame, command=self.video_list.yview)
        self.video_list.config(yscrollcommand=sb.set)
        self.video_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.video_list.bind("<Double-Button-1>", lambda e: self.start_detection())

        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Button(row, text="+ Ajouter vidéo…", command=self.add_videos).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="↻", width=3, command=self.refresh_videos).pack(side="left", padx=(4, 0))

        opts = ttk.LabelFrame(left, text="Options")
        opts.pack(fill="x", pady=8)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.imgsz_var = tk.StringVar(value="640")
        self.persons_var = tk.BooleanVar(value=False)
        self.save_var = tk.BooleanVar(value=True)

        g = ttk.Frame(opts)
        g.pack(fill="x", padx=6, pady=6)
        ttk.Label(g, text="Modèle").grid(row=0, column=0, sticky="w")
        ttk.Label(g, text=MODEL_NAME, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="e")
        ttk.Label(g, text="Taille image").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(g, textvariable=self.imgsz_var, values=["320", "416", "480", "640", "960"], width=14).grid(row=1, column=1, sticky="ew")
        ttk.Label(g, text="Confiance").grid(row=2, column=0, sticky="w")
        self.conf_lbl = ttk.Label(g, text="0.25")
        self.conf_lbl.grid(row=2, column=1, sticky="e")
        ttk.Scale(g, from_=0.05, to=0.95, variable=self.conf_var,
                  command=lambda v: self.conf_lbl.config(text=f"{float(v):.2f}")).grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Checkbutton(g, text="Personnes seulement", variable=self.persons_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(g, text="Sauvegarder la vidéo annotée", variable=self.save_var).grid(row=5, column=0, columnspan=2, sticky="w")
        g.columnconfigure(1, weight=1)

        self.btn_start = ttk.Button(left, text="▶  Lancer la détection", command=self.start_detection)
        self.btn_start.pack(fill="x", ipady=4)
        self.btn_stop = ttk.Button(left, text="■  Stop", command=self.stop_detection, state="disabled")
        self.btn_stop.pack(fill="x", pady=4)
        ttk.Button(left, text="📂  Ouvrir output_videos", command=lambda: open_path(OUTPUT_DIR)).pack(fill="x")

        stats = ttk.LabelFrame(left, text="Statistiques")
        stats.pack(fill="x", pady=8)
        self.stat_vars = {k: tk.StringVar(value="–") for k in ("Personnes", "Objets", "Inférence", "FPS moyen", "Frame")}
        for i, (k, v) in enumerate(self.stat_vars.items()):
            ttk.Label(stats, text=k).grid(row=i, column=0, sticky="w", padx=6)
            ttk.Label(stats, textvariable=v, font=("Segoe UI", 10, "bold")).grid(row=i, column=1, sticky="e", padx=6)
        stats.columnconfigure(1, weight=1)

        right = ttk.Frame(self.tab_det)
        right.pack(side="left", fill="both", expand=True, pady=4)
        self.canvas = tk.Canvas(right, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(10, 10, anchor="nw", fill="#aaa", font=("Segoe UI", 12), tags="hint",
                                text="Choisissez une vidéo à gauche puis cliquez sur « Lancer la détection »")
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))
        self.status = tk.StringVar(value="Prêt.")
        ttk.Label(right, textvariable=self.status).pack(anchor="w")

    # ------------------------------------------------------------------ UI: stress tab
    def _build_stress_tab(self):
        top = ttk.Frame(self.tab_stress)
        top.pack(fill="x", padx=6, pady=6)

        n_log = psutil.cpu_count(logical=True)
        n_phy = psutil.cpu_count(logical=False)
        ram = psutil.virtual_memory().total / 1024 ** 3
        ttk.Label(top, text=f"CPU : {n_log} cœurs logiques / {n_phy} physiques   |   RAM : {ram:.1f} Go",
                  style="Big.TLabel").grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 6))

        self.s_workers = tk.IntVar(value=n_log)
        self.s_threads = tk.IntVar(value=1)
        self.s_mode = tk.StringVar(value="parallel")
        self.s_duration = tk.IntVar(value=60)
        self.s_save = tk.BooleanVar(value=False)
        self.s_streams = tk.IntVar(value=0)

        ttk.Label(top, text="Workers (process)").grid(row=1, column=0, sticky="w")
        self.workers_spin = ttk.Spinbox(top, from_=1, to=n_log * 2, textvariable=self.s_workers, width=5)
        self.workers_spin.grid(row=1, column=1, padx=(4, 16))
        ttk.Label(top, text="Threads / worker").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(top, from_=1, to=n_log, textvariable=self.s_threads, width=5).grid(row=1, column=3, padx=(4, 16))
        ttk.Label(top, text="Mode").grid(row=1, column=4, sticky="w")
        mode_cb = ttk.Combobox(top, textvariable=self.s_mode, values=["parallel", "split", "replicate"],
                               state="readonly", width=10)
        mode_cb.grid(row=1, column=5, padx=(4, 16))
        mode_cb.bind("<<ComboboxSelected>>", lambda e: self._on_mode())
        ttk.Label(top, text="Durée (s, 0 = complet)").grid(row=1, column=6, sticky="w")
        ttk.Spinbox(top, from_=0, to=3600, increment=10, textvariable=self.s_duration, width=6).grid(row=1, column=7, padx=4)
        ttk.Checkbutton(top, text="Sauvegarder les vidéos", variable=self.s_save).grid(row=2, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(top, text=f"({MODEL_NAME} ; taille / confiance / personnes : options de l'onglet Détection)",
                  foreground="#666").grid(row=2, column=3, columnspan=5, sticky="w")
        ttk.Label(top, text="Nombre de vidéos (dupliquées)").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Spinbox(top, from_=0, to=32, textvariable=self.s_streams, width=5,
                    command=self._on_mode).grid(row=4, column=2, sticky="w", padx=4)
        ttk.Label(top, text="0 = chaque vidéo une fois ; ex. 6 avec 2 vidéos -> chaque vidéo 3 fois (cam1..cam6)",
                  foreground="#666").grid(row=4, column=3, columnspan=5, sticky="w")
        self.mode_help = tk.StringVar()
        ttk.Label(top, textvariable=self.mode_help, foreground="#1c5cab").grid(row=3, column=0, columnspan=8, sticky="w")
        self._on_mode()

        btns = ttk.Frame(self.tab_stress)
        btns.pack(fill="x", padx=6)
        self.btn_stress = ttk.Button(btns, text="▶  Lancer le stress test", command=self.start_stress)
        self.btn_stress.pack(side="left", ipady=3)
        self.btn_stress_stop = ttk.Button(btns, text="■  Arrêter", command=self.stop_stress, state="disabled")
        self.btn_stress_stop.pack(side="left", padx=6)
        ttk.Button(btns, text="📂  Ouvrir reports", command=lambda: open_path(ROOT / "reports")).pack(side="left")

        # live per-core bars
        self.core_frame = ttk.LabelFrame(self.tab_stress, text="Usage CPU par cœur (live)")
        self.core_frame.pack(fill="x", padx=6, pady=8)
        self.core_bars = []
        for i in range(n_log):
            ttk.Label(self.core_frame, text=f"Core {i}").grid(row=i, column=0, sticky="w", padx=6)
            pb = ttk.Progressbar(self.core_frame, maximum=100, length=400)
            pb.grid(row=i, column=1, sticky="ew", padx=6, pady=1)
            lbl = ttk.Label(self.core_frame, text="0%", width=6)
            lbl.grid(row=i, column=2)
            self.core_bars.append((pb, lbl))
        self.core_frame.columnconfigure(1, weight=1)

        log_frame = ttk.Frame(self.tab_stress)
        log_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.log = tk.Text(log_frame, bg="#111", fg="#ddd", font=("Consolas", 9), wrap="none")
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.config(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        psutil.cpu_percent(percpu=True)
        self.after(1000, self._update_cores)

    def _on_mode(self):
        mode = self.s_mode.get()
        try:
            streams = int(self.s_streams.get())
        except (tk.TclError, ValueError):
            streams = 0
        n = streams if streams > 0 else len(list_videos(INPUT_DIR))
        self.workers_spin.config(state="disabled" if mode == "parallel" else "normal")
        self.mode_help.set({
            "parallel": f"parallel : toutes les vidéos en même temps, 1 worker par vidéo ({n} vidéo(s) -> {n} worker(s)). "
                        "Chaque vidéo est traitée en entier par son worker.",
            "split": "split : N workers (1 par cœur) prennent les vidéos une par une ; les vidéos en trop attendent.",
            "replicate": "replicate : chaque worker traite toutes les vidéos (charge maximale, pour tester le CPU).",
        }[mode])

    # ------------------------------------------------------------------ videos
    def refresh_videos(self):
        self.videos = list_videos(INPUT_DIR)
        self.video_list.delete(0, "end")
        for v in self.videos:
            self.video_list.insert("end", f"{v.name}   ({v.stat().st_size / 1e6:.1f} Mo)")
        if self.videos:
            self.video_list.selection_set(0)
        self._on_mode()  # refresh the "N videos -> N workers" hint

    def add_videos(self):
        files = filedialog.askopenfilenames(title="Choisir des vidéos",
                                            filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"), ("Tous", "*.*")])
        INPUT_DIR.mkdir(exist_ok=True)
        for f in files:
            dst = INPUT_DIR / Path(f).name
            if Path(f).resolve() != dst.resolve():
                shutil.copy2(f, dst)
        self.refresh_videos()

    # ------------------------------------------------------------------ detection
    def get_model(self):
        if self.model is None:
            self.model = load_model()
        return self.model

    def start_detection(self):
        if self.det_thread and self.det_thread.is_alive():
            return
        sel = self.video_list.curselection()
        if not sel:
            messagebox.showinfo("Vidéo", "Choisissez une vidéo (ou ajoutez-en une).")
            return
        video = self.videos[sel[0]]
        try:
            imgsz = int(self.imgsz_var.get())
        except ValueError:
            imgsz = 640
        params = dict(conf=round(self.conf_var.get(), 2), imgsz=imgsz,
                      classes=[PERSON_CLASS_ID] if self.persons_var.get() else None,
                      out=(OUTPUT_DIR / f"{video.stem}_detected.mp4") if self.save_var.get() else None)

        self.stop_flag.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.canvas.delete("hint")
        self.progress.config(value=0)
        self.det_thread = threading.Thread(target=self._detect_worker, args=(video, params), daemon=True)
        self.det_thread.start()

    def _detect_worker(self, video: Path, p: dict):
        try:
            self.log_q.put(("status", f"Chargement du modèle {MODEL_NAME} …"))
            model = self.get_model()
            self.log_q.put(("status", f"Détection sur {video.name} …"))

            def on_frame(frame, info):
                if self.frame_q.full():
                    try:
                        self.frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_q.put((frame.copy(), info))
                return not self.stop_flag.is_set()

            st = process_video(model, video, p["out"], conf=p["conf"], imgsz=p["imgsz"], classes=p["classes"],
                               progress_every=0, on_frame=on_frame)
            msg = (f"Terminé : {st['frames']} frames, {st['fps']} FPS, max {st['max_persons_in_frame']} personnes/frame"
                   + (f"  →  {Path(st['output']).name}" if st["output"] else ""))
            if self.stop_flag.is_set():
                msg = "Arrêté. " + msg
            self.log_q.put(("status", msg))
        except Exception as e:
            self.log_q.put(("status", f"Erreur : {e}"))
        finally:
            self.log_q.put(("det_done", None))

    def stop_detection(self):
        self.stop_flag.set()

    def _show_frame(self, frame, info):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.cvtColor(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("img")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center", tags="img")

        total = info["total"]
        self.stat_vars["Personnes"].set(str(info["persons"]))
        self.stat_vars["Objets"].set(str(info["objects"]))
        self.stat_vars["Inférence"].set(f"{info['infer_ms']:.0f} ms")
        self.stat_vars["FPS moyen"].set(f"{info['avg_fps']:.1f}")
        self.stat_vars["Frame"].set(f"{info['frame']} / {total if total > 0 else '?'}")
        if total > 0:
            self.progress.config(value=info["frame"] / total * 100)

    # ------------------------------------------------------------------ stress test
    def start_stress(self):
        if self.stress_proc and self.stress_proc.poll() is None:
            return
        if not list_videos(INPUT_DIR):
            messagebox.showinfo("Vidéos", "Ajoutez au moins une vidéo dans input_videos.")
            return
        if self.det_thread and self.det_thread.is_alive():
            messagebox.showinfo("Occupé", "Arrêtez d'abord la détection en cours.")
            return
        cmd = [sys.executable, "-u", str(ROOT / "stress_test.py"), "--threads", str(self.s_threads.get()),
               "--mode", self.s_mode.get(), "--duration", str(self.s_duration.get()),
               "--imgsz", self.imgsz_var.get(),
               "--conf", f"{self.conf_var.get():.2f}", "--progress-every", "50"]
        if self.s_mode.get() != "parallel":
            cmd += ["--workers", str(self.s_workers.get())]
        if self.s_streams.get() > 0:
            cmd += ["--streams", str(self.s_streams.get())]
        if not self.s_save.get():
            cmd.append("--no-save")
        if self.persons_var.get():
            cmd.append("--persons-only")

        self.log.delete("1.0", "end")
        self._append_log("$ " + " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)) + "\n\n")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.stress_proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                                            creationflags=flags, env=env)
        self.btn_stress.config(state="disabled")
        self.btn_stress_stop.config(state="normal")
        threading.Thread(target=self._read_stress, daemon=True).start()

    def _read_stress(self):
        proc = self.stress_proc
        for line in proc.stdout:
            self.log_q.put(("log", line))
        proc.wait()
        self.log_q.put(("log", f"\n[process terminé, code {proc.returncode}]\n"))
        self.log_q.put(("stress_done", None))

    def stop_stress(self):
        proc = self.stress_proc
        if proc and proc.poll() is None:
            if sys.platform == "win32":  # kill the whole process tree (all workers)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                for c in psutil.Process(proc.pid).children(recursive=True):
                    c.kill()
                proc.kill()

    def _append_log(self, text: str):
        self.log.insert("end", text)
        self.log.see("end")

    def _update_cores(self):
        for (pb, lbl), v in zip(self.core_bars, psutil.cpu_percent(percpu=True)):
            pb.config(value=v)
            lbl.config(text=f"{v:.0f}%")
        self.after(1000, self._update_cores)

    # ------------------------------------------------------------------ main loop
    def _poll(self):
        try:
            frame, info = self.frame_q.get_nowait()
            self._show_frame(frame, info)
        except queue.Empty:
            pass
        lines = []
        while True:
            try:
                kind, val = self.log_q.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(val)
            elif kind == "det_done":
                self.btn_start.config(state="normal")
                self.btn_stop.config(state="disabled")
            elif kind == "log":
                if not val.startswith("[MON"):  # MON lines are shown by the live bars
                    lines.append(val)
            elif kind == "stress_done":
                self.btn_stress.config(state="normal")
                self.btn_stress_stop.config(state="disabled")
        if lines:
            self._append_log("".join(lines))
        self.after(30, self._poll)

    def on_close(self):
        self.stop_flag.set()
        self.stop_stress()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
