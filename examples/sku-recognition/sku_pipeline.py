"""
SKU Recognition - DeepStream Line Crossing Pipeline
====================================================
ใช้ DeepStream pipeline ตรวจจับชิ้นงานบนสายพานผ่าน line crossing detection
เมื่อชิ้นงานผ่านเส้น จะ capture ภาพบันทึกลง output_folder

Pipeline:
  nvurisrcbin → nvstreammux → nvinferserver → nvtracker
               → nvvideoconvert → nvdsanalytics
               → [sku_capture probe]           ← บันทึกภาพตรงนี้
               → nvdsosd → fakesink/xvimagesink

Usage:
    python sku_pipeline.py --source rtsp://192.168.1.100:554/stream
    python sku_pipeline.py --source file:///path/to/video.mp4
    python sku_pipeline.py --source /path/to/video.mp4 --output ./captures

Environment variables:
    USE_GDINO=true/false    ใช้ GroundingDINO หรือ nvinfer (default: true)
    DISABLE_SOM_OVERLAY=true/false
"""

import sys
import os
import re
import time
import argparse
import sysconfig
import threading
import queue as queue_module

# ---- ต้อง set GST_PLUGIN_PATH ก่อน import pyservicemaker ----
_PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gst-plugins")
_existing = os.environ.get("GST_PLUGIN_PATH", "")
os.environ["GST_PLUGIN_PATH"] = f"{_PLUGIN_DIR}:{_existing}" if _existing else _PLUGIN_DIR

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd
from simple_config_updater import update_config_type_name
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

np.random.seed(1000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_NAME = "sku-recognition"
STREAM_WIDTH = 3840
STREAM_HEIGHT = 2160

USE_GDINO = (os.environ.get("USE_GDINO", "true") == "true")
USE_CUSTOM_MODEL = (os.environ.get("USE_CUSTOM_MODEL", "false") == "true")
DISABLE_SOM_OVERLAY = (os.environ.get("DISABLE_SOM_OVERLAY", "false") == "true")
ENABLE_DISPLAY = (os.environ.get("ENABLE_DISPLAY", "false") == "true")


def _can_open_display(display: str) -> bool:
    """ตรวจว่า X display เปิดได้จริงไหม."""
    try:
        import subprocess
        result = subprocess.run(
            ["xdpyinfo", "-display", display],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            import socket
            if display.startswith(":"):
                num = display.split(":")[1].split(".")[0]
                sock_path = f"/tmp/.X11-unix/X{num}"
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect(sock_path)
                s.close()
                return True
            else:
                host, rest = display.rsplit(":", 1)
                port = 6000 + int(rest.split(".")[0])
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect((host or "localhost", port))
                s.close()
                return True
        except Exception:
            return False


# Path ของ config files (อยู่ในโฟลเดอร์เดียวกับ script นี้)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NVDSANALYTICS = os.path.join(_SCRIPT_DIR, "config_nvdsanalytics.txt")

CONFIG_INFER_GDINO   = os.path.join(_SCRIPT_DIR, "gdinoconfig_grpc.txt")
CONFIG_INFER_NVINFER = os.path.join(_SCRIPT_DIR, "nvdsinfer_config.yaml")
CONFIG_INFER_CUSTOM  = os.environ.get(
    "SKU_CUSTOM_MODEL_CONFIG",
    os.path.join(_SCRIPT_DIR, "custom_model", "nvinfer_config.yaml")
)
TRACKER_CONFIG = os.path.join(_SCRIPT_DIR, "via_tracker_config_fast.yml")


# ---------------------------------------------------------------------------
# Parse line coordinates from nvdsanalytics config
# ---------------------------------------------------------------------------

def parse_line_coords(config_path: str, line_name: str):
    """
    อ่าน config_nvdsanalytics.txt แล้วดึงพิกัดเส้นของ line_name ออกมา
    Format: line-crossing-{NAME}=x1;y1;x2;y2
    คืนค่า (x1, y1, x2, y2) หรือ default กลาง frame ถ้าไม่เจอ
    """
    default = (STREAM_WIDTH // 2, 50, STREAM_WIDTH // 2, STREAM_HEIGHT - 50)
    try:
        pattern = re.compile(
            rf"line-crossing-{re.escape(line_name)}\s*=\s*(\d+);(\d+);(\d+);(\d+)"
        )
        with open(config_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
                    print(f"[SKU] Line '{line_name}': ({x1},{y1})→({x2},{y2})", flush=True)
                    return x1, y1, x2, y2
    except Exception as e:
        print(f"[SKU] Warning: cannot parse line coords from {config_path}: {e}", flush=True)
    print(f"[SKU] Using default line coords: {default}", flush=True)
    return default


# ---------------------------------------------------------------------------
# Background frame reader (OpenCV) + async JPEG saver
# ---------------------------------------------------------------------------

class FrameCaptureBG:
    """
    Background thread ที่อ่าน frame จาก VideoCapture และเก็บ frame ล่าสุดไว้
    เมื่อมี line crossing → ดึง frame ออกแล้วส่งไปบันทึกใน save thread
    """

    def __init__(self, source_url: str):
        # แปลง file:// URI → local path สำหรับ OpenCV
        if source_url.startswith("file://"):
            self._src = source_url[7:]
        else:
            self._src = source_url

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._read_thread = None
        self._save_thread = None
        self._save_queue = queue_module.Queue(maxsize=200)

    def start(self):
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True, name="FrameReader")
        self._read_thread.start()
        self._save_thread = threading.Thread(target=self._save_loop, daemon=True, name="FrameSaver")
        self._save_thread.start()

    def _read_loop(self):
        is_rtsp = self._src.startswith("rtsp://")
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            print(f"[FrameCaptureBG] Cannot open: {self._src}", flush=True)
            return

        print(f"[FrameCaptureBG] Opened: {self._src}", flush=True)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                if is_rtsp:
                    time.sleep(0.05)
                    continue
                else:
                    # ไฟล์วีดีโอ: loop กลับต้น
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            with self._lock:
                self._frame = frame  # BGR, HWC

        cap.release()

    def get_frame(self):
        """คืน copy ของ frame ล่าสุด (BGR) หรือ None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _save_loop(self):
        while True:
            item = self._save_queue.get()
            if item is None:
                break
            path, frame, quality = item
            try:
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                print(f"[SKU Capture] Saved: {path}", flush=True)
            except Exception as e:
                print(f"[SKU Capture] Save error {path}: {e}", flush=True)

    def schedule_save(self, path: str, frame, quality: int = 95):
        try:
            self._save_queue.put_nowait((path, frame, quality))
        except queue_module.Full:
            print("[FrameCaptureBG] Save queue full, dropping frame", flush=True)

    def stop(self):
        self._running = False
        self._save_queue.put(None)


# ---------------------------------------------------------------------------
# Line Crossing Capture — BatchMetadataOperator probe
# ---------------------------------------------------------------------------

class LineCrossingCaptureProbe(BatchMetadataOperator):
    """
    Centroid-based line crossing detector.

    ไม่ใช้ pyds NvDsAnalyticsObjInfo (ไม่ accessible จาก BatchMetadataOperator)
    แต่ track centroid ของ object ข้ามเฟรม แล้วตรวจว่าเส้นผ่านทาง geometric cross-product

    Frame capture ทำผ่าน background OpenCV VideoCapture thread
    (ไม่ต้องการ GstBuffer access)
    """

    def __init__(self, output_dir: str, source_url: str,
                 line_x1: int, line_y1: int, line_x2: int, line_y2: int,
                 line_name: str = "SKU_LINE",
                 cooldown_sec: float = 2.0,
                 crop_padding: float = 0.15):
        super().__init__()
        self.output_dir = output_dir
        self.line_name = line_name
        self.cooldown_sec = cooldown_sec
        self.crop_padding = crop_padding

        # Line segment endpoints
        self.lx1, self.ly1 = line_x1, line_y1
        self.lx2, self.ly2 = line_x2, line_y2

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.capture_count = 0

        self._last_captured: dict = {}   # obj_id → last capture timestamp
        self._prev_centroids: dict = {}  # obj_id → (cx, cy) previous frame

        # Background reader
        self._cap_bg = FrameCaptureBG(source_url)
        self._cap_bg.start()

        self._rgb_colors = np.random.random((1000, 3))
        print(
            f"[SKU Probe] Ready — output: {output_dir} | line: '{line_name}' "
            f"({line_x1},{line_y1})→({line_x2},{line_y2})",
            flush=True,
        )

    # ------------------------------------------------------------------ #

    def _crossed_line(self, px: float, py: float, cx: float, cy: float) -> bool:
        """True ถ้า segment (px,py)→(cx,cy) ตัดกับเส้น (lx1,ly1)→(lx2,ly2)."""
        dx = self.lx2 - self.lx1
        dy = self.ly2 - self.ly1
        # Cross-product signs
        d1 = dx * (py - self.ly1) - dy * (px - self.lx1)
        d2 = dx * (cy - self.ly1) - dy * (cx - self.lx1)
        return (d1 > 0 and d2 <= 0) or (d1 < 0 and d2 >= 0)

    def _trigger_capture(self, obj_meta, label: str):
        """ดึง frame จาก background reader แล้ว schedule บันทึกไฟล์."""
        frame = self._cap_bg.get_frame()
        if frame is None:
            print("[SKU Probe] No frame available yet", flush=True)
            return

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        obj_id = obj_meta.object_id
        fh, fw = frame.shape[:2]

        left = int(obj_meta.rect_params.left)
        top = int(obj_meta.rect_params.top)
        w = int(obj_meta.rect_params.width)
        h = int(obj_meta.rect_params.height)

        # Crop with padding
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)
        x1 = max(0, left - pad_x)
        y1 = max(0, top - pad_y)
        x2 = min(fw, left + w + pad_x)
        y2 = min(fh, top + h + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crop_path = os.path.join(
                self.output_dir, f"sku_{now_str}_id{obj_id}_{label}_crop.jpg"
            )
            self._cap_bg.schedule_save(crop_path, crop, quality=95)

        # Full frame with bbox overlay
        full = frame.copy()
        cv2.rectangle(full, (left, top), (left + w, top + h), (0, 255, 0), 3)
        cv2.putText(
            full, f"ID:{obj_id} {label}", (left, max(16, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        # วาดเส้น crossing
        cv2.line(full, (self.lx1, self.ly1), (self.lx2, self.ly2), (0, 0, 255), 2)

        full_path = os.path.join(
            self.output_dir, f"sku_{now_str}_id{obj_id}_{label}_full.jpg"
        )
        self._cap_bg.schedule_save(full_path, full, quality=90)

        self.capture_count += 1

    # ------------------------------------------------------------------ #

    def handle_metadata(self, batch_meta):
        """เรียกทุก frame — ตรวจ line crossing และ trigger capture."""
        now = time.time()

        for frame_meta in batch_meta.frame_items:
            for obj in frame_meta.object_items:
                obj_id = obj.object_id

                # Centroid ปัจจุบัน
                rect = obj.rect_params
                cx = rect.left + rect.width / 2
                cy = rect.top + rect.height / 2

                # ตรวจ line crossing เทียบกับ frame ก่อนหน้า
                prev = self._prev_centroids.get(obj_id)
                if prev is not None:
                    px, py = prev
                    if self._crossed_line(px, py, cx, cy):
                        last_t = self._last_captured.get(obj_id, 0.0)
                        if now - last_t >= self.cooldown_sec:
                            self._last_captured[obj_id] = now
                            label = getattr(obj, "obj_label", "") or "obj"
                            print(
                                f"[SKU Probe] *** LINE CROSSING *** "
                                f"obj_id={obj_id} label={label} "
                                f"pos=({cx:.0f},{cy:.0f})",
                                flush=True,
                            )
                            self._trigger_capture(obj, label)

                self._prev_centroids[obj_id] = (cx, cy)

                # OSD: สีแตกต่างกันตาม obj_id
                color = self._rgb_colors[obj_id % 1000]
                obj.rect_params.border_color = osd.Color(
                    float(color[0]), float(color[1]), float(color[2]), 1.0
                )
                obj.rect_params.border_width = 2

                text = obj.text_params
                text.display_text = f"{obj_id}".encode("ascii")
                text.x_offset = int(cx)
                text.y_offset = int(cy)
                obj.text_params = text


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_and_run_pipeline(source_url: str,
                            output_folder: str,
                            gdinoprompt: str,
                            gdinothreshold: float,
                            line_name: str,
                            cooldown_sec: float,
                            frame_skip_interval: int,
                            stream_name: str):
    """Build and run the SKU recognition DeepStream pipeline."""

    # ---- ตรวจสอบ source URL ----
    if not (source_url.startswith("rtsp://") or source_url.startswith("file://")):
        source_url = "file://" + os.path.abspath(source_url)

    is_live = source_url.startswith("rtsp://")

    # ---- เลือก infer config ----
    if USE_CUSTOM_MODEL:
        infer_config = CONFIG_INFER_CUSTOM
        infer_element = "nvinfer"
    elif USE_GDINO:
        infer_config = CONFIG_INFER_GDINO
        infer_element = "nvinferserver"
    else:
        infer_config = CONFIG_INFER_NVINFER
        infer_element = "nvinfer"

    print(f"[SKU Pipeline] Source: {source_url}")
    print(f"[SKU Pipeline] Output: {output_folder}")
    print(f"[SKU Pipeline] Line: '{line_name}' | Prompt: '{gdinoprompt}'")
    print(f"[SKU Pipeline] Model: {'CUSTOM' if USE_CUSTOM_MODEL else ('GDino' if USE_GDINO else 'nvinfer')}")

    # ---- อัปเดต GDino config ด้วย prompt และ threshold จริง ----
    if USE_GDINO and not USE_CUSTOM_MODEL:
        infer_config = update_config_type_name(
            CONFIG_INFER_GDINO, gdinoprompt, gdinothreshold, frame_skip_interval
        )
        print(f"[SKU Pipeline] GDino config updated: prompt='{gdinoprompt}' threshold={gdinothreshold}")

    # ---- Parse line coordinates ----
    lx1, ly1, lx2, ly2 = parse_line_coords(CONFIG_NVDSANALYTICS, line_name)

    pipeline = Pipeline(PIPELINE_NAME)

    # Source
    pipeline.add("nvurisrcbin", "decbin", {"uri": source_url})
    if is_live:
        pipeline["decbin"].set({
            "latency": 500, "leaky": 2,
            "max-size-buffers": 2, "num-extra-surfaces": 10,
            "init-rtsp-reconnect-interval": 10
        })

    # Mux
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1,
        "width": STREAM_WIDTH, "height": STREAM_HEIGHT,
        "batched-push-timeout": -1,
        "live-source": is_live,
        "buffer-pool-size": 16,
    })

    # Inference
    pipeline.add(infer_element, "inferserver", {"config-file-path": infer_config})

    # Queues
    pipeline.add("queue", "queue1")
    pipeline.add("queue", "queue2")
    pipeline.add("queue", "queue3")

    # Tracker
    pipeline.add("nvtracker", "tracker", {
        "user-meta-pool-size": 256,
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": TRACKER_CONFIG,
        "compute-hw": 1,
    })

    # Video convert (ก่อน nvdsanalytics)
    pipeline.add("nvvideoconvert", "convert1", {"compute-hw": 1})
    # Video convert (ก่อน nvdsosd — ต้องการ format RGBA)
    pipeline.add("nvvideoconvert", "convert2", {"compute-hw": 1})

    # nvdsanalytics — วาดเส้น line crossing บน OSD
    pipeline.add("nvdsanalytics", "analytics", {
        "config-file": CONFIG_NVDSANALYTICS,
    })

    # OSD
    if DISABLE_SOM_OVERLAY:
        pipeline.add("queue", "nvdsosd")
    else:
        pipeline.add("nvdsosd", "nvdsosd")

    # ---- Sink: display หรือ fakesink ----
    _arch = sysconfig.get_platform()
    _display_env = os.environ.get("DISPLAY", "")
    _use_display = ENABLE_DISPLAY and bool(_display_env) and _can_open_display(_display_env)

    if ENABLE_DISPLAY and not _display_env:
        print("[SKU Pipeline] WARNING: ENABLE_DISPLAY=true แต่ไม่พบ DISPLAY env — fallback fakesink", flush=True)
    elif ENABLE_DISPLAY and _display_env and not _use_display:
        print(f"[SKU Pipeline] WARNING: ไม่สามารถเชื่อมต่อ display '{_display_env}' — fallback fakesink", flush=True)

    if _use_display:
        if "aarch64" in _arch:
            pipeline.add("nvoverlaysink", "sink", {"sync": False, "display-id": 0})
        else:
            pipeline.add("nvvideoconvert", "convert3", {"compute-hw": 1})
            pipeline.add("capsfilter", "cpucaps", {"caps": "video/x-raw, format=BGRx"})
            pipeline.add("xvimagesink", "sink", {"sync": False})
        print(f"[SKU Pipeline] Display mode: ON (DISPLAY={_display_env})", flush=True)
    else:
        pipeline.add("fakesink", "sink", {"sync": False, "qos": False})
        if not ENABLE_DISPLAY:
            print("[SKU Pipeline] Display mode: OFF", flush=True)

    # ---- Attach line crossing capture probe (หลัง analytics) ----
    capture_probe = LineCrossingCaptureProbe(
        output_dir=output_folder,
        source_url=source_url,
        line_x1=lx1, line_y1=ly1, line_x2=lx2, line_y2=ly2,
        line_name=line_name,
        cooldown_sec=cooldown_sec,
        crop_padding=0.15,
    )
    pipeline.attach("analytics", Probe("sku_capture", capture_probe))

    # ---- Link pipeline ----
    pipeline.link(("decbin", "mux"), ("", "sink_%u"))
    if _use_display and "aarch64" not in _arch:
        pipeline.link(
            "mux", "inferserver", "queue1", "tracker",
            "queue2", "convert1", "analytics",
            "convert2", "nvdsosd", "queue3", "convert3", "cpucaps", "sink"
        )
    else:
        pipeline.link(
            "mux", "inferserver", "queue1", "tracker",
            "queue2", "convert1", "analytics",
            "convert2", "nvdsosd", "queue3", "sink"
        )

    print("[SKU Pipeline] Starting...", flush=True)
    pipeline.start().wait()

    print(f"\n[SKU Pipeline] Done. Total captures: {capture_probe.capture_count}", flush=True)
    print(f"[SKU Pipeline] Images saved to: {output_folder}", flush=True)
    capture_probe._cap_bg.stop()
    time.sleep(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SKU Recognition - DeepStream Line Crossing")
    parser.add_argument("--source", required=True,
                        help="Video source: RTSP URL, file:// URI, or local file path")
    parser.add_argument("--output", default="./captures",
                        help="Output folder for captured JPEG images (default: ./captures)")
    parser.add_argument("--prompt", default="product . box . bottle . can . package",
                        help="GDino detection prompt (objects to detect on conveyor)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--line-name", default="SKU_LINE",
                        help="Line name to monitor — must match config_nvdsanalytics.txt (default: SKU_LINE)")
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Cooldown seconds between captures of same object (default: 2.0)")
    parser.add_argument("--frame-skip", type=int, default=0,
                        help="Inference frame skip interval (default: 0 = every frame)")
    parser.add_argument("--stream-name", default="conveyor",
                        help="Stream name for output file naming (default: conveyor)")
    parser.add_argument("--display", action="store_true",
                        help="Show output window (requires DISPLAY env / X11). "
                             "Can also set ENABLE_DISPLAY=true env var.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.display:
        os.environ["ENABLE_DISPLAY"] = "true"
        globals()["ENABLE_DISPLAY"] = True
    build_and_run_pipeline(
        source_url=args.source,
        output_folder=args.output,
        gdinoprompt=args.prompt,
        gdinothreshold=args.threshold,
        line_name=args.line_name,
        cooldown_sec=args.cooldown,
        frame_skip_interval=args.frame_skip,
        stream_name=args.stream_name,
    )
