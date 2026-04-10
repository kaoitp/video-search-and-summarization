"""
SKU Recognition - DeepStream Line Crossing Pipeline
====================================================
ใช้ DeepStream pipeline ตรวจจับชิ้นงานบนสายพานผ่าน line crossing detection
เมื่อชิ้นงานผ่านเส้น จะ capture ภาพบันทึกลง output_folder

Pipeline:
  nvurisrcbin → nvstreammux → nvinferserver → nvtracker
               → nvvideoconvert → nvdsanalytics
               → [skucapture probe]           ← บันทึกภาพตรงนี้
               → nvdsosd → fakesink

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
import time
import argparse
import sysconfig

# ---- ต้อง set GST_PLUGIN_PATH ก่อน import pyservicemaker ----
_PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gst-plugins")
_existing = os.environ.get("GST_PLUGIN_PATH", "")
os.environ["GST_PLUGIN_PATH"] = f"{_PLUGIN_DIR}:{_existing}" if _existing else _PLUGIN_DIR

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd
import pyds
import numpy as np
import cv2
import torch
import time
from datetime import datetime
from pathlib import Path

try:
    from pyservicemaker import Buffer
    _HAS_BUFFER = True
except ImportError:
    _HAS_BUFFER = False

np.random.seed(1000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_NAME = "sku-recognition"
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

USE_GDINO = (os.environ.get("USE_GDINO", "true") == "true")
USE_CUSTOM_MODEL = (os.environ.get("USE_CUSTOM_MODEL", "false") == "true")
DISABLE_SOM_OVERLAY = (os.environ.get("DISABLE_SOM_OVERLAY", "false") == "true")

# Path ของ config files (อยู่ในโฟลเดอร์เดียวกับ script นี้)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NVDSANALYTICS = os.path.join(_SCRIPT_DIR, "config_nvdsanalytics.txt")

# ใช้ config จาก cv-event-detector สำหรับ inference (GDino/nvinfer)
_CV_DIR = os.path.join(_SCRIPT_DIR, "..", "cv-event-detector")
CONFIG_INFER_GDINO   = os.path.join(_CV_DIR, "gdinoconfig_grpc.txt")
CONFIG_INFER_NVINFER = os.path.join(_CV_DIR, "nvdsinfer_config.yaml")
# Custom model config — override ด้วย env var SKU_CUSTOM_MODEL_CONFIG
CONFIG_INFER_CUSTOM  = os.environ.get(
    "SKU_CUSTOM_MODEL_CONFIG",
    os.path.join(_SCRIPT_DIR, "custom_model", "nvinfer_config.yaml")
)
TRACKER_CONFIG = os.path.join(_CV_DIR, "via_tracker_config_fast.yml")

# ---------------------------------------------------------------------------
# Line Crossing Capture — BatchMetadataOperator probe
# ---------------------------------------------------------------------------

class LineCrossingCaptureProbe(BatchMetadataOperator):
    """
    DeepStream BatchMetadataOperator probe ที่ detect line crossing จาก
    NvDsAnalyticsObjInfo.lcStatus และ capture frame ลงไฟล์

    หมายเหตุ: BatchMetadataOperator ไม่มี access ไปยัง GstBuffer โดยตรง
    ดังนั้น frame capture ในที่นี้ใช้ pyds.get_nvds_buf_surface ผ่าน batch_id
    ซึ่งต้องการให้ upstream convert เป็น RGBA/RGB ก่อน
    """

    def __init__(self, output_dir: str, line_name: str = "SKU_LINE",
                 cooldown_sec: float = 2.0, save_full_frame: bool = True,
                 crop_padding: float = 0.15):
        super().__init__()
        self.output_dir = output_dir
        self.line_name = line_name
        self.cooldown_sec = cooldown_sec
        self.save_full_frame = save_full_frame
        self.crop_padding = crop_padding

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.capture_count = 0
        self._last_captured: dict = {}  # obj_id -> timestamp

        self._rgb_colors = np.random.random((1000, 3))
        print(f"[SKU Probe] Ready — output: {output_dir} | line: '{line_name}'", flush=True)

    # ------------------------------------------------------------------ #

    def _save_numpy_frame(self, frame_rgb: np.ndarray, obj_meta, frame_meta):
        """Save crop and full frame from numpy array."""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        obj_id = obj_meta.object_id
        label = getattr(obj_meta, "obj_label", "") or "obj"

        fh, fw = frame_rgb.shape[:2]
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
        crop = frame_rgb[y1:y2, x1:x2]

        if crop.size > 0:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR if frame_rgb.shape[2] == 4 else cv2.COLOR_RGB2BGR)
            crop_path = os.path.join(self.output_dir, f"sku_{now_str}_id{obj_id}_{label}_crop.jpg")
            cv2.imwrite(crop_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[SKU Probe] *** LINE CROSSING *** → {crop_path}", flush=True)

        if self.save_full_frame:
            full = frame_rgb.copy()
            cv2.rectangle(full, (left, top), (left + w, top + h), (0, 255, 0, 255), 3)
            cv2.putText(full, f"ID:{obj_id} {label}", (left, max(0, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)
            full_c = cv2.cvtColor(full, cv2.COLOR_RGBA2BGR if full.shape[2] == 4 else cv2.COLOR_RGB2BGR)
            full_path = os.path.join(self.output_dir, f"sku_{now_str}_id{obj_id}_{label}_full.jpg")
            cv2.imwrite(full_path, full_c, [cv2.IMWRITE_JPEG_QUALITY, 90])

        self.capture_count += 1

    def _try_capture(self, frame_meta, obj_meta):
        """Attempt to extract frame pixels via pyds.get_nvds_buf_surface."""
        try:
            # pyds.get_nvds_buf_surface ต้องการ surface_index (= batch_id)
            # ฟังก์ชันนี้ return numpy array (RGBA, HWC)
            frame_array = pyds.get_nvds_buf_surface(
                hash(frame_meta),  # fallback: ใช้ frame_meta address (อาจไม่ถูกต้องทุก version)
                frame_meta.batch_id
            )
            if frame_array is not None:
                self._save_numpy_frame(frame_array, obj_meta, frame_meta)
                return True
        except Exception as e:
            print(f"[SKU Probe] Frame capture failed (pyds): {e}", flush=True)
        return False

    # ------------------------------------------------------------------ #

    def handle_metadata(self, batch_meta):
        """Called every frame. Check line crossing and trigger capture."""
        now = time.time()

        for frame_meta in batch_meta.frame_items:
            # Draw OSD colors for tracked objects
            for object_meta in frame_meta.object_items:
                # --- ตรวจสอบ line crossing ใน NvDsAnalyticsObjInfo ---
                l_user = object_meta.obj_user_meta_list
                while l_user:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                        analytics = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)

                        # lcStatus: list/dict ของชื่อเส้นที่ object นี้เพิ่งข้าม
                        lc_status = analytics.lcStatus
                        if lc_status:
                            # ตรวจว่าข้าม line ที่เราสนใจ (หรือถ้า line_name ว่างคือจับทุกเส้น)
                            matched = (not self.line_name) or (self.line_name in lc_status)
                            if matched:
                                obj_id = object_meta.object_id
                                last_t = self._last_captured.get(obj_id, 0.0)
                                if now - last_t >= self.cooldown_sec:
                                    self._last_captured[obj_id] = now
                                    print(f"[SKU Probe] Crossing detected: obj_id={obj_id} "
                                          f"label={object_meta.obj_label} "
                                          f"lcStatus={lc_status}", flush=True)
                                    self._try_capture(frame_meta, object_meta)

                    except StopIteration:
                        break
                    except Exception:
                        pass

                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                # ---- OSD: ระบายสีแต่ละ object ให้ต่างกัน ----
                obj_id = object_meta.object_id
                color = self._rgb_colors[obj_id % 1000]
                object_meta.rect_params.border_color = osd.Color(color[0], color[1], color[2], 1.0)
                object_meta.rect_params.border_width = 2

                text = object_meta.text_params
                cx = object_meta.rect_params.left + object_meta.rect_params.width / 2
                cy = object_meta.rect_params.top + object_meta.rect_params.height / 2
                text.display_text = f"{obj_id}".encode("ascii")
                text.x_offset = int(cx)
                text.y_offset = int(cy)
                object_meta.text_params = text


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
    print(f"[SKU Pipeline] Model: {'CUSTOM ('+CONFIG_INFER_CUSTOM+')' if USE_CUSTOM_MODEL else ('GDino' if USE_GDINO else 'TrafficCamNet')}")

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

    # nvdsanalytics — line crossing detection
    pipeline.add("nvdsanalytics", "analytics", {
        "config-file": CONFIG_NVDSANALYTICS,
    })

    # OSD
    if DISABLE_SOM_OVERLAY:
        pipeline.add("queue", "nvdsosd")
    else:
        pipeline.add("nvdsosd", "nvdsosd")

    # Sink
    pipeline.add("fakesink", "sink", {
        "sync": False if is_live else False,
        "qos": False,
    })

    # ---- Attach line crossing capture probe (หลัง analytics) ----
    capture_probe = LineCrossingCaptureProbe(
        output_dir=output_folder,
        line_name=line_name,
        cooldown_sec=cooldown_sec,
        save_full_frame=True,
        crop_padding=0.15,
    )
    pipeline.attach("analytics", Probe("sku_capture", capture_probe))

    # ---- Link pipeline ----
    pipeline.link(("decbin", "mux"), ("", "sink_%u"))
    pipeline.link(
        "mux", "inferserver", "queue1", "tracker",
        "queue2", "convert1", "analytics",
        "convert2", "nvdsosd", "queue3", "sink"
    )

    print("[SKU Pipeline] Starting...", flush=True)
    pipeline.start().wait()

    print(f"\n[SKU Pipeline] Done. Total captures: {capture_probe.capture_count}", flush=True)
    print(f"[SKU Pipeline] Images saved to: {output_folder}", flush=True)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
