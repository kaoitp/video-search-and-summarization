"""
SKU Recognition - Line Crossing Frame Capture GStreamer Plugin
=============================================================
GStreamer BaseTransform plugin that monitors nvdsanalytics line crossing events
and saves JPEG frames of objects crossing the line.

This plugin sits in the DeepStream pipeline after nvdsanalytics and captures
individual frames when objects cross the configured virtual line.

Pipeline position:
  ... → nvvideoconvert → nvdsanalytics → [gstskucapture] → nvdsosd → fakesink
"""

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase

import torch
import numpy as np
import cv2
import os
import pyds
import time
from datetime import datetime
from pathlib import Path

try:
    from pyservicemaker import Buffer
except ImportError:
    Buffer = None

Gst.init(None)

GST_PLUGIN_NAME = "skucapture"


class GstSkuCapture(GstBase.BaseTransform):
    """
    DeepStream GStreamer plugin for capturing frames at line crossing events.

    Reads NvDsAnalyticsObjInfo.lcStatus per object and NvDsAnalyticsFrameMeta
    for line crossing counts. When an object crosses the configured line,
    saves the full frame and a cropped region of the object to the output folder.
    """

    __gstmetadata__ = (
        "SKU Line Crossing Capture",
        "Transform",
        "Captures JPEG frames when objects cross the detection line",
        "SKU Recognition System",
    )

    _src_pad_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGB")
    )
    _sink_pad_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGB")
    )
    __gsttemplates__ = (_src_pad_template, _sink_pad_template)

    __gproperties__ = {
        "output-folder": (
            str, "Output Folder",
            "Directory where captured JPEG files are saved",
            "/tmp/sku/captures",
            GObject.ParamFlags.READWRITE,
        ),
        "line-name": (
            str, "Line Name",
            "Name of the line crossing to monitor (must match config_nvdsanalytics.txt)",
            "SKU_LINE",
            GObject.ParamFlags.READWRITE,
        ),
        "cooldown-sec": (
            float, "Cooldown Seconds",
            "Minimum seconds between captures for the same object ID",
            0.0, 60.0, 2.0,
            GObject.ParamFlags.READWRITE,
        ),
        "save-full-frame": (
            bool, "Save Full Frame",
            "Also save the full annotated frame in addition to the cropped object",
            True,
            GObject.ParamFlags.READWRITE,
        ),
        "crop-padding": (
            float, "Crop Padding",
            "Fractional padding around the bounding box when cropping (0.0 - 0.5)",
            0.0, 0.5, 0.15,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        GstBase.BaseTransform.__init__(self)
        self.output_folder = "/tmp/sku/captures"
        self.line_name = "SKU_LINE"
        self.cooldown_sec = 2.0
        self.save_full_frame = True
        self.crop_padding = 0.15

        self.width = None
        self.height = None
        self.capture_count = 0
        self.frame_count = 0
        # track_id -> last capture timestamp
        self._last_captured: dict[int, float] = {}
        self._cuda_stream = None

    def do_get_property(self, prop):
        if prop.name == "output-folder":
            return self.output_folder
        elif prop.name == "line-name":
            return self.line_name
        elif prop.name == "cooldown-sec":
            return self.cooldown_sec
        elif prop.name == "save-full-frame":
            return self.save_full_frame
        elif prop.name == "crop-padding":
            return self.crop_padding
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "output-folder":
            self.output_folder = value
        elif prop.name == "line-name":
            self.line_name = value
        elif prop.name == "cooldown-sec":
            self.cooldown_sec = value
        elif prop.name == "save-full-frame":
            self.save_full_frame = value
        elif prop.name == "crop-padding":
            self.crop_padding = value
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    def do_start(self):
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self._cuda_stream = torch.cuda.Stream(device=torch.device("cuda"))
        Gst.info(f"[SKUCapture] Started. Output: {self.output_folder} | Line: {self.line_name}")
        return True

    def do_stop(self):
        Gst.info(f"[SKUCapture] Stopped. Total captures: {self.capture_count}")
        return True

    def do_set_caps(self, incaps, outcaps):
        structure = incaps.get_structure(0)
        self.width = structure.get_int("width").value
        self.height = structure.get_int("height").value
        Gst.info(f"[SKUCapture] Caps set: {self.width}x{self.height}")
        return True

    # ------------------------------------------------------------------ #
    #  Frame helpers
    # ------------------------------------------------------------------ #

    def _extract_frame_numpy(self, gst_buffer, batch_id: int):
        """Extract a single frame as a numpy array (RGB, HWC) using pyservicemaker Buffer."""
        if Buffer is None:
            return None
        try:
            buf = Buffer(gst_buffer)
            frame_tensor_dl = buf.extract(batch_id)
            frame_tensor = torch.utils.dlpack.from_dlpack(frame_tensor_dl)
            # Move to CPU synchronously
            with torch.cuda.stream(self._cuda_stream):
                frame_cpu = frame_tensor.contiguous().to("cpu", non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._cuda_stream)
            return frame_cpu.numpy()  # shape: (H, W, 3), RGB
        except Exception as e:
            Gst.warning(f"[SKUCapture] Frame extraction failed: {e}")
            return None

    def _crop_object(self, frame_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
        """Crop object region with padding."""
        left, top, width, height = bbox
        pad_x = int(width * self.crop_padding)
        pad_y = int(height * self.crop_padding)
        fh, fw = frame_rgb.shape[:2]
        x1 = max(0, int(left) - pad_x)
        y1 = max(0, int(top) - pad_y)
        x2 = min(fw, int(left + width) + pad_x)
        y2 = min(fh, int(top + height) + pad_y)
        return frame_rgb[y1:y2, x1:x2]

    def _save_jpeg(self, name_base: str, img_rgb: np.ndarray):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.output_folder, name_base + ".jpg")
        cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    def _save_capture(self, gst_buffer, frame_meta, obj_meta):
        """Save crop + (optionally) full frame for a crossing object."""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        obj_id = obj_meta.object_id
        label = obj_meta.obj_label or "obj"

        frame_rgb = self._extract_frame_numpy(gst_buffer, frame_meta.batch_id)
        if frame_rgb is None:
            return

        bbox = (
            obj_meta.rect_params.left,
            obj_meta.rect_params.top,
            obj_meta.rect_params.width,
            obj_meta.rect_params.height,
        )

        # Save cropped object (main input for VLM SKU recognition)
        crop = self._crop_object(frame_rgb, bbox)
        if crop.size > 0:
            crop_path = self._save_jpeg(f"sku_{now_str}_id{obj_id}_{label}_crop", crop)
            Gst.info(f"[SKUCapture] Crop saved: {crop_path}")
            print(f"[SKUCapture] LINE CROSSING captured → {crop_path}", flush=True)

        # Save full frame with annotation
        if self.save_full_frame:
            annotated = frame_rgb.copy()
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(annotated, f"ID:{obj_id} {label}", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            full_path = self._save_jpeg(f"sku_{now_str}_id{obj_id}_{label}_full", annotated)
            Gst.info(f"[SKUCapture] Full frame saved: {full_path}")

        self.capture_count += 1

    # ------------------------------------------------------------------ #
    #  Main processing
    # ------------------------------------------------------------------ #

    def do_transform_ip(self, gst_buffer: Gst.Buffer) -> Gst.FlowReturn:
        self.frame_count += 1

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        except Exception as e:
            Gst.warning(f"[SKUCapture] Cannot get batch meta: {e}")
            return Gst.FlowReturn.OK

        now = time.time()

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # ---- Check each object for line crossing ----
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Read NvDsAnalyticsObjInfo attached to this object
                l_user = obj_meta.obj_user_meta_list
                while l_user:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                        analytics_info = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)

                        # lcStatus is a list of line names this object crossed this frame
                        if analytics_info.lcStatus:
                            # Filter by configured line name (empty line_name = capture all crossings)
                            matched = (not self.line_name) or (self.line_name in analytics_info.lcStatus)
                            if matched:
                                obj_id = obj_meta.object_id
                                last = self._last_captured.get(obj_id, 0.0)
                                if now - last >= self.cooldown_sec:
                                    self._last_captured[obj_id] = now
                                    self._save_capture(gst_buffer, frame_meta, obj_meta)

                    except StopIteration:
                        break
                    except Exception:
                        pass

                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.FlowReturn.OK


# ------------------------------------------------------------------ #
#  Plugin registration
# ------------------------------------------------------------------ #

def plugin_init(plugin):
    GObject.type_register(GstSkuCapture)
    if not Gst.Element.register(plugin, GST_PLUGIN_NAME, 0, GstSkuCapture):
        return False
    return True


GST_PLUGIN_DESC = Gst.PluginDesc(
    major_version=Gst.VERSION_MAJOR,
    minor_version=Gst.VERSION_MINOR,
    name=GST_PLUGIN_NAME,
    description="SKU Line Crossing Frame Capture Plugin",
    init=plugin_init,
    version="1.0.0",
    license="Proprietary",
    source="sku-recognition",
    package="sku-recognition",
    origin="",
)

# Allow running as a Python GStreamer plugin
if __name__ == "__main__":
    __gstelementfactory__ = (GST_PLUGIN_NAME, 0, GstSkuCapture)
