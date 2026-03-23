######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
######################################################################################################
import gradio as gr
import requests
import os
import json
import uuid
import threading
import time
import subprocess
import pandas as pd
from datetime import datetime, timezone

API_URL = os.environ.get("NV_CV_EVENT_DETECTOR_API_URL", "http://localhost:23491")
ALERTBRIDGE_URL = os.environ.get("NV_ALERTBRIDGE_URL", "http://alert-bridge:9080")
DEFAULT_OUTPUT_FOLDER = os.environ.get("ALERT_REVIEW_MEDIA_BASE_DIR", "/tmp/cv-output")

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(path):
    return requests.get(f"{API_URL}{path}", timeout=10)

def _post(path, payload):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=30)

def _delete(path, payload=None):
    return requests.delete(f"{API_URL}{path}", json=payload, timeout=10)

# ---------------------------------------------------------------------------
# Background watcher state  (stream_id → watcher info dict)
# ---------------------------------------------------------------------------

_watchers: dict[str, dict] = {}
_watchers_lock = threading.Lock()

# local cache: pipeline_id → name  (populated when pipeline is created via this UI)
_pipeline_names: dict[str, str] = {}

ALERT_ENDPOINT = "/api/v1/alerts"


def get_hw_status():
    parts = []
    # GPU via nvidia-smi — try PATH then common container locations
    _smi = next(
        (p for p in ["nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"]
         if os.path.isfile(p) or subprocess.run(["which", p], capture_output=True).returncode == 0),
        None,
    )
    try:
        if not _smi:
            raise FileNotFoundError("nvidia-smi not found — add 'runtime: nvidia' to cv-ui in compose file")
        res = subprocess.run(
            [_smi,
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in res.stdout.strip().splitlines():
            idx, name, mu, mt, util, temp = [x.strip() for x in line.split(",")]
            pct = f"{int(mu)*100//max(int(mt),1)}%"
            parts.append(f"GPU{idx} [{name}]  {mu}/{mt} MB ({pct})  util {util}%  {temp}°C")
    except Exception as e:
        parts.append(f"GPU: N/A ({e})")
    # Watchers
    with _watchers_lock:
        n = len(_watchers)
    if n:
        parts.append(f"🤖 {n} watcher active")
    return "   |   ".join(parts) if parts else "Hardware info unavailable"

def _build_event_payload(video_path, sensor_id, stream_name,
                          prompt, system_prompt,
                          event_type, severity,
                          chunk_duration, num_frames, enable_reasoning,
                          do_verification):
    cv_meta = video_path.replace(".mp4", ".json")
    return {
        "id": str(uuid.uuid4()),
        "version": "1.0",
        "@timestamp": (lambda n: n.strftime("%Y-%m-%dT%H:%M:%S.") + f"{n.microsecond // 1000:03d}Z")(datetime.now(timezone.utc)),
        "sensor_id": sensor_id or "sensor-1",
        "stream_name": stream_name or None,
        "video_path": video_path,
        "cv_metadata_path": cv_meta if os.path.exists(cv_meta) else None,
        "alert": {
            "severity": severity.upper(),
            "status": "REVIEW_PENDING",
            "type": event_type or "event",
            "description": prompt,
        },
        "event": {
            "type": event_type or "event",
            "description": prompt,
        },
        "vss_params": {
            "chunk_duration": int(chunk_duration),
            "chunk_overlap_duration": 3,
            "cv_metadata_overlay": True,
            "num_frames_per_chunk": int(num_frames),
            "enable_reasoning": bool(enable_reasoning),
            "do_verification": bool(do_verification),
            "debug": False,
            "vlm_params": {
                "prompt": prompt,
                "system_prompt": system_prompt if system_prompt and system_prompt.strip() else "You are a helpful assistant. Answer the user's question. Answer in yes or no only.",
            },
        },
        "confidence": 1.0,
        "meta_labels": [],
    }


def _watcher_loop(stream_id: str, output_folder: str, sensor_id: str,
                  stream_name: str, prompt: str, system_prompt: str,
                  event_type: str, severity: str,
                  chunk_duration: int, num_frames: int, enable_reasoning: bool,
                  do_verification: bool,
                  stop_event: threading.Event, poll_sec: int = 5):
    """Background thread: poll info.txt, auto-submit new clips to alert-bridge."""
    submitted: set[str] = set()
    info_path = os.path.join(output_folder, "info.txt")
    _log(stream_id, f"Watcher started — watching {info_path} every {poll_sec}s")

    while not stop_event.wait(poll_sec):
        try:
            if not os.path.exists(info_path):
                continue
            with open(info_path, "r") as f:
                clips = [l.strip() for l in f if l.strip()]

            for clip in clips:
                if clip in submitted:
                    continue
                video_path = os.path.join(output_folder, clip + ".mp4")
                if not os.path.exists(video_path):
                    continue  # still writing
                payload = _build_event_payload(
                    video_path, sensor_id, stream_name,
                    prompt, system_prompt,
                    event_type, severity,
                    chunk_duration, num_frames, enable_reasoning,
                    do_verification,
                )
                try:
                    r = requests.post(f"{ALERTBRIDGE_URL}{ALERT_ENDPOINT}",
                                      json=payload, timeout=30)
                    status = f"HTTP {r.status_code}"
                    ok = r.status_code in (200, 201, 202)
                except Exception as e:
                    status = str(e)
                    ok = False

                submitted.add(clip)
                icon = "✅" if ok else "❌"
                _log(stream_id, f"{icon} {clip}.mp4 → {status}")

        except Exception as e:
            _log(stream_id, f"⚠️ watcher error: {e}")

    _log(stream_id, "Watcher stopped")


def _log(stream_id: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(f"[watcher/{stream_id[:8]}] {line}")
    with _watchers_lock:
        if stream_id in _watchers:
            _watchers[stream_id]["log"].append(line)
            # keep last 200 lines
            _watchers[stream_id]["log"] = _watchers[stream_id]["log"][-200:]


def start_watcher(stream_id: str, output_folder: str, sensor_id: str,
                  stream_name: str, prompt: str, system_prompt: str,
                  event_type: str, severity: str,
                  chunk_duration: int, num_frames: int, enable_reasoning: bool,
                  do_verification: bool,
                  poll_sec: int = 5):
    stop_event = threading.Event()
    t = threading.Thread(
        target=_watcher_loop,
        args=(stream_id, output_folder, sensor_id, stream_name,
              prompt, system_prompt, event_type, severity,
              chunk_duration, num_frames, enable_reasoning,
              do_verification,
              stop_event, poll_sec),
        daemon=True,
        name=f"watcher-{stream_id[:8]}",
    )
    with _watchers_lock:
        _watchers[stream_id] = {
            "thread": t,
            "stop_event": stop_event,
            "output_folder": output_folder,
            "prompt": prompt,
            "log": [],
        }
    t.start()


def stop_watcher(stream_id: str):
    with _watchers_lock:
        entry = _watchers.pop(stream_id, None)
    if entry:
        entry["stop_event"].set()


def get_watcher_log(stream_id: str) -> str:
    sid = stream_id.strip() if stream_id else ""
    if not sid:
        return "กรุณาเลือก Stream ID"
    with _watchers_lock:
        entry = _watchers.get(sid)
    if not entry:
        return f"ไม่มี watcher สำหรับ stream {sid[:12]}..."
    lines = entry["log"]
    return "\n".join(lines[-50:]) if lines else "(ยังไม่มี log)"


def get_watcher_status_table():
    with _watchers_lock:
        rows = []
        for sid, entry in _watchers.items():
            alive = "🟢 running" if entry["thread"].is_alive() else "🔴 stopped"
            folder = entry.get("output_folder", "")
            prompt_preview = (entry.get("prompt") or "")[:40]
            rows.append([sid, alive, folder, prompt_preview])
    return rows

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def create_pipeline(name, endpoint_url, pipeline_type,
                    min_clip, max_clip, frame_skip, min_detect):
    payload = {
        "name": name.strip(),
        "endpoint_url": endpoint_url.strip() if endpoint_url.strip() else None,
        "type": pipeline_type.strip(),
        "params": {
            "min_clip_duration": int(min_clip),
            "max_clip_duration": int(max_clip),
            "frame_skip_interval": int(frame_skip),
            "minimum_detection_threshold": int(min_detect),
        },
    }
    try:
        r = _post("/api/pipeline", payload)
        data = r.json()
        if data.get("status") == "success":
            pid = data["id"]
            _pipeline_names[pid] = name.strip()
            choices, default = _pipeline_choices()
            return (
                f"✅ Pipeline สร้างสำเร็จ\nID: {pid}\nชื่อ: {name.strip()}",
                gr.update(choices=choices, value=pid),
            )
        return f"❌ {data.get('message', r.text)}", gr.update()
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}", gr.update()


def delete_pipeline(pipeline_id):
    if not pipeline_id:
        return "❌ กรุณาระบุ Pipeline ID", gr.update()
    # extract raw id if in "name · id" format
    pid = pipeline_id.split(" · ")[-1].strip() if " · " in pipeline_id else pipeline_id.strip()
    try:
        r = _delete("/api/pipeline", {"id": pid, "cleanup_resources": True})
        data = r.json()
        if data.get("status") == "success":
            _pipeline_names.pop(pid, None)
        msg = "✅ ลบ Pipeline สำเร็จ" if data.get("status") == "success" else f"❌ {data.get('message', r.text)}"
        choices, default = _pipeline_choices()
        return msg, gr.update(choices=choices, value=default)
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}", gr.update()


def _pipeline_choices():
    try:
        data = _get("/api/pipelines").json()
        pipelines = data.get("pipelines", [])
        choices = []
        for p in pipelines:
            pid = p["id"]
            name = _pipeline_names.get(pid, "")
            label = f"{name} · {pid}" if name else pid
            choices.append(label)
        return choices, (choices[0] if choices else None)
    except Exception:
        return [], None


def _extract_pipeline_id(label: str) -> str:
    """Extract raw pipeline ID from a dropdown label like 'name · id'."""
    return label.split(" · ")[-1].strip() if label else ""


def refresh_pipeline_dropdown():
    choices, default = _pipeline_choices()
    return gr.update(choices=choices, value=default)


def get_pipelines_table():
    try:
        data = _get("/api/pipelines").json()
        rows = []
        for p in data.get("pipelines", []):
            pid = p["id"]
            name = _pipeline_names.get(pid, "—")
            rows.append([name, pid, p.get("created_at", "")[:19]])
        return rows
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Combined setup: create pipeline + add all streams in one call
# ---------------------------------------------------------------------------

def setup_pipeline_and_streams(
    name, endpoint_url, pipeline_type,
    min_clip, max_clip, frame_skip, min_detect,
    streams_df,
    detection_classes, box_threshold,
    roi_x, roi_y, roi_w, roi_h,
    auto_review, poll_sec,
    vlm_prompt, vlm_system_prompt,
    event_type, severity,
    chunk_duration, num_frames, enable_reasoning, do_verification,
):
    # Step 1: Create pipeline
    pipeline_payload = {
        "name": name.strip(),
        "endpoint_url": endpoint_url.strip() if endpoint_url.strip() else None,
        "type": pipeline_type.strip(),
        "params": {
            "min_clip_duration": int(min_clip),
            "max_clip_duration": int(max_clip),
            "frame_skip_interval": int(frame_skip),
            "minimum_detection_threshold": int(min_detect),
        },
    }
    try:
        r = _post("/api/pipeline", pipeline_payload)
        data = r.json()
        if data.get("status") != "success":
            return f"❌ สร้าง Pipeline ล้มเหลว: {data.get('message', r.text)}"
        pipeline_id = data["id"]
    except Exception as e:
        return f"❌ สร้าง Pipeline ล้มเหลว: {e}"

    lines = [f"✅ Pipeline สร้างสำเร็จ\nID: {pipeline_id}\n"]

    # Step 2: Shared CV params
    classes = [c.strip() for c in detection_classes.strip().splitlines() if c.strip()]
    cv_prompt = " . ".join(classes) if classes else None
    has_roi = int(roi_w) > 0 and int(roi_h) > 0
    gdino_rois = [[int(roi_x), int(roi_y), int(roi_w), int(roi_h)]] if has_roi else [[]]

    # Step 3: Add each stream row
    rows_iter = [dict(row) for _, row in streams_df.iterrows()] if isinstance(streams_df, pd.DataFrame) else streams_df
    added = 0
    for row in rows_iter:
        stream_url = str(row.get("Stream URL", "")).strip()
        if not stream_url:
            continue
        stream_name = str(row.get("Stream Name", "stream")).strip() or "stream"
        sensor_id   = str(row.get("Sensor ID",   "sensor-1")).strip() or "sensor-1"

        safe_name        = stream_name.replace(" ", "_")
        ts_tag           = datetime.now().strftime("%Y%m%d_%H%M%S")
        stream_subfolder = os.path.join(DEFAULT_OUTPUT_FOLDER, f"{safe_name}_{ts_tag}")

        stream_payload = {
            "version": "1.0",
            "stream_url": stream_url,
            "pipeline_id": pipeline_id,
            "output_folder": stream_subfolder,
            "sensor_id": sensor_id,
            "stream_name": safe_name,
            "processing_state": "enabled",
            "cv_params": {
                "gdinoprompt": cv_prompt,
                "gdinothreshold": float(box_threshold),
                "gdino_rois": gdino_rois,
            },
        }
        try:
            r = _post("/api/addstream", stream_payload)
            d = r.json()
            if d.get("status") == "success":
                sid = d["stream_id"]
                watcher_note = ""
                if auto_review and vlm_prompt.strip():
                    start_watcher(
                        stream_id=sid,
                        output_folder=stream_subfolder,
                        sensor_id=sensor_id,
                        stream_name=safe_name,
                        prompt=vlm_prompt.strip(),
                        system_prompt=vlm_system_prompt.strip(),
                        event_type=event_type.strip() or "event",
                        severity=severity,
                        chunk_duration=int(chunk_duration),
                        num_frames=int(num_frames),
                        enable_reasoning=bool(enable_reasoning),
                        do_verification=bool(do_verification),
                        poll_sec=int(poll_sec),
                    )
                    watcher_note = "  🤖 Auto-Review เปิดแล้ว"
                elif auto_review:
                    watcher_note = "  ⚠️ Auto-Review: ต้องใส่ VLM Prompt"
                lines.append(f"  ✅ [{stream_name}] {sid}{watcher_note}")
                added += 1
            else:
                lines.append(f"  ❌ [{stream_name}] {d.get('message', r.text)}")
        except Exception as e:
            lines.append(f"  ❌ [{stream_name}] {e}")

    lines.append(f"\nสรุป: เพิ่ม {added} stream สำเร็จ")
    return "\n".join(lines)


def add_streams_to_pipeline(
    pipeline_label,
    streams_df,
    detection_classes, box_threshold,
    roi_x, roi_y, roi_w, roi_h,
    auto_review, poll_sec,
    vlm_prompt, vlm_system_prompt,
    event_type, severity,
    chunk_duration, num_frames, enable_reasoning, do_verification,
):
    pipeline_id = _extract_pipeline_id(pipeline_label) if pipeline_label else ""
    if not pipeline_id:
        return "❌ กรุณาเลือก Pipeline ก่อน"

    classes = [c.strip() for c in detection_classes.strip().splitlines() if c.strip()]
    cv_prompt = " . ".join(classes) if classes else None
    has_roi = int(roi_w) > 0 and int(roi_h) > 0
    gdino_rois = [[int(roi_x), int(roi_y), int(roi_w), int(roi_h)]] if has_roi else [[]]

    rows_iter = [dict(row) for _, row in streams_df.iterrows()] if isinstance(streams_df, pd.DataFrame) else streams_df
    lines = []
    added = 0
    for row in rows_iter:
        stream_url = str(row.get("Stream URL", "")).strip()
        if not stream_url:
            continue
        stream_name = str(row.get("Stream Name", "stream")).strip() or "stream"
        sensor_id   = str(row.get("Sensor ID",   "sensor-1")).strip() or "sensor-1"

        safe_name        = stream_name.replace(" ", "_")
        ts_tag           = datetime.now().strftime("%Y%m%d_%H%M%S")
        stream_subfolder = os.path.join(DEFAULT_OUTPUT_FOLDER, f"{safe_name}_{ts_tag}")

        payload = {
            "version": "1.0",
            "stream_url": stream_url,
            "pipeline_id": pipeline_id,
            "output_folder": stream_subfolder,
            "sensor_id": sensor_id,
            "stream_name": safe_name,
            "processing_state": "enabled",
            "cv_params": {
                "gdinoprompt": cv_prompt,
                "gdinothreshold": float(box_threshold),
                "gdino_rois": gdino_rois,
            },
        }
        try:
            r = _post("/api/addstream", payload)
            d = r.json()
            if d.get("status") == "success":
                sid = d["stream_id"]
                watcher_note = ""
                if auto_review and vlm_prompt.strip():
                    start_watcher(
                        stream_id=sid,
                        output_folder=stream_subfolder,
                        sensor_id=sensor_id,
                        stream_name=safe_name,
                        prompt=vlm_prompt.strip(),
                        system_prompt=vlm_system_prompt.strip(),
                        event_type=event_type.strip() or "event",
                        severity=severity,
                        chunk_duration=int(chunk_duration),
                        num_frames=int(num_frames),
                        enable_reasoning=bool(enable_reasoning),
                        do_verification=bool(do_verification),
                        poll_sec=int(poll_sec),
                    )
                    watcher_note = "  🤖"
                elif auto_review:
                    watcher_note = "  ⚠️ (ต้องใส่ VLM Prompt)"
                lines.append(f"  ✅ [{stream_name}]  {sid}{watcher_note}")
                added += 1
            else:
                lines.append(f"  ❌ [{stream_name}]  {d.get('message', r.text)}")
        except Exception as e:
            lines.append(f"  ❌ [{stream_name}]  {e}")

    lines.append(f"\nสรุป: เพิ่ม {added} stream สำเร็จ")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def add_stream(pipeline_id, stream_url, sensor_id, stream_name, output_folder,
               detection_classes, box_threshold,
               roi_x, roi_y, roi_w, roi_h,
               vlm_prompt, vlm_system_prompt,
               event_type, severity,
               chunk_duration, num_frames, enable_reasoning, do_verification,
               poll_sec, auto_review):
    if not pipeline_id:
        return "❌ กรุณาเลือก Pipeline ก่อน"
    if not stream_url.strip():
        return "❌ กรุณาระบุ Stream URL"

    # Build subfolder per stream so info.txt files don't collide
    safe_name = (stream_name.strip() or "stream").replace(" ", "_")
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stream_subfolder = os.path.join(output_folder.strip(), f"{safe_name}_{ts_tag}")

    classes = [c.strip() for c in detection_classes.strip().splitlines() if c.strip()]
    prompt = " . ".join(classes) if classes else None

    has_roi = int(roi_w) > 0 and int(roi_h) > 0
    gdino_rois = [[int(roi_x), int(roi_y), int(roi_w), int(roi_h)]] if has_roi else [[]]

    payload = {
        "version": "1.0",
        "stream_url": stream_url.strip(),
        "pipeline_id": pipeline_id,
        "output_folder": stream_subfolder,
        "sensor_id": sensor_id.strip() if sensor_id.strip() else None,
        "stream_name": safe_name,
        "processing_state": "enabled",
        "cv_params": {
            "gdinoprompt": prompt,
            "gdinothreshold": float(box_threshold),
            "gdino_rois": gdino_rois,
        },
    }
    try:
        r = _post("/api/addstream", payload)
        data = r.json()
        if data.get("status") != "success":
            return f"❌ {data.get('message', r.text)}"

        sid = data["stream_id"]
        state = data.get("processing_state", "")

        # Start auto-review watcher if enabled and prompt is provided
        watcher_note = ""
        if auto_review and vlm_prompt.strip():
            start_watcher(
                stream_id=sid,
                output_folder=stream_subfolder,
                sensor_id=sensor_id.strip() or "sensor-1",
                stream_name=safe_name,
                prompt=vlm_prompt.strip(),
                system_prompt=vlm_system_prompt.strip(),
                event_type=event_type.strip() or "event",
                severity=severity,
                chunk_duration=int(chunk_duration),
                num_frames=int(num_frames),
                enable_reasoning=bool(enable_reasoning),
                do_verification=bool(do_verification),
                poll_sec=int(poll_sec),
            )
            watcher_note = f"\n🤖 Auto-Review: เปิดแล้ว (poll ทุก {int(poll_sec)} วิ)"
        elif auto_review and not vlm_prompt.strip():
            watcher_note = "\n⚠️ Auto-Review: ไม่ได้เปิด (กรุณาใส่ VLM Prompt)"

        return (
            f"✅ เพิ่ม Stream สำเร็จ\n"
            f"Stream ID    : {sid}\n"
            f"State        : {state}\n"
            f"Output Folder: {stream_subfolder}"
            f"{watcher_note}"
        )
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"


def get_streams_table():
    try:
        data = _get("/api/streams").json()
        rows = []
        with _watchers_lock:
            watcher_ids = set(_watchers.keys())
        for s in data.get("streams", []):
            sid = s["stream_id"]
            pid = s.get("pipeline_id", "")
            pipe_name = _pipeline_names.get(pid, "—")
            stream_name = s.get("stream_name", "—")
            auto = "🤖" if sid in watcher_ids else ""
            rows.append([
                stream_name,
                sid[:12] + "...",
                pipe_name,
                s.get("processing_state", ""),
                s.get("timestamp", "")[:19],
                auto,
            ])
        return rows
    except Exception:
        return []


def get_stream_id_choices():
    try:
        data = _get("/api/streams").json()
        streams = data.get("streams", [])
        with _watchers_lock:
            watcher_ids = set(_watchers.keys())
        choices = []
        for s in streams:
            sid = s["stream_id"]
            sname = s.get("stream_name", "")
            label = f"{sname}  {sid}"  if sname else sid
            label += f"  [{s.get('processing_state', '')}]"
            if sid in watcher_ids:
                label += "  🤖"
            choices.append(label)
        return gr.update(choices=choices, value=None)
    except Exception:
        return gr.update(choices=[], value=None)


def streams_table_and_dropdown():
    return get_streams_table(), get_stream_id_choices()


def pick_stream_id(choice):
    if not choice:
        return ""
    return choice.split("  ")[0].strip()


def stop_stream(stream_id):
    sid = stream_id.strip() if stream_id else ""
    if not sid:
        return "❌ กรุณาระบุ Stream ID"
    stop_watcher(sid)
    try:
        r = requests.delete(f"{API_URL}/api/stream",
                            json={"stream_id": sid, "version": "1.0"},
                            timeout=60)
        data = r.json()
        if data.get("status") == "success":
            return "✅ หยุด Stream และ watcher สำเร็จ"
        return f"❌ {data.get('message', r.text)}"
    except requests.exceptions.ReadTimeout:
        # Server is still terminating the DeepStream process — verify by checking stream list
        try:
            streams = _get("/api/streams").json().get("streams", [])
            still_exists = any(s["stream_id"] == sid for s in streams)
            if not still_exists:
                return "✅ หยุด Stream สำเร็จ (server ใช้เวลา terminate process นานกว่าปกติ)"
            return "⚠️ Timeout — กรุณากด Refresh เพื่อตรวจสอบสถานะ stream"
        except Exception:
            return "⚠️ Timeout — กรุณากด Refresh เพื่อตรวจสอบสถานะ stream"
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"


def get_stream_status(stream_id, wait_ms):
    sid = stream_id.strip() if stream_id else ""
    if not sid:
        return "❌ กรุณาระบุ Stream ID"
    try:
        r = requests.get(
            f"{API_URL}/api/streams/{sid}/status",
            params={"timeout_ms": int(wait_ms)},
            timeout=int(wait_ms) / 1000 + 5,
        )
        data = r.json()
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"


def health_check():
    parts = []
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        status = r.json().get("status", "unknown")
        parts.append(f"{'🟢' if status == 'healthy' else '🔴'} CV Detector: {status}")
    except Exception as e:
        parts.append(f"🔴 CV Detector: {e}")
    try:
        r = requests.get(f"{ALERTBRIDGE_URL}/health", timeout=5)
        ct = r.headers.get("content-type", "")
        status = r.json().get("status", "ok") if "json" in ct else "ok"
        parts.append(f"🟢 Alert-Bridge: {status}")
    except Exception as e:
        parts.append(f"🔴 Alert-Bridge: {e}")
    with _watchers_lock:
        n = len(_watchers)
    if n:
        parts.append(f"🤖 Watchers: {n} active")
    return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Manual VLM Review helpers (Tab 4)
# ---------------------------------------------------------------------------


def list_clips(output_folder: str):
    """Scan base folder and all subfolders for info.txt, return all clips found."""
    if not output_folder.strip():
        return [], gr.update(choices=[]), "❌ กรุณาระบุ Output Folder"
    base = output_folder.strip()
    if not os.path.isdir(base):
        return [], gr.update(choices=[]), f"❌ ไม่พบ folder: {base}"

    rows, choices, folder_count = [], [], set()
    for dirpath, _, filenames in os.walk(base):
        if "info.txt" not in filenames:
            continue
        try:
            with open(os.path.join(dirpath, "info.txt")) as f:
                clips = [l.strip() for l in f if l.strip()]
        except Exception:
            continue
        rel_dir = os.path.relpath(dirpath, base)
        for c in clips:
            full_mp4 = os.path.join(dirpath, c + ".mp4")
            if not os.path.exists(full_mp4):
                continue
            display = os.path.join(rel_dir, c + ".mp4") if rel_dir != "." else c + ".mp4"
            rows.append([display, rel_dir if rel_dir != "." else "(root)"])
            choices.append(full_mp4)   # store full path as value
            folder_count.add(dirpath)

    if not rows:
        return [], gr.update(choices=[]), f"ไม่พบ clips ใน {base} หรือ subfolders"
    return rows, gr.update(choices=choices), f"พบ {len(rows)} clips จาก {len(folder_count)} folder"


def submit_clip_to_alertbridge(video_path, sensor_id, stream_name,
                                prompt, system_prompt,
                                event_type, severity,
                                chunk_duration, num_frames, enable_reasoning,
                                do_verification):
    if not video_path:
        return "❌ กรุณาเลือก clip ก่อน"
    if not prompt.strip():
        return "❌ กรุณาใส่ VLM Prompt ก่อน"
    payload = _build_event_payload(video_path, sensor_id, stream_name,
                                   prompt, system_prompt,
                                   event_type, severity,
                                   chunk_duration, num_frames, enable_reasoning,
                                   do_verification)
    try:
        r = requests.post(f"{ALERTBRIDGE_URL}{ALERT_ENDPOINT}", json=payload, timeout=30)
        ok = r.status_code in (200, 201, 202)
        return f"{'✅' if ok else '❌'} HTTP {r.status_code}\n{r.text[:300]}"
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"


def submit_uploaded_clip(uploaded_file, sensor_id, stream_name,
                          prompt, system_prompt,
                          event_type, severity,
                          chunk_duration, num_frames, enable_reasoning,
                          do_verification):
    if not uploaded_file:
        return "❌ กรุณา upload clip ก่อน"
    if not prompt.strip():
        return "❌ กรุณาใส่ VLM Prompt ก่อน"
    video_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
    payload = _build_event_payload(video_path, sensor_id, stream_name,
                                   prompt, system_prompt,
                                   event_type, severity,
                                   chunk_duration, num_frames, enable_reasoning,
                                   do_verification)
    try:
        r = requests.post(f"{ALERTBRIDGE_URL}{ALERT_ENDPOINT}", json=payload, timeout=30)
        ok = r.status_code in (200, 201, 202)
        return f"{'✅' if ok else '❌'} HTTP {r.status_code}\n{r.text[:300]}"
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"


def submit_all_clips(output_folder, sensor_id, stream_name,
                     prompt, system_prompt, event_type, severity,
                     chunk_duration, num_frames, enable_reasoning, do_verification):
    if not prompt.strip():
        return "❌ กรุณาใส่ VLM Prompt ก่อน"
    base = output_folder.strip()
    all_videos = []
    for dirpath, _, filenames in os.walk(base):
        if "info.txt" not in filenames:
            continue
        try:
            with open(os.path.join(dirpath, "info.txt")) as f:
                clips = [l.strip() for l in f if l.strip()]
        except Exception:
            continue
        for c in clips:
            full_mp4 = os.path.join(dirpath, c + ".mp4")
            if os.path.exists(full_mp4):
                all_videos.append(full_mp4)
    if not all_videos:
        return f"❌ ไม่พบ clips ใน {base}"
    results = []
    for video_path in all_videos:
        payload = _build_event_payload(video_path, sensor_id, stream_name,
                                       prompt, system_prompt,
                                       event_type, severity,
                                       chunk_duration, num_frames, enable_reasoning,
                                       do_verification)
        try:
            r = requests.post(f"{ALERTBRIDGE_URL}{ALERT_ENDPOINT}", json=payload, timeout=30)
            icon = "✅" if r.status_code in (200, 201, 202) else "❌"
            results.append(f"{icon} {c}.mp4 → HTTP {r.status_code}")
        except Exception as e:
            results.append(f"❌ {c}.mp4 → {e}")
    return "\n".join(results)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.status-box textarea { font-family: monospace; font-size: 0.85rem; }
.stream-table { font-size: 0.85rem; }
.hw-box textarea { font-family: monospace; font-size: 0.8rem; background: #1a1a2e; color: #00ff88; }
footer { display: none !important; }
"""

with gr.Blocks(title="CV Event Detector UI") as demo:

    gr.Markdown("# CV Event Detector — Pipeline & Stream Manager")

    # ── Service health + Hardware monitoring ──────────────────────────────────
    with gr.Row():
        health_txt = gr.Textbox(label="Service Status", interactive=False,
                                value=health_check, every=30, scale=3)
        btn_health = gr.Button("🔄", scale=0, min_width=60)
    with gr.Row():
        hw_txt = gr.Textbox(label="Hardware (GPU)", interactive=False,
                            value=get_hw_status, every=10, scale=4,
                            elem_classes="hw-box")
        btn_hw = gr.Button("🔄", scale=0, min_width=60)
    btn_health.click(health_check, outputs=health_txt)
    btn_hw.click(get_hw_status, outputs=hw_txt)

    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Tabs():

        # ── TAB 1: Pipeline ───────────────────────────────────────────────────
        with gr.TabItem("⚙️ Pipeline"):
            gr.Markdown("### สร้าง Pipeline ใหม่")
            with gr.Row():
                with gr.Column():
                    p_name      = gr.Textbox(label="ชื่อ Pipeline", value="cv-pipeline")
                    p_type      = gr.Textbox(label="ประเภท (type)", value="object_detection")
                    p_endpoint  = gr.Textbox(label="Alert Bridge URL", value=ALERTBRIDGE_URL)
                with gr.Column():
                    p_min_clip   = gr.Number(label="Min Clip Duration (วิ)", value=5, minimum=1)
                    p_max_clip   = gr.Number(label="Max Clip Duration (วิ)", value=60, minimum=1)
                    p_fskip      = gr.Slider(label="Frame Skip Interval", minimum=0, maximum=10, step=1, value=0)
                    p_min_detect = gr.Slider(label="Min Object Detection Threshold", minimum=1, maximum=50, step=1, value=3)

            btn_create_pipe = gr.Button("➕ สร้าง Pipeline", variant="primary")
            p_result = gr.Textbox(label="ผลลัพธ์", lines=3, elem_classes="status-box")
            _pipe_dd_state = gr.State(value=None)

            gr.Markdown("---")
            gr.Markdown("### Pipeline ที่มีอยู่")
            with gr.Row():
                btn_refresh_pipes = gr.Button("🔄 โหลดใหม่")
                p_del_dd = gr.Dropdown(label="เลือก Pipeline ที่จะลบ",
                                       choices=_pipeline_choices()[0],
                                       allow_custom_value=True, scale=3)
                btn_del_pipe = gr.Button("🗑️ ลบ", variant="stop")
            p_pipe_table = gr.Dataframe(
                headers=["ชื่อ", "Pipeline ID", "Created At"],
                datatype=["str", "str", "str"],
                elem_classes="stream-table",
                interactive=False,
                value=get_pipelines_table,
                every=60,
            )
            p_del_result = gr.Textbox(label="ผลการลบ", lines=2, elem_classes="status-box")

            btn_create_pipe.click(
                create_pipeline,
                inputs=[p_name, p_endpoint, p_type,
                        p_min_clip, p_max_clip, p_fskip, p_min_detect],
                outputs=[p_result, _pipe_dd_state],
            ).then(get_pipelines_table, outputs=p_pipe_table
            ).then(lambda: gr.update(choices=_pipeline_choices()[0]), outputs=p_del_dd)

            btn_refresh_pipes.click(get_pipelines_table, outputs=p_pipe_table
            ).then(lambda: gr.update(choices=_pipeline_choices()[0]), outputs=p_del_dd)

            btn_del_pipe.click(
                delete_pipeline,
                inputs=[p_del_dd],
                outputs=[p_del_result, p_del_dd],
            ).then(get_pipelines_table, outputs=p_pipe_table)

        # ── TAB 2: เพิ่ม Streams ──────────────────────────────────────────────
        with gr.TabItem("📷 เพิ่ม Streams"):
            gr.Markdown("เลือก Pipeline แล้วระบุ stream ทุกกล้องในตาราง — กดปุ่มเดียวเพิ่มทั้งหมด")
            with gr.Row():
                s_pipeline = gr.Dropdown(
                    label="Pipeline",
                    choices=_pipeline_choices()[0],
                    allow_custom_value=True, scale=3,
                )
                btn_s_refresh_pipe = gr.Button("🔄", scale=0, min_width=60)

            with gr.Row():
                # ── left: streams table + CV ──────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("#### 📷 รายการ Streams")
                    s_streams_df = gr.Dataframe(
                        headers=["Stream URL", "Stream Name", "Sensor ID"],
                        datatype=["str", "str", "str"],
                        row_count=(3, "dynamic"),
                        col_count=(3, "fixed"),
                        interactive=True,
                        label="ระบุ stream (เพิ่มแถวได้ตามต้องการ)",
                        value=[
                            ["", "cam-1", "sensor-1"],
                            ["", "cam-2", "sensor-2"],
                            ["", "cam-3", "sensor-3"],
                        ],
                    )
                    gr.Markdown("#### 🔍 CV Detection (ใช้ร่วมกันทุก stream)")
                    s_classes = gr.Textbox(
                        label="Detection Classes (หนึ่งคลาสต่อบรรทัด)",
                        lines=3, placeholder="person\ncar\ntruck", value="person",
                    )
                    s_threshold = gr.Slider(label="Box Threshold", minimum=0.1, maximum=1.0, step=0.05, value=0.3)
                    gr.Markdown("#### ROI — เว้นว่างถ้าใช้ทั้งภาพ")
                    with gr.Row():
                        s_roi_x = gr.Number(label="X", value=0, minimum=0, precision=0)
                        s_roi_y = gr.Number(label="Y", value=0, minimum=0, precision=0)
                        s_roi_w = gr.Number(label="W (0=ทั้งภาพ)", value=0, minimum=0, precision=0)
                        s_roi_h = gr.Number(label="H (0=ทั้งภาพ)", value=0, minimum=0, precision=0)

                # ── right: VLM / Auto-Review ──────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("#### 🤖 Auto VLM Review (ใช้ร่วมกันทุก stream)")
                    s_auto_review = gr.Checkbox(
                        label="เปิด Auto-Review (ส่ง clip ให้ Alert-Bridge อัตโนมัติ)", value=True)
                    s_poll_sec = gr.Slider(
                        label="Poll Interval (วินาที)", minimum=3, maximum=60, step=1, value=5)
                    s_vlm_prompt = gr.Textbox(
                        label="VLM Prompt *", lines=5,
                        placeholder="e.g. Does the video show signs of overcrowding?",
                    )
                    s_vlm_sys_prompt = gr.Textbox(
                        label="System Prompt (ไม่บังคับ)", lines=2,
                        placeholder="e.g. You are a warehouse safety monitoring system.",
                    )
                    gr.Markdown("#### Event Info")
                    with gr.Row():
                        s_event_type = gr.Textbox(label="Event Type", value="over_crowding", scale=2)
                        s_severity   = gr.Dropdown(
                            label="Severity", choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            value="MEDIUM", scale=1,
                        )
                    gr.Markdown("#### VSS Params")
                    with gr.Row():
                        s_chunk_dur   = gr.Number(label="Chunk Duration (วิ)", value=60, minimum=1, precision=0)
                        s_num_frames  = gr.Number(label="Frames per Chunk", value=8, minimum=1, precision=0)
                    with gr.Row():
                        s_reasoning       = gr.Checkbox(label="Enable Reasoning", value=False)
                        s_do_verification = gr.Checkbox(label="Do Verification", value=True)

            btn_add_streams = gr.Button("➕ เพิ่ม Streams ทั้งหมด", variant="primary", size="lg")
            s_result = gr.Textbox(label="ผลลัพธ์", lines=8, elem_classes="status-box")

            btn_s_refresh_pipe.click(
                lambda: gr.update(choices=_pipeline_choices()[0]),
                outputs=s_pipeline,
            )
            btn_add_streams.click(
                add_streams_to_pipeline,
                inputs=[
                    s_pipeline, s_streams_df,
                    s_classes, s_threshold,
                    s_roi_x, s_roi_y, s_roi_w, s_roi_h,
                    s_auto_review, s_poll_sec,
                    s_vlm_prompt, s_vlm_sys_prompt,
                    s_event_type, s_severity,
                    s_chunk_dur, s_num_frames, s_reasoning, s_do_verification,
                ],
                outputs=s_result,
            )

            # sync pipeline dropdown from Tab 1
            _pipe_dd_state.change(
                lambda v: gr.update(value=v) if v else gr.update(),
                inputs=_pipe_dd_state,
                outputs=s_pipeline,
            )

        # ── TAB 3: จัดการ Stream ─────────────────────────────────────────────
        with gr.TabItem("📋 จัดการ Stream"):
            gr.Markdown("### รายการ Stream ทั้งหมด  (🤖 = มี Auto-Review เปิดอยู่)")
            with gr.Row():
                btn_refresh_streams = gr.Button("🔄 โหลดใหม่", variant="secondary")

            m_stream_table = gr.Dataframe(
                headers=["Stream Name", "Stream ID", "Pipeline", "State", "Timestamp", "Auto"],
                datatype=["str", "str", "str", "str", "str", "str"],
                elem_classes="stream-table",
                interactive=False,
                value=get_streams_table,
                every=10,
            )

            gr.Markdown("---")
            gr.Markdown("### ตรวจสอบ / หยุด / ดู Watcher Log")

            m_stream_dd = gr.Dropdown(
                label="เลือก Stream",
                choices=[],
                allow_custom_value=False,
            )
            with gr.Row():
                m_stream_id = gr.Textbox(
                    label="Stream ID",
                    scale=4, interactive=True,
                    placeholder="เลือกจาก dropdown หรือพิมพ์โดยตรง",
                )
                m_wait_ms = gr.Number(
                    label="Wait (ms)", value=500, minimum=0,
                    maximum=30000, precision=0, scale=1,
                )
            with gr.Row():
                btn_m_status = gr.Button("🔍 ดูสถานะ")
                btn_m_wlog   = gr.Button("📜 ดู Watcher Log")
                btn_m_stop   = gr.Button("⏹️ หยุด Stream + Watcher", variant="stop")

            m_status_out = gr.Textbox(label="สถานะ / Watcher Log", lines=12, elem_classes="status-box")
            m_stop_out   = gr.Textbox(label="ผลการหยุด", lines=2, elem_classes="status-box")

            m_stream_dd.change(pick_stream_id, inputs=m_stream_dd, outputs=m_stream_id)
            btn_refresh_streams.click(streams_table_and_dropdown, outputs=[m_stream_table, m_stream_dd])
            btn_m_status.click(get_stream_status, inputs=[m_stream_id, m_wait_ms], outputs=m_status_out)
            btn_m_wlog.click(get_watcher_log, inputs=m_stream_id, outputs=m_status_out)
            btn_m_stop.click(stop_stream, inputs=[m_stream_id], outputs=m_stop_out).then(
                streams_table_and_dropdown, outputs=[m_stream_table, m_stream_dd]
            )

        # ── TAB 4: Manual VLM Review ──────────────────────────────────────────
        with gr.TabItem("📤 Manual VLM Review"):
            gr.Markdown(
                "ส่ง clips ไปให้ Alert-Bridge **ด้วยตัวเอง** "
                "(ใช้กรณีปิด Auto-Review หรือส่งซ้ำ)\n\n"
                f"`Alert-Bridge: {ALERTBRIDGE_URL}`"
            )
            with gr.Row():
                # ── left: clip selection ──────────────────────────────────────
                with gr.Column(scale=2):
                    with gr.Tab("📁 จาก Folder"):
                        gr.Markdown("สแกน folder และ subfolders ทั้งหมดที่มี `info.txt`")
                        t4_output_folder = gr.Textbox(label="Base Folder", value=DEFAULT_OUTPUT_FOLDER)
                        btn_t4_list = gr.Button("🔍 สแกน Clips")
                        t4_clip_msg = gr.Textbox(label="", lines=1, interactive=False, show_label=False)
                        t4_clip_table = gr.Dataframe(
                            headers=["Clip (relative path)", "Folder"],
                            datatype=["str", "str"],
                            interactive=False, elem_classes="stream-table",
                        )
                        t4_clip_dd = gr.Dropdown(label="เลือก Clip (สำหรับส่งทีละชิ้น)", choices=[])
                        with gr.Row():
                            btn_t4_one = gr.Button("📤 ส่ง Clip ที่เลือก", variant="primary")
                            btn_t4_all = gr.Button("📤 ส่งทุก Clip", variant="primary")

                    with gr.Tab("⬆️ Upload Clip"):
                        gr.Markdown("Upload ไฟล์ `.mp4` แล้วส่งตรงไปยัง Alert-Bridge")
                        t4_upload = gr.File(
                            label="เลือกไฟล์ .mp4",
                            file_types=[".mp4"],
                            type="filepath",
                        )
                        btn_t4_upload = gr.Button("📤 ส่ง Clip ที่ Upload", variant="primary")

                # ── right: prompt & event settings ───────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("#### VLM Prompt")
                    t4_prompt     = gr.Textbox(label="VLM Prompt *", lines=5)
                    t4_sys_prompt = gr.Textbox(label="System Prompt (ไม่บังคับ)", lines=2)
                    gr.Markdown("#### Event Info")
                    with gr.Row():
                        t4_event_type = gr.Textbox(label="Event Type", value="over_crowding", scale=2)
                        t4_severity   = gr.Dropdown(
                            label="Severity",
                            choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            value="MEDIUM", scale=1,
                        )
                    t4_sensor_id   = gr.Textbox(label="Sensor ID", value="sensor-1")
                    t4_stream_name = gr.Textbox(label="Stream Name", value="")
                    gr.Markdown("#### VSS Params")
                    with gr.Row():
                        t4_chunk_dur  = gr.Number(label="Chunk Duration (วิ)", value=60, minimum=1, precision=0)
                        t4_num_frames = gr.Number(label="Frames per Chunk", value=8, minimum=1, precision=0)
                    with gr.Row():
                        t4_reasoning       = gr.Checkbox(label="Enable Reasoning", value=False)
                        t4_do_verification = gr.Checkbox(label="Do Verification", value=True)

            t4_result = gr.Textbox(label="ผลลัพธ์", lines=8, elem_classes="status-box")

            # shared inputs for prompt/event settings
            _t4_common = [t4_sensor_id, t4_stream_name,
                          t4_prompt, t4_sys_prompt,
                          t4_event_type, t4_severity,
                          t4_chunk_dur, t4_num_frames, t4_reasoning, t4_do_verification]

            btn_t4_list.click(list_clips, inputs=t4_output_folder,
                              outputs=[t4_clip_table, t4_clip_dd, t4_clip_msg])
            btn_t4_one.click(
                submit_clip_to_alertbridge,
                inputs=[t4_clip_dd] + _t4_common,
                outputs=t4_result,
            )
            btn_t4_all.click(
                submit_all_clips,
                inputs=[t4_output_folder] + _t4_common,
                outputs=t4_result,
            )
            btn_t4_upload.click(
                submit_uploaded_clip,
                inputs=[t4_upload] + _t4_common,
                outputs=t4_result,
            )

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("CV_UI_PORT", "7862"))
    demo.launch(server_name="0.0.0.0", server_port=port,
                css=CSS, theme=gr.themes.Soft())
