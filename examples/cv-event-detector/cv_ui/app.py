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
from datetime import datetime, timezone

API_URL = os.environ.get("NV_CV_EVENT_DETECTOR_API_URL", "http://localhost:23491")
ALERTBRIDGE_URL = os.environ.get("NV_ALERTBRIDGE_URL", "http://alert-bridge:9080")
DEFAULT_OUTPUT_FOLDER = os.environ.get("DEFAULT_OUTPUT_FOLDER", "/tmp/cv-output")

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

ALERT_ENDPOINT = "/api/v1/alerts"

def _build_event_payload(video_path, sensor_id, stream_name,
                          prompt, system_prompt,
                          event_type, event_desc, severity,
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
                  event_type: str, event_desc: str, severity: str,
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
                    event_type, event_desc, severity,
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
                  event_type: str, event_desc: str, severity: str,
                  chunk_duration: int, num_frames: int, enable_reasoning: bool,
                  do_verification: bool,
                  poll_sec: int = 5):
    stop_event = threading.Event()
    t = threading.Thread(
        target=_watcher_loop,
        args=(stream_id, output_folder, sensor_id, stream_name,
              prompt, system_prompt, event_type, event_desc, severity,
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
            choices, default = _pipeline_choices()
            return (
                f"✅ Pipeline สร้างสำเร็จ\nID: {pid}",
                gr.update(choices=choices, value=pid),
            )
        return f"❌ {data.get('message', r.text)}", gr.update()
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}", gr.update()


def delete_pipeline(pipeline_id):
    if not pipeline_id:
        return "❌ กรุณาระบุ Pipeline ID", gr.update()
    try:
        r = _delete("/api/pipeline", {"id": pipeline_id, "cleanup_resources": True})
        data = r.json()
        msg = "✅ ลบ Pipeline สำเร็จ" if data.get("status") == "success" else f"❌ {data.get('message', r.text)}"
        choices, default = _pipeline_choices()
        return msg, gr.update(choices=choices, value=default)
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}", gr.update()


def _pipeline_choices():
    try:
        data = _get("/api/pipelines").json()
        pipelines = data.get("pipelines", [])
        choices = [p["id"] for p in pipelines]
        return choices, (choices[0] if choices else None)
    except Exception:
        return [], None


def refresh_pipeline_dropdown():
    choices, default = _pipeline_choices()
    return gr.update(choices=choices, value=default)


def get_pipelines_table():
    try:
        data = _get("/api/pipelines").json()
        rows = []
        for p in data.get("pipelines", []):
            rows.append([p["id"], p.get("config", ""), p.get("created_at", "")[:19]])
        return rows
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def add_stream(pipeline_id, stream_url, sensor_id, stream_name, output_folder,
               detection_classes, box_threshold,
               roi_x, roi_y, roi_w, roi_h,
               vlm_prompt, vlm_system_prompt,
               event_type, event_desc, severity,
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
                event_desc=event_desc.strip() or "Event detected",
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
            auto = "🤖" if sid in watcher_ids else ""
            rows.append([
                sid,
                s.get("pipeline_id", "")[:12] + "...",
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
        choices = [
            f"{s['stream_id']}  [{s.get('processing_state', '')}]"
            + ("  🤖" if s['stream_id'] in watcher_ids else "")
            + f"  {s.get('timestamp', '')[:16]}"
            for s in streams
        ]
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
        r = _delete("/api/stream", {"stream_id": sid, "version": "1.0"})
        data = r.json()
        if data.get("status") == "success":
            return f"✅ หยุด Stream และ watcher สำเร็จ"
        return f"❌ {data.get('message', r.text)}"
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

def _read_info_txt(output_folder: str):
    info_path = os.path.join(output_folder.strip(), "info.txt")
    if not os.path.exists(info_path):
        return []
    with open(info_path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def list_clips(output_folder: str):
    if not output_folder.strip():
        return [], gr.update(choices=[]), "❌ กรุณาระบุ Output Folder"
    clips = _read_info_txt(output_folder)
    if not clips:
        return [], gr.update(choices=[]), f"ไม่พบ info.txt หรือไม่มี clips ใน {output_folder}"
    rows = [[c + ".mp4", c + ".json"] for c in clips]
    choices = [c + ".mp4" for c in clips]
    return rows, gr.update(choices=choices), f"พบ {len(clips)} clips"


def submit_clip_to_alertbridge(output_folder, clip_choice,
                                sensor_id, stream_name,
                                prompt, system_prompt,
                                event_type, event_desc, severity,
                                chunk_duration, num_frames, enable_reasoning,
                                do_verification):
    if not clip_choice:
        return "❌ กรุณาเลือก clip ก่อน"
    if not prompt.strip():
        return "❌ กรุณาใส่ VLM Prompt ก่อน"
    video_path = os.path.join(output_folder.strip(), clip_choice)
    payload = _build_event_payload(video_path, sensor_id, stream_name,
                                   prompt, system_prompt,
                                   event_type, event_desc, severity,
                                   chunk_duration, num_frames, enable_reasoning,
                                   do_verification)
    try:
        r = requests.post(f"{ALERTBRIDGE_URL}{ALERT_ENDPOINT}", json=payload, timeout=30)
        ok = r.status_code in (200, 201, 202)
        return f"{'✅' if ok else '❌'} HTTP {r.status_code}\n{r.text[:300]}"
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"


def submit_all_clips(output_folder, sensor_id, stream_name,
                     prompt, system_prompt, event_type, event_desc, severity,
                     chunk_duration, num_frames, enable_reasoning, do_verification):
    if not prompt.strip():
        return "❌ กรุณาใส่ VLM Prompt ก่อน"
    clips = _read_info_txt(output_folder)
    if not clips:
        return f"❌ ไม่พบ clips ใน {output_folder}"
    results = []
    for c in clips:
        video_path = os.path.join(output_folder.strip(), c + ".mp4")
        payload = _build_event_payload(video_path, sensor_id, stream_name,
                                       prompt, system_prompt,
                                       event_type, event_desc, severity,
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
footer { display: none !important; }
"""

with gr.Blocks(title="CV Event Detector UI") as demo:

    gr.Markdown("# CV Event Detector — Pipeline & Stream Manager")

    # ── Health bar ────────────────────────────────────────────────────────────
    with gr.Row():
        health_txt = gr.Textbox(label="", interactive=False, show_label=False,
                                value=health_check, every=30, scale=4)
        btn_health = gr.Button("🔄 ตรวจสอบ", scale=1)
    btn_health.click(health_check, outputs=health_txt)

    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Tabs():

        # ── TAB 1: Pipeline Setup ─────────────────────────────────────────────
        with gr.TabItem("⚙️ จัดการ Pipeline"):
            gr.Markdown("### สร้าง Pipeline ใหม่")
            with gr.Row():
                with gr.Column():
                    t1_name     = gr.Textbox(label="ชื่อ Pipeline", value="cv-pipeline")
                    t1_type     = gr.Textbox(label="ประเภท (type)", value="object_detection")
                    t1_endpoint = gr.Textbox(label="Alert Bridge URL", value=ALERTBRIDGE_URL)
                with gr.Column():
                    t1_min_clip   = gr.Number(label="Min Clip Duration (วินาที)", value=5, minimum=1)
                    t1_max_clip   = gr.Number(label="Max Clip Duration (วินาที)", value=60, minimum=1)
                    t1_fskip      = gr.Slider(label="Frame Skip Interval", minimum=0, maximum=10, step=1, value=0)
                    t1_min_detect = gr.Slider(label="Min Object Detection Threshold", minimum=1, maximum=50, step=1, value=3)

            btn_create_pipe = gr.Button("➕ สร้าง Pipeline", variant="primary")
            t1_result = gr.Textbox(label="ผลลัพธ์", lines=3, elem_classes="status-box")

            gr.Markdown("---")
            gr.Markdown("### Pipeline ที่มีอยู่")
            with gr.Row():
                btn_refresh_pipes = gr.Button("🔄 โหลดใหม่")
                t1_del_id = gr.Textbox(label="Pipeline ID ที่จะลบ", scale=3)
                btn_del_pipe = gr.Button("🗑️ ลบ Pipeline", variant="stop")

            t1_pipe_table = gr.Dataframe(
                headers=["Pipeline ID", "Config", "Created At"],
                datatype=["str", "str", "str"],
                elem_classes="stream-table",
                interactive=False,
                value=get_pipelines_table,
                every=60,
            )
            t1_pipe_result = gr.Textbox(label="ผลลัพธ์", lines=2, elem_classes="status-box")
            _pipeline_dd_shared = gr.State(value=None)

            btn_create_pipe.click(
                create_pipeline,
                inputs=[t1_name, t1_endpoint, t1_type,
                        t1_min_clip, t1_max_clip, t1_fskip, t1_min_detect],
                outputs=[t1_result, _pipeline_dd_shared],
            ).then(get_pipelines_table, outputs=t1_pipe_table)
            btn_refresh_pipes.click(get_pipelines_table, outputs=t1_pipe_table)
            btn_del_pipe.click(
                delete_pipeline,
                inputs=[t1_del_id],
                outputs=[t1_pipe_result, _pipeline_dd_shared],
            ).then(get_pipelines_table, outputs=t1_pipe_table)

        # ── TAB 2: Add Stream ─────────────────────────────────────────────────
        with gr.TabItem("➕ เพิ่ม Stream"):
            gr.Markdown(
                "กรอกข้อมูล Stream แล้วกด **เพิ่ม Stream** ได้เรื่อยๆ "
                "ฟอร์มจะ**ไม่ล้างค่า**หลัง submit เพื่อให้เพิ่มหลาย stream ได้สะดวก"
            )
            with gr.Row():
                # ── left: stream identity ─────────────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("#### Stream Source")
                    t2_pipeline = gr.Dropdown(
                        label="Pipeline",
                        choices=_pipeline_choices()[0],
                        allow_custom_value=True,
                    )
                    btn_t2_refresh_pipe = gr.Button("🔄 โหลด Pipeline")
                    t2_stream_url = gr.Textbox(
                        label="Stream URL",
                        placeholder="rtsp://192.168.1.1:554/stream  หรือ  file:///tmp/video.mp4",
                    )
                    t2_stream_name   = gr.Textbox(label="Stream Name", value="cam-1")
                    t2_sensor_id     = gr.Textbox(label="Sensor ID", value="sensor-1")
                    t2_output_folder = gr.Textbox(label="Output Base Folder", value=DEFAULT_OUTPUT_FOLDER)

                    gr.Markdown("#### CV Detection Parameters")
                    t2_classes = gr.Textbox(
                        label="Detection Classes (หนึ่งคลาสต่อบรรทัด)",
                        lines=4,
                        placeholder="person\ncar\ntruck",
                        value="person",
                    )
                    t2_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.1, maximum=1.0, step=0.05, value=0.3)
                    gr.Markdown("#### ROI — เว้นว่างถ้าใช้ทั้งภาพ")
                    with gr.Row():
                        t2_roi_x = gr.Number(label="X", value=0, minimum=0, precision=0)
                        t2_roi_y = gr.Number(label="Y", value=0, minimum=0, precision=0)
                        t2_roi_w = gr.Number(label="Width (0=ทั้งภาพ)", value=0, minimum=0, precision=0)
                        t2_roi_h = gr.Number(label="Height (0=ทั้งภาพ)", value=0, minimum=0, precision=0)

                # ── right: VLM auto-review ────────────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("#### 🤖 Auto VLM Review")
                    t2_auto_review = gr.Checkbox(
                        label="เปิด Auto-Review (ส่ง clip ให้ Alert-Bridge อัตโนมัติ)",
                        value=True,
                    )
                    t2_poll_sec = gr.Slider(
                        label="Poll Interval (วินาที)", minimum=3, maximum=60, step=1, value=5)
                    t2_vlm_prompt = gr.Textbox(
                        label="VLM Prompt *",
                        lines=5,
                        placeholder="e.g. Does the video show signs of overcrowding? Describe what you see.",
                    )
                    t2_vlm_sys_prompt = gr.Textbox(
                        label="System Prompt (ไม่บังคับ)",
                        lines=2,
                        placeholder="e.g. You are a warehouse safety monitoring system.",
                    )
                    gr.Markdown("#### Event Info")
                    with gr.Row():
                        t2_event_type = gr.Textbox(label="Event Type", value="over_crowding", scale=2)
                        t2_severity   = gr.Dropdown(
                            label="Severity",
                            choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            value="MEDIUM", scale=1,
                        )
                    t2_event_desc = gr.Textbox(
                        label="Event Description", value="Event detected by CV pipeline")
                    gr.Markdown("#### VSS Params")
                    with gr.Row():
                        t2_chunk_dur      = gr.Number(label="Chunk Duration (วิ)", value=60, minimum=1, precision=0)
                        t2_num_frames     = gr.Number(label="Frames per Chunk", value=8, minimum=1, precision=0)
                    with gr.Row():
                        t2_reasoning      = gr.Checkbox(label="Enable Reasoning", value=False)
                        t2_do_verification = gr.Checkbox(label="Do Verification (แสดง Yes/No alert result)", value=True)

            btn_add_stream = gr.Button("➕ เพิ่ม Stream", variant="primary", size="lg")
            t2_result = gr.Textbox(label="ผลลัพธ์", lines=6, elem_classes="status-box")

            btn_t2_refresh_pipe.click(refresh_pipeline_dropdown, outputs=t2_pipeline)
            btn_add_stream.click(
                add_stream,
                inputs=[t2_pipeline, t2_stream_url, t2_sensor_id, t2_stream_name,
                        t2_output_folder, t2_classes, t2_threshold,
                        t2_roi_x, t2_roi_y, t2_roi_w, t2_roi_h,
                        t2_vlm_prompt, t2_vlm_sys_prompt,
                        t2_event_type, t2_event_desc, t2_severity,
                        t2_chunk_dur, t2_num_frames, t2_reasoning, t2_do_verification,
                        t2_poll_sec, t2_auto_review],
                outputs=t2_result,
            )

        # ── TAB 3: Stream Manager ─────────────────────────────────────────────
        with gr.TabItem("📋 จัดการ Stream"):
            gr.Markdown("### รายการ Stream ทั้งหมด  (🤖 = มี Auto-Review เปิดอยู่)")
            with gr.Row():
                btn_refresh_streams = gr.Button("🔄 โหลดใหม่", variant="secondary")

            t3_stream_table = gr.Dataframe(
                headers=["Stream ID", "Pipeline ID", "State", "Timestamp", "Auto"],
                datatype=["str", "str", "str", "str", "str"],
                elem_classes="stream-table",
                interactive=False,
                value=get_streams_table,
                every=10,
            )

            gr.Markdown("---")
            gr.Markdown("### ตรวจสอบ / หยุด / ดู Watcher Log")

            t3_stream_dd = gr.Dropdown(
                label="เลือก Stream",
                choices=[],
                allow_custom_value=False,
            )
            with gr.Row():
                t3_stream_id = gr.Textbox(
                    label="Stream ID (copy ได้จากช่องนี้)",
                    scale=4, interactive=True,
                    placeholder="เลือกจาก dropdown หรือพิมพ์โดยตรง",
                )
                t3_wait_ms = gr.Number(
                    label="Wait (ms)", value=500, minimum=0,
                    maximum=30000, precision=0, scale=1,
                )
            with gr.Row():
                btn_status    = gr.Button("🔍 ดูสถานะ")
                btn_wlog      = gr.Button("📜 ดู Watcher Log")
                btn_stop      = gr.Button("⏹️ หยุด Stream + Watcher", variant="stop")

            t3_status_out = gr.Textbox(label="สถานะ / Watcher Log", lines=12, elem_classes="status-box")
            t3_stop_out   = gr.Textbox(label="ผลการหยุด", lines=2, elem_classes="status-box")

            t3_stream_dd.change(pick_stream_id, inputs=t3_stream_dd, outputs=t3_stream_id)
            btn_refresh_streams.click(streams_table_and_dropdown, outputs=[t3_stream_table, t3_stream_dd])
            btn_status.click(get_stream_status, inputs=[t3_stream_id, t3_wait_ms], outputs=t3_status_out)
            btn_wlog.click(get_watcher_log, inputs=t3_stream_id, outputs=t3_status_out)
            btn_stop.click(stop_stream, inputs=[t3_stream_id], outputs=t3_stop_out).then(
                streams_table_and_dropdown, outputs=[t3_stream_table, t3_stream_dd]
            )

        # ── TAB 4: Manual VLM Review ──────────────────────────────────────────
        with gr.TabItem("📤 Manual VLM Review"):
            gr.Markdown(
                "ส่ง clips ไปให้ Alert-Bridge **ด้วยตัวเอง** "
                "(ใช้กรณีปิด Auto-Review หรือส่งซ้ำ)\n\n"
                f"`Alert-Bridge: {ALERTBRIDGE_URL}`"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### เลือก Output Folder และ Clip")
                    t4_output_folder = gr.Textbox(label="Output Folder", value=DEFAULT_OUTPUT_FOLDER)
                    btn_t4_list = gr.Button("🔍 โหลด Clips")
                    t4_clip_msg = gr.Textbox(label="", lines=1, interactive=False, show_label=False)
                    t4_clip_table = gr.Dataframe(
                        headers=["Clip (.mp4)", "Metadata (.json)"],
                        datatype=["str", "str"],
                        interactive=False, elem_classes="stream-table",
                    )
                    t4_clip_dd = gr.Dropdown(label="เลือก Clip (สำหรับส่งทีละชิ้น)", choices=[])

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
                    t4_event_desc  = gr.Textbox(label="Event Description", value="Event detected by CV pipeline")
                    t4_sensor_id   = gr.Textbox(label="Sensor ID", value="sensor-1")
                    t4_stream_name = gr.Textbox(label="Stream Name", value="")
                    gr.Markdown("#### VSS Params")
                    with gr.Row():
                        t4_chunk_dur  = gr.Number(label="Chunk Duration (วิ)", value=60, minimum=1, precision=0)
                        t4_num_frames = gr.Number(label="Frames per Chunk", value=8, minimum=1, precision=0)
                    with gr.Row():
                        t4_reasoning       = gr.Checkbox(label="Enable Reasoning", value=False)
                        t4_do_verification = gr.Checkbox(label="Do Verification (แสดง Yes/No alert result)", value=True)

            with gr.Row():
                btn_t4_one = gr.Button("📤 ส่ง Clip ที่เลือก", variant="primary")
                btn_t4_all = gr.Button("📤 ส่งทุก Clip", variant="primary")

            t4_result = gr.Textbox(label="ผลลัพธ์", lines=8, elem_classes="status-box")

            btn_t4_list.click(list_clips, inputs=t4_output_folder,
                              outputs=[t4_clip_table, t4_clip_dd, t4_clip_msg])
            btn_t4_one.click(
                submit_clip_to_alertbridge,
                inputs=[t4_output_folder, t4_clip_dd,
                        t4_sensor_id, t4_stream_name,
                        t4_prompt, t4_sys_prompt,
                        t4_event_type, t4_event_desc, t4_severity,
                        t4_chunk_dur, t4_num_frames, t4_reasoning, t4_do_verification],
                outputs=t4_result,
            )
            btn_t4_all.click(
                submit_all_clips,
                inputs=[t4_output_folder, t4_sensor_id, t4_stream_name,
                        t4_prompt, t4_sys_prompt,
                        t4_event_type, t4_event_desc, t4_severity,
                        t4_chunk_dur, t4_num_frames, t4_reasoning, t4_do_verification],
                outputs=t4_result,
            )

    _pipeline_dd_shared.change(
        lambda v: gr.update(value=v) if v else gr.update(),
        inputs=_pipeline_dd_shared,
        outputs=t2_pipeline,
    )

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("CV_UI_PORT", "7862"))
    demo.launch(server_name="0.0.0.0", server_port=port,
                css=CSS, theme=gr.themes.Soft())
