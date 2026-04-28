######################################################################################################
# CV Event Detector UI — FastAPI backend (replaces Gradio)
######################################################################################################
from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import httpx
import json
import os
import shutil
import subprocess
import threading
import uuid
import requests
from datetime import datetime, timezone

app = FastAPI()

CV_API_URL = os.environ.get("NV_CV_EVENT_DETECTOR_API_URL", "http://nv-cv-event-detector:23491")
ALERTBRIDGE_URL = os.environ.get("NV_ALERTBRIDGE_URL", "http://alert-bridge:9080")
DEFAULT_OUTPUT_FOLDER = (
    os.environ.get("ALERT_REVIEW_MEDIA_BASE_DIR")
    or os.environ.get("FILTERED_CLIP_PATH")
    or "/tmp/cv-output"
)

_watchers: dict = {}
_lock = threading.Lock()

# ── Watcher helpers ───────────────────────────────────────────────────────────

def _log(sid: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _lock:
        if sid in _watchers:
            logs = _watchers[sid]["log"]
            logs.append(f"[{ts}] {msg}")
            _watchers[sid]["log"] = logs[-200:]


def _build_payload(video_path, sensor_id, stream_name, prompt, sys_prompt,
                   ev_type, severity, chunk_dur, num_frames, reasoning, verify):
    cv_meta = video_path.replace(".mp4", ".json")
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid.uuid4()), "version": "1.0",
        "@timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
        "sensor_id": sensor_id or "sensor-1",
        "stream_name": stream_name or None,
        "video_path": video_path,
        "cv_metadata_path": cv_meta if os.path.exists(cv_meta) else None,
        "alert": {"severity": severity.upper(), "status": "REVIEW_PENDING",
                  "type": ev_type, "description": prompt},
        "event": {"type": ev_type, "description": prompt},
        "vss_params": {
            "chunk_duration": int(chunk_dur), "chunk_overlap_duration": 3,
            "cv_metadata_overlay": True, "num_frames_per_chunk": int(num_frames),
            "enable_reasoning": bool(reasoning), "do_verification": bool(verify),
            "debug": False,
            "vlm_params": {
                "prompt": prompt,
                "system_prompt": sys_prompt or "You are a helpful assistant. Answer the user's question. Answer in yes or no only.",
            },
        },
        "confidence": 1.0, "meta_labels": [],
    }


def _watcher_loop(sid, folder, sensor_id, stream_name, prompt, sys_prompt,
                  ev_type, severity, chunk_dur, num_frames, reasoning, verify,
                  stop_ev, poll_sec):
    submitted: set = set()
    info_path = os.path.join(folder, "info.txt")
    _log(sid, f"เริ่ม watcher — ดู {info_path} ทุก {poll_sec}s")
    while not stop_ev.wait(poll_sec):
        try:
            if not os.path.exists(info_path):
                continue
            with open(info_path) as f:
                clips = [l.strip() for l in f if l.strip()]
            for clip in clips:
                if clip in submitted:
                    continue
                mp4 = os.path.join(folder, clip + ".mp4")
                if not os.path.exists(mp4):
                    continue
                payload = _build_payload(mp4, sensor_id, stream_name, prompt, sys_prompt,
                                         ev_type, severity, chunk_dur, num_frames, reasoning, verify)
                try:
                    r = requests.post(f"{ALERTBRIDGE_URL}/api/v1/alerts", json=payload, timeout=30)
                    ok = r.status_code in (200, 201, 202)
                    _log(sid, f"{'✅' if ok else '❌'} {clip}.mp4 → HTTP {r.status_code}")
                except Exception as e:
                    _log(sid, f"❌ {clip}.mp4 → {e}")
                submitted.add(clip)
        except Exception as e:
            _log(sid, f"⚠️ {e}")
    _log(sid, "Watcher หยุดแล้ว")


# ── Capture single frame from stream ─────────────────────────────────────────

@app.post("/api/capture-frame")
async def capture_frame(request: Request):
    data = await request.json()
    url = (data.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "url required")

    import base64
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = ["ffmpeg", "-y"]
        if url.startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp"]
        # file:// → pass as-is; ffmpeg understands it
        cmd += ["-i", url, "-vframes", "1", "-q:v", "2", tmp_path]

        result = subprocess.run(cmd, capture_output=True, timeout=20)

        if result.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            stderr = result.stderr.decode(errors="replace")[-400:]
            return {"error": f"ffmpeg failed: {stderr}"}

        with open(tmp_path, "rb") as f:
            raw = f.read()
        encoded = base64.b64encode(raw).decode()
        return {"image": f"data:image/jpeg;base64,{encoded}"}

    except subprocess.TimeoutExpired:
        return {"error": "Timeout — stream ไม่ตอบสนองใน 20 วิ"}
    except FileNotFoundError:
        return {"error": "ffmpeg ไม่ได้ติดตั้งในระบบ"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Live MJPEG stream ─────────────────────────────────────────────────────────

@app.get("/api/stream-live")
async def stream_live(url: str = ""):
    if not url:
        raise HTTPException(400, "url required")

    async def generate():
        cmd = ["ffmpeg", "-y"]
        if url.startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += [
            "-i", url,
            "-vf", "scale=854:-2",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "5",
            "-r", "10",
            "pipe:1",
        ]
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            buf = b""
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    soi = buf.find(b"\xff\xd8")
                    if soi == -1:
                        buf = b""
                        break
                    eoi = buf.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        buf = buf[soi:]
                        break
                    jpg = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpg + b"\r\n"
                    )
        except Exception:
            pass
        finally:
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Static SPA ────────────────────────────────────────────────────────────────

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse({"message": "place index.html in static/"})


# ── Config ────────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return {
        "default_output_folder": DEFAULT_OUTPUT_FOLDER,
        "alertbridge_url": ALERTBRIDGE_URL,
    }


# ── CV API proxy ──────────────────────────────────────────────────────────────

@app.api_route("/cv/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_cv(path: str, request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.request(
            request.method,
            f"{CV_API_URL}/{path}",
            content=body or None,
            headers={"content-type": request.headers.get("content-type", "application/json")},
            params=dict(request.query_params),
        )
    return JSONResponse(r.json(), status_code=r.status_code)


# ── Add Streams (SSE) ─────────────────────────────────────────────────────────

def _sse(msg: str, done: bool = False) -> str:
    return f"data: {json.dumps({'msg': msg, 'done': done})}\n\n"


@app.post("/api/add-streams")
async def add_streams(request: Request):
    data = await request.json()
    pipeline_id = data["pipeline_id"]
    streams = data["streams"]
    cv_prompt = data.get("cv_prompt")
    threshold = float(data.get("box_threshold", 0.3))
    watcher = data.get("watcher", {})

    async def generate():
        total = len(streams)
        added = 0
        yield _sse(f"⏳ เริ่มเพิ่ม {total} stream...\n")
        for s in streams:
            name = (s.get("name") or "stream").strip()
            url = (s.get("url") or "").strip()
            sensor_id = (s.get("sensor_id") or "sensor-1").strip()
            safe = name.replace(" ", "_")
            rx = int(s.get("zone_x") or 0)
            ry = int(s.get("zone_y") or 0)
            rw = int(s.get("zone_w") or 0)
            rh = int(s.get("zone_h") or 0)
            has_zone = rw > 0 and rh > 0
            rois = [[rx, ry, rw, rh]] if has_zone else [[]]
            zone_txt = f"zone({rx},{ry},{rw}×{rh})" if has_zone else "zone(ทั้งภาพ)"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_folder = os.path.join(DEFAULT_OUTPUT_FOLDER, f"{safe}_{ts}")

            yield _sse(f"  ⏳ [{name}] {zone_txt} — กำลังเพิ่ม...")

            payload = {
                "version": "1.0", "stream_url": url, "pipeline_id": pipeline_id,
                "output_folder": out_folder, "sensor_id": sensor_id, "stream_name": safe,
                "processing_state": "enabled",
                "cv_params": {"gdinoprompt": cv_prompt, "gdinothreshold": threshold, "gdino_rois": rois},
            }
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.post(f"{CV_API_URL}/api/addstream", json=payload)
                    d = r.json()
                if d.get("status") == "success":
                    stream_id = d["stream_id"]
                    auto = watcher.get("enabled") and (watcher.get("prompt") or "").strip()
                    note = ""
                    if auto:
                        stop_ev = threading.Event()
                        t = threading.Thread(
                            target=_watcher_loop,
                            args=(stream_id, out_folder, sensor_id, safe,
                                  watcher["prompt"], watcher.get("system_prompt", ""),
                                  watcher.get("event_type", "event"),
                                  watcher.get("severity", "MEDIUM"),
                                  watcher.get("chunk_duration", 60),
                                  watcher.get("num_frames", 8),
                                  watcher.get("enable_reasoning", False),
                                  watcher.get("do_verification", True),
                                  stop_ev, int(watcher.get("poll_sec", 5))),
                            daemon=True,
                        )
                        with _lock:
                            _watchers[stream_id] = {
                                "thread": t, "stop_event": stop_ev,
                                "output_folder": out_folder,
                                "prompt": watcher["prompt"], "log": [],
                            }
                        t.start()
                        note = "  🤖"
                    added += 1
                    yield _sse(f"  ✅ [{name}] {zone_txt} — {stream_id}{note}")
                else:
                    yield _sse(f"  ❌ [{name}] {d.get('message', r.text)}")
            except Exception as e:
                yield _sse(f"  ❌ [{name}] {e}")

        yield _sse(f"\nสรุป: เพิ่ม {added}/{total} stream สำเร็จ", done=True)

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Watchers ──────────────────────────────────────────────────────────────────

@app.get("/api/watchers")
async def list_watchers():
    with _lock:
        return {"watchers": [
            {"stream_id": sid, "alive": w["thread"].is_alive(),
             "output_folder": w.get("output_folder", ""),
             "prompt": (w.get("prompt") or "")[:60]}
            for sid, w in _watchers.items()
        ]}


@app.delete("/api/watchers/{stream_id}")
async def stop_watcher(stream_id: str):
    with _lock:
        entry = _watchers.pop(stream_id, None)
    if not entry:
        raise HTTPException(404, "Watcher not found")
    entry["stop_event"].set()
    return {"status": "stopped"}


@app.get("/api/watchers/{stream_id}/log")
async def watcher_log(stream_id: str):
    with _lock:
        entry = _watchers.get(stream_id)
    if not entry:
        raise HTTPException(404, "Watcher not found")
    return {"log": entry["log"][-50:]}


# ── Clips ─────────────────────────────────────────────────────────────────────

@app.get("/api/clips")
async def list_clips(folder: str = ""):
    base = folder.strip() or DEFAULT_OUTPUT_FOLDER
    if not os.path.isdir(base):
        return {"clips": [], "error": f"ไม่พบ folder: {base}"}
    clips = []
    seen: set = set()
    for dirpath, _, filenames in os.walk(base):
        # Prefer info.txt ordering when available
        if "info.txt" in filenames:
            try:
                with open(os.path.join(dirpath, "info.txt")) as f:
                    for clip in [l.strip() for l in f if l.strip()]:
                        mp4 = os.path.join(dirpath, clip + ".mp4")
                        if os.path.exists(mp4) and mp4 not in seen:
                            seen.add(mp4)
                            clips.append({
                                "path": mp4,
                                "name": clip + ".mp4",
                                "folder": os.path.relpath(dirpath, base),
                            })
            except Exception:
                pass
        # Also pick up any .mp4 not referenced by info.txt
        for fn in filenames:
            if fn.lower().endswith(".mp4"):
                mp4 = os.path.join(dirpath, fn)
                if mp4 not in seen:
                    seen.add(mp4)
                    clips.append({
                        "path": mp4,
                        "name": fn,
                        "folder": os.path.relpath(dirpath, base),
                    })
    return {"clips": clips}


@app.post("/api/clips/upload")
async def upload_clip(file: UploadFile = File(...)):
    upload_dir = os.path.join(DEFAULT_OUTPUT_FOLDER, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(file.filename or "clip.mp4")
    if not safe_name.lower().endswith(".mp4"):
        raise HTTPException(400, "รองรับเฉพาะไฟล์ .mp4")
    dest = os.path.join(upload_dir, safe_name)
    base_path, ext = os.path.splitext(dest)
    counter = 0
    while os.path.exists(dest):
        counter += 1
        dest = f"{base_path}_{counter}{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"path": dest, "name": os.path.basename(dest)}


@app.post("/api/clips/submit")
async def submit_clip(request: Request):
    d = await request.json()
    if not d.get("prompt", "").strip():
        raise HTTPException(400, "prompt is required")
    payload = _build_payload(
        d["video_path"], d.get("sensor_id", "sensor-1"), d.get("stream_name"),
        d["prompt"], d.get("system_prompt", ""),
        d.get("event_type", "event"), d.get("severity", "MEDIUM"),
        d.get("chunk_duration", 60), d.get("num_frames", 8),
        d.get("enable_reasoning", False), d.get("do_verification", True),
    )
    try:
        r = requests.post(f"{ALERTBRIDGE_URL}/api/v1/alerts", json=payload, timeout=30)
        return {"ok": r.status_code in (200, 201, 202), "status": r.status_code}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Hardware / Health ─────────────────────────────────────────────────────────

@app.get("/api/hardware")
async def hardware():
    try:
        smi = next(
            (p for p in ["nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"]
             if os.path.isfile(p)),
            "nvidia-smi",
        )
        r = subprocess.run(
            [smi, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in r.stdout.strip().splitlines():
            idx, name, mu, mt, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({"idx": idx, "name": name,
                         "mu": int(mu), "mt": int(mt),
                         "pct": int(mu) * 100 // max(int(mt), 1),
                         "util": int(util), "temp": int(temp)})
        return {"gpus": gpus}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


@app.get("/api/health")
async def health():
    result: dict = {}
    for name, url in [("cv_detector", f"{CV_API_URL}/health"),
                      ("alert_bridge", f"{ALERTBRIDGE_URL}/health")]:
        try:
            r = requests.get(url, timeout=5)
            result[name] = r.json().get("status", "ok")
        except Exception:
            result[name] = "error"
    with _lock:
        result["watchers"] = len(_watchers)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CV_UI_PORT", "7862"))
    uvicorn.run(app, host="0.0.0.0", port=port)
