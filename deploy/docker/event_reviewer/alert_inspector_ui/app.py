######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
######################################################################################################
"""
Alert Inspector UI — rebuilt.
Mirrors the NVIDIA vss-alert-inspector-ui data flow:
  • Primary  : reads directly from Redis stream `alert-bridge-enhanced-stream`
               (same stream the alert-bridge WebSocket feeds from)
  • Fallback : WebSocket ws://alert-bridge:9080/ws  +  REST GET /api/v1/alerts
Configuration comes entirely from environment variables set in .env / compose.
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import gradio as gr
import requests

# ─── Configuration (from environment — no compose edits needed) ───────────────
ALERT_BRIDGE_WS_URL = os.environ.get("ALERT_BRIDGE_BASE_URL", "ws://alert-bridge:9080")
ALERT_BRIDGE_HTTP   = (ALERT_BRIDGE_WS_URL
                       .replace("ws://",  "http://")
                       .replace("wss://", "https://"))
BACKEND_IP          = os.environ.get("BACKEND_IP",   "via-server")
BACKEND_PORT        = os.environ.get("BACKEND_PORT", "8000")
VSS_URL             = f"http://{BACKEND_IP}:{BACKEND_PORT}"
_vlm_model_cache: str | None = None


def _get_vlm_model() -> str | None:
    global _vlm_model_cache
    if _vlm_model_cache:
        return _vlm_model_cache
    try:
        r = requests.get(f"{VSS_URL}/models", timeout=5)
        if r.ok:
            models = r.json().get("data", [])
            if models:
                _vlm_model_cache = models[0]["id"]
                return _vlm_model_cache
    except Exception:
        pass
    return None
MEDIA_DIR           = os.environ.get("ALERT_REVIEW_MEDIA_BASE_DIR", "/tmp/alerts")
GRADIO_HOST         = os.environ.get("GRADIO_SERVER", "0.0.0.0")
GRADIO_PORT         = int(os.environ.get("GRADIO_PORT", "7860"))

# Redis — same defaults the alert-bridge config.yaml uses
REDIS_HOST   = os.environ.get("REDIS_HOST",   "redis")
REDIS_PORT   = int(os.environ.get("REDIS_PORT",   "6379"))
STREAM_NAME  = os.environ.get("ALERT_STREAM", "alert-bridge-enhanced-stream")

PAGE_SIZE = 20


# ─── In-memory store ──────────────────────────────────────────────────────────
_store: list[dict] = []
_lock  = threading.Lock()
_MAX   = 10_000
_last_id = "0-0"          # Redis stream cursor (track what we've read)
_last_id_lock = threading.Lock()


def _upsert(alert: dict) -> None:
    """Insert or update alert by id (newest-first)."""
    with _lock:
        aid = alert.get("id")
        if aid:
            for i, a in enumerate(_store):
                if a.get("id") == aid:
                    _store[i] = alert
                    return
        _store.insert(0, alert)
        if len(_store) > _MAX:
            _store.pop()


def _snapshot() -> list[dict]:
    with _lock:
        return list(_store)


# ─── Redis helpers ────────────────────────────────────────────────────────────
def _redis_client():
    import redis
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                       decode_responses=True, socket_timeout=5,
                       socket_connect_timeout=5)


def _parse_stream_entry(fields: dict) -> dict | None:
    """Extract the JSON alert payload from a Redis stream entry's fields."""
    # The alert-bridge stores the full ReviewAlertResponse JSON as a single field.
    # Try common field names first, then fall back to any field that looks like JSON.
    for key in ("data", "message", "payload", "alert", "result", "body", "event"):
        val = fields.get(key, "")
        if val and isinstance(val, str):
            try:
                obj = json.loads(val)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    # Fallback: try every field value
    for val in fields.values():
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                obj = json.loads(val)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


# ─── Fetch: read ALL history from Redis stream ────────────────────────────────
_fetch_status = ""
_fetch_lock   = threading.Lock()


def _fetch_redis_all() -> str:
    """XRANGE the full stream and populate the store. Returns status string."""
    global _last_id
    try:
        r = _redis_client()
        entries = r.xrange(STREAM_NAME, "-", "+")
        count = 0
        for msg_id, fields in entries:
            obj = _parse_stream_entry(fields)
            if obj:
                _upsert(obj)
                count += 1
        with _last_id_lock:
            if entries:
                _last_id = entries[-1][0]   # advance cursor
        msg = f"✅ Redis '{STREAM_NAME}': {count} alerts loaded"
    except ImportError:
        msg = "❌ redis-py not installed (add redis to requirements.txt)"
    except Exception as e:
        msg = f"❌ Redis {REDIS_HOST}:{REDIS_PORT} — {e}"

    # Also try the REST fallback
    rest_msg = _fetch_rest_fallback()
    combined = msg if not rest_msg else f"{msg}  |  {rest_msg}"

    with _fetch_lock:
        _fetch_status = combined
    return combined


def _fetch_rest_fallback() -> str:
    """Try GET /api/v1/alerts as supplemental source. Returns brief status."""
    try:
        r = requests.get(f"{ALERT_BRIDGE_HTTP}/api/v1/alerts", timeout=5)
        if r.ok:
            payload = r.json()
            items = (payload if isinstance(payload, list)
                     else payload.get("alerts",
                          payload.get("results",
                          payload.get("data", []))))
            count = 0
            for item in reversed(items or []):
                if isinstance(item, dict):
                    _upsert(item)
                    count += 1
            return f"REST +{count}" if count else ""
        return f"REST HTTP {r.status_code}"
    except Exception:
        return ""


# ─── Background: real-time Redis XREAD ───────────────────────────────────────
def _redis_stream_loop() -> None:
    """Continuously tail the stream for new entries."""
    global _last_id
    r = None
    while True:
        try:
            if r is None:
                r = _redis_client()

            with _last_id_lock:
                cursor = _last_id

            # block=2000 ms — wait for new messages
            results = r.xread({STREAM_NAME: cursor}, count=50, block=2000)
            if results:
                for _stream, entries in results:
                    for msg_id, fields in entries:
                        obj = _parse_stream_entry(fields)
                        if obj:
                            _upsert(obj)
                    with _last_id_lock:
                        _last_id = entries[-1][0]
        except ImportError:
            time.sleep(60)   # redis-py not installed — stop retrying fast
        except Exception as e:
            r = None
            time.sleep(5)


# ─── Background: WebSocket from alert-bridge (supplemental) ──────────────────
def _ws_loop() -> None:
    try:
        import websocket
    except ImportError:
        return

    # Try known paths; the alert-bridge config uses consumer_group_prefix="websocket_instance"
    paths = ["/ws", "/ws/alerts", "/api/v1/ws", "/api/ws", ""]
    while True:
        for path in paths:
            try:
                ws = websocket.create_connection(
                    ALERT_BRIDGE_WS_URL + path, timeout=5)
                while True:
                    raw = ws.recv()
                    try:
                        data = json.loads(raw)
                        for item in ([data] if isinstance(data, dict) else data):
                            if isinstance(item, dict):
                                _upsert(item)
                    except Exception:
                        pass
            except Exception:
                pass
        time.sleep(10)


# Start background threads
threading.Thread(target=_redis_stream_loop, daemon=True, name="redis-tail").start()
threading.Thread(target=_ws_loop,           daemon=True, name="ws-listener").start()


# ─── Utilities ────────────────────────────────────────────────────────────────
SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡",
            "LOW": "🟢", "INFORMAL": "⚪"}
VLM_ICON = {"SUCCESS": "✅", "FAILURE": "❌"}

# Symlink directory inside /tmp — always allowed by Gradio regardless of allowed_paths.
# Needed because ALERT_REVIEW_MEDIA_BASE_DIR is mounted as a volume but NOT passed as
# an env var to this container in the default compose, so MEDIA_DIR may be wrong.
_PREVIEW_DIR = "/tmp/alert-previews"
os.makedirs(_PREVIEW_DIR, exist_ok=True)


def _fmt_ts(ts: str, tz_name: str = "UTC") -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        try:
            tz = ZoneInfo(tz_name)
            dt = dt.astimezone(tz)
        except Exception:
            pass
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ts or "—"


def _video_path(alert: dict) -> str | None:
    """Locate the video file and return a path Gradio is allowed to serve."""
    vp = alert.get("video_path", "")
    if not vp:
        return None

    # Search candidates: absolute path as-is, then under MEDIA_DIR
    actual = None
    for c in [vp, os.path.join(MEDIA_DIR, vp.lstrip("/"))]:
        if os.path.exists(c):
            actual = c
            break

    if not actual:
        return None

    # Gradio 6 calls Path.resolve() (follows symlinks) before checking allowed_paths,
    # so symlinks are useless here.  Use a hard link instead — a hard link IS a real
    # file entry in _PREVIEW_DIR with no indirection to resolve.
    # If the video is on a different filesystem from /tmp (common), fall back to copy.
    dest = os.path.join(_PREVIEW_DIR, os.path.basename(actual))
    if not os.path.exists(dest):
        try:
            os.link(actual, dest)           # hard link — instant, zero extra space
        except OSError:
            shutil.copy2(actual, dest)      # cross-filesystem fallback
    return dest


# ─── Pagination ───────────────────────────────────────────────────────────────
def load_page(page: int, search: str, force_fetch: bool = False, tz: str = "UTC"):
    fetch_msg = ""
    if force_fetch or len(_store) == 0:
        fetch_msg = _fetch_redis_all()

    alerts = [a for a in _snapshot() if a.get("video_path")]

    q = search.strip().lower()
    if q:
        def _match(a):
            hay = " ".join([
                a.get("id", ""),
                a.get("sensor_id", ""),
                a.get("stream_name", "") or "",
                (a.get("alert") or {}).get("type", ""),
                (a.get("alert") or {}).get("description", "") or "",
                (a.get("result") or {}).get("description", "") or "",
                (a.get("result") or {}).get("reasoning", "") or "",
            ]).lower()
            return q in hay
        alerts = list(filter(_match, alerts))

    total       = len(alerts)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page        = max(1, min(int(page), total_pages))
    chunk       = alerts[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]

    rows = []
    for a in chunk:
        ai  = a.get("alert") or {}
        res = a.get("result") or {}
        sev = (ai.get("severity") or "").upper()
        vlm = (res.get("status")   or "").upper()
        ver = res.get("verification_result")
        rows.append([
            (a.get("id") or "")[:12],
            _fmt_ts(a.get("@timestamp", ""), tz),
            a.get("sensor_id",   "") or "—",
            a.get("stream_name", "") or "—",
            f"{SEV_ICON.get(sev,'⚫')} {sev}" if sev else "—",
            ai.get("type",   "") or "—",
            ai.get("status", "") or "—",
            f"{VLM_ICON.get(vlm,'⏳')} {vlm}" if vlm else "⏳ PENDING",
            "✅" if ver is True else ("❌" if ver is False else "—"),
        ])

    info = f"Page {page} / {total_pages}   ({total} alerts)"
    return rows, info, page, chunk, fetch_msg


# ─── Row selection ────────────────────────────────────────────────────────────
def select_alert(evt: gr.SelectData, page_alerts: list, tz: str = "UTC"):
    idx = evt.index[0]
    if not page_alerts or not (0 <= idx < len(page_alerts)):
        return None, "*Select a row to preview details.*", "{}", "No alert selected.", {}

    a   = page_alerts[idx]
    ai  = a.get("alert") or {}
    res = a.get("result") or {}
    sev = (ai.get("severity") or "").upper()
    vlm = (res.get("status")   or "").upper()
    ver = res.get("verification_result")

    md = f"""### Alert `{(a.get("id") or "")[:12]}…`

| | |
|:---|:---|
| **Timestamp** | {_fmt_ts(a.get("@timestamp", "—"), tz)} |
| **Sensor ID** | `{a.get("sensor_id", "—")}` |
| **Stream** | {a.get("stream_name", "—")} |

#### Alert Info
| | |
|:---|:---|
| **Type** | {ai.get("type", "—")} |
| **Severity** | {SEV_ICON.get(sev, "⚫")} {sev or "—"} |
| **Status** | {ai.get("status", "—")} |
| **Description** | {ai.get("description", "—")} |

#### VLM Analysis
| | |
|:---|:---|
| **Result** | {VLM_ICON.get(vlm, "⏳")} {vlm or "PENDING"} |
| **Verified** | {"✅ Yes" if ver is True else ("❌ No" if ver is False else "—")} |
| **Reviewed By** | {res.get("reviewed_by", "—")} |
| **Reviewed At** | {res.get("reviewed_at", "—")} |

#### Description
{res.get("description") or ai.get("description") or "*No description available*"}

#### Reasoning
{res.get("reasoning") or "*No reasoning available*"}
"""

    ctx = "\n".join([
        f"Alert ID   : {a.get('id', 'N/A')}",
        f"Type       : {ai.get('type','N/A')} | Severity: {sev}",
        f"Description: {ai.get('description','N/A')}",
        f"VLM Status : {vlm or 'PENDING'}",
        f"Analysis   : {res.get('description','N/A')}",
        f"Reasoning  : {res.get('reasoning','N/A')}",
        f"Verified   : {'Yes' if ver is True else ('No' if ver is False else 'N/A')}",
    ])

    return _video_path(a), md, json.dumps(a, indent=2, ensure_ascii=False), ctx, a


# ─── Chat ─────────────────────────────────────────────────────────────────────
_via_file_cache: dict[str, str] = {}   # local_path → VIA file_id
_via_chat_ready: set[str] = set()      # via_id → already summarized with enable_chat


def _get_or_upload_via_file(local_path: str) -> tuple[str | None, str]:
    """Return (VIA file_id, error_detail). Upload the file if not yet in VIA."""
    if not local_path:
        return None, "video_local_path is None (ไม่พบไฟล์วิดีโอในระบบ)"
    if not os.path.exists(local_path):
        return None, f"ไม่พบไฟล์: `{local_path}`"

    if local_path in _via_file_cache:
        return _via_file_cache[local_path], ""

    fname = os.path.basename(local_path)

    # Check if VIA already has a file with the same name
    try:
        r = requests.get(f"{VSS_URL}/files", timeout=5)
        if r.ok:
            payload = r.json()
            files = (payload if isinstance(payload, list)
                     else payload.get("files", payload.get("data", [])))
            for f in (files or []):
                if not isinstance(f, dict):
                    continue
                if fname in (f.get("filename") or f.get("name") or ""):
                    fid = f.get("id") or f.get("file_id")
                    if fid:
                        _via_file_cache[local_path] = fid
                        return fid, ""
    except Exception as e:
        pass  # non-fatal; proceed to upload

    # Upload the file
    try:
        with open(local_path, "rb") as fp:
            r = requests.post(
                f"{VSS_URL}/files",
                data={"purpose": "vision", "media_type": "video"},
                files={"file": (fname, fp, "video/mp4")},
                timeout=300,
            )
        if r.ok:
            data = r.json()
            fid = data.get("id") or data.get("file_id")
            if fid:
                _via_file_cache[local_path] = fid
                return fid, ""
            return None, f"Upload OK แต่ไม่มี id ใน response: {r.text[:300]}"
        return None, f"Upload failed HTTP {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return None, f"Upload exception: {e}"


def send_chat(user_msg: str, history: list, ctx: str, alert: dict):
    if not user_msg.strip():
        yield history, ""
        return

    history = history + [{"role": "user", "content": user_msg}]

    def _err(msg):
        return history + [{"role": "assistant", "content": msg}], ""

    # 1. Upload video to VIA if needed
    video_local_path = _video_path(alert) if alert else None
    via_id, upload_err = (_get_or_upload_via_file(video_local_path)
                          if video_local_path else (None, "ยังไม่ได้เลือก alert"))
    if not via_id:
        vp = (alert or {}).get("video_path", "")
        yield _err(
            f"⚠️ ไม่สามารถ upload วิดีโอไปยัง VIA ได้\n\n"
            f"**video_path:** `{vp or '(ไม่มี)'}`\n"
            f"**local_path:** `{video_local_path or '(ไม่พบ)'}`\n"
            f"**error:** {upload_err}"
        )
        return

    # 2. Query VLM via /summarize (streaming) — works without CA-RAG
    # Build prompt from conversation history + current question
    system_instruction = (
        "You are a concise security analyst assistant. "
        "Match your response length to the question: "
        "answer simple yes/no or factual questions in 1-2 sentences; "
        "provide full detail only when the user explicitly asks for analysis, explanation, or reasoning."
    )
    if ctx and ctx.strip() and ctx != "No alert selected.":
        prompt = f"{system_instruction}\n\nAlert context:\n{ctx}\n\nQuestion: {user_msg}"
    else:
        prompt = f"{system_instruction}\n\nQuestion: {user_msg}"

    bot = ""
    yield history + [{"role": "assistant", "content": "⏳ กำลังวิเคราะห์วิดีโอ…"}], ""
    try:
        with requests.post(
            f"{VSS_URL}/summarize",
            json={"id": via_id, "model": _get_vlm_model(),
                  "prompt": prompt, "stream": True, "max_tokens": 512,
                  "num_frames": 6},
            stream=True, timeout=600,
        ) as r:
            if not r.ok:
                yield _err(f"⚠️ API error HTTP {r.status_code}: {r.text[:300]}")
                return
            for line in r.iter_lines():
                if not line:
                    continue
                line = line if isinstance(line, str) else line.decode()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choice = chunk.get("choices", [{}])[0]
                    # non-streaming: full message at once
                    content = (choice.get("message") or {}).get("content", "")
                    # streaming delta fallback
                    if not content:
                        content = (choice.get("delta") or {}).get("content", "")
                    if content:
                        bot += content
                        yield history + [{"role": "assistant", "content": bot}], ""
                except Exception:
                    pass
    except Exception as e:
        yield _err(f"⚠️ Connection error: {e}")
        return

    if not bot:
        bot = "*(ไม่ได้รับคำตอบจาก VLM)*"
    yield history + [{"role": "assistant", "content": bot}], ""


# ─── Status bar ───────────────────────────────────────────────────────────────
def get_status() -> str:
    parts = []

    # Redis
    try:
        r = _redis_client()
        r.ping()
        length = r.xlen(STREAM_NAME)
        parts.append(f"🟢 Redis ({length} stream entries)")
    except Exception as e:
        parts.append(f"🔴 Redis: {e}")

    # Alert-Bridge
    try:
        r2 = requests.get(f"{ALERT_BRIDGE_HTTP}/health", timeout=3)
        parts.append("🟢 Alert-Bridge" if r2.ok else f"🟡 Alert-Bridge ({r2.status_code})")
    except Exception:
        parts.append("🔴 Alert-Bridge")

    # VSS
    try:
        r3 = requests.get(f"{VSS_URL}/health/live", timeout=3)
        parts.append("🟢 VSS" if r3.ok else f"🟡 VSS ({r3.status_code})")
    except Exception:
        parts.append("🔴 VSS")

    with _lock:
        parts.append(f"📊 {len(_store)} alerts cached")

    return "  |  ".join(parts)


# ─── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
footer { display: none !important; }

.app-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
    padding: 22px 32px;
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid #3a3a5c;
}

.status-bar textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    background: #111122 !important;
    color: #00ff9d !important;
    border-color: #334 !important;
    border-radius: 6px !important;
}

.fetch-info textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.78rem !important;
}

.page-counter input, .page-counter textarea {
    text-align: center !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    background: #f0f4ff !important;
}

.json-box textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.8rem !important;
}

.ctx-box textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.82rem !important;
}

/* Table hover */
.alert-tbl table tbody tr:hover { background: #e8eeff !important; cursor: pointer; }
/* Constrain table height */
.alert-tbl { max-height: 430px; overflow-y: auto; }
"""


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="🚨 Alert Inspector") as demo:

    # ── Shared state ──────────────────────────────────────────────────────────
    _page    = gr.State(1)
    _palerts = gr.State([])
    _ctx     = gr.State("No alert selected.")
    _alert   = gr.State({})
    _tz      = gr.State("UTC")

    # ── Header ───────────────────────────────────────────────────────────────
    with gr.Row(elem_classes="app-header"):
        with gr.Column(scale=3):
            gr.HTML("""
                <h1 style="color:#ddeeff;margin:0;font-size:1.75rem;font-weight:700;">
                    🚨 Alert Inspector
                </h1>
                <p style="color:#8899cc;margin:5px 0 0;font-size:0.92rem;">
                    Real-time security alert review &nbsp;•&nbsp; VSS Analysis Dashboard
                </p>
            """)
        with gr.Column(scale=2):
            status_box = gr.Textbox(
                value=get_status, every=30,
                interactive=False, show_label=False,
                elem_classes="status-bar",
            )

    # ── Main layout: fixed preview on left, tabs on right ────────────────────
    with gr.Row():

        # ── Left: always-visible preview panel ────────────────────────────
        with gr.Column(scale=7):
            video_out = gr.Video(
                label="📹  Video Preview",
                autoplay=True,
                height=270,
            )
            detail_md = gr.Markdown(
                "*Click a row in the table to preview the video and view alert details.*"
            )
            with gr.Accordion("🔧  Raw JSON", open=False):
                json_out = gr.Textbox(
                    value="{}", show_label=False,
                    lines=22, max_lines=50,
                    elem_classes="json-box",
                )

        # ── Right: tabs ───────────────────────────────────────────────────
        with gr.Column(scale=11):
            with gr.Tabs():

                # ═══════════════════ Tab 1: Alert List ═══════════════════
                with gr.TabItem("📋  Alert List"):
                    with gr.Row():
                        search_in   = gr.Textbox(
                            placeholder="🔍  Filter by sensor, type, description, ID…",
                            show_label=False, scale=5,
                        )
                        btn_search  = gr.Button("Search", scale=1, min_width=80)
                        btn_refresh = gr.Button("🔄  Refresh", variant="secondary",
                                                scale=1, min_width=90)

                    fetch_status_box = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Fetch status will appear here…",
                        elem_classes="fetch-info",
                    )

                    tbl = gr.Dataframe(
                        headers=["ID", "Timestamp", "Sensor", "Stream",
                                 "Severity", "Type", "Alert Status", "VLM Result", "Verified"],
                        datatype=["str"] * 9,
                        interactive=False,
                        elem_classes="alert-tbl",
                    )

                    with gr.Row():
                        btn_prev = gr.Button("◀  Prev",  variant="secondary",
                                             scale=1, min_width=90)
                        pg_info  = gr.Textbox(
                            value="—", show_label=False, interactive=False,
                            scale=4, elem_classes="page-counter",
                        )
                        btn_next = gr.Button("Next  ▶", variant="secondary",
                                             scale=1, min_width=90)

                # ═══════════════════ Tab 2: Chat ══════════════════════════
                with gr.TabItem("💬  Chat with VLM"):
                    with gr.Row():

                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(label="Conversation", height=460)
                            with gr.Row():
                                chat_in  = gr.Textbox(
                                    placeholder="Ask about the selected alert or any surveillance question…",
                                    show_label=False, scale=5,
                                )
                                btn_send = gr.Button("Send ➤", variant="primary",
                                                     scale=1, min_width=80)
                            btn_clr = gr.Button("🗑️  Clear conversation", variant="secondary")

                        with gr.Column(scale=1):
                            gr.Markdown("### 📌  Alert Context")
                            gr.Markdown(
                                "Select an alert in the **Alert List** tab to provide "
                                "context to the VLM chat."
                            )
                            ctx_box = gr.Textbox(
                                value="No alert selected.",
                                interactive=False, lines=18, show_label=False,
                                elem_classes="ctx-box",
                            )

    # ═══ Event wiring ════════════════════════════════════════════════════════

    OUTPUTS = [tbl, pg_info, _page, _palerts, fetch_status_box]

    def _load(page, search, force=False, tz="UTC"):
        return load_page(page, search, force_fetch=force, tz=tz)

    # Initial load — detect browser timezone + force fetch from Redis
    def _init(tz: str = "UTC"):
        rows, info, page, chunk, msg = load_page(1, "", force_fetch=True, tz=tz)
        return tz, rows, info, page, chunk, msg

    demo.load(
        fn=_init,
        outputs=[_tz] + OUTPUTS,
        js="() => [Intl.DateTimeFormat().resolvedOptions().timeZone]",
    )

    # Refresh — force fetch
    btn_refresh.click(fn=lambda s, tz: _load(1, s, force=True, tz=tz),
                      inputs=[search_in, _tz], outputs=OUTPUTS)

    # Search — filter existing store
    btn_search.click(fn=lambda s, tz: _load(1, s, tz=tz),
                     inputs=[search_in, _tz], outputs=OUTPUTS)
    search_in.submit(fn=lambda s, tz: _load(1, s, tz=tz),
                     inputs=[search_in, _tz], outputs=OUTPUTS)

    # Pagination
    btn_prev.click(fn=lambda pg, s, tz: _load(max(1, pg - 1), s, tz=tz),
                   inputs=[_page, search_in, _tz], outputs=OUTPUTS)
    btn_next.click(fn=lambda pg, s, tz: _load(pg + 1, s, tz=tz),
                   inputs=[_page, search_in, _tz], outputs=OUTPUTS)

    # Row click → video + details + sync chat context
    tbl.select(fn=select_alert,
               inputs=[_palerts, _tz],
               outputs=[video_out, detail_md, json_out, _ctx, _alert])
    _ctx.change(fn=lambda c: c, inputs=[_ctx], outputs=[ctx_box])

    # Chat
    btn_send.click(fn=send_chat,
                   inputs=[chat_in, chatbot, _ctx, _alert],
                   outputs=[chatbot, chat_in])
    chat_in.submit(fn=send_chat,
                   inputs=[chat_in, chatbot, _ctx, _alert],
                   outputs=[chatbot, chat_in])
    btn_clr.click(fn=lambda: [], outputs=[chatbot])


# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        allowed_paths=[MEDIA_DIR, _PREVIEW_DIR],
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )
