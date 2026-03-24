######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
######################################################################################################
"""
Alert Inspector UI — rebuilt
Paginated alert table, inline video preview, and VLM chat.
All configuration is read from environment variables (set via .env / compose).
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

import gradio as gr
import requests

# ─── Configuration from environment (no changes to compose needed) ────────────
ALERT_BRIDGE_WS_URL  = os.environ.get("ALERT_BRIDGE_BASE_URL", "ws://alert-bridge:9080")
ALERT_BRIDGE_HTTP    = (ALERT_BRIDGE_WS_URL
                        .replace("ws://",  "http://")
                        .replace("wss://", "https://"))
BACKEND_IP           = os.environ.get("BACKEND_IP",   "via-server")
BACKEND_PORT         = os.environ.get("BACKEND_PORT", "8000")
VSS_URL              = f"http://{BACKEND_IP}:{BACKEND_PORT}"
MEDIA_DIR            = os.environ.get("ALERT_REVIEW_MEDIA_BASE_DIR", "/tmp/alerts")
GRADIO_HOST          = os.environ.get("GRADIO_SERVER", "0.0.0.0")
GRADIO_PORT          = int(os.environ.get("GRADIO_PORT", "7860"))
PAGE_SIZE            = 20


# ─── In-memory alert store ─────────────────────────────────────────────────────
_store: list[dict] = []
_lock  = threading.Lock()
_MAX   = 10_000


def _upsert(alert: dict) -> None:
    """Insert or update an alert by id (newest-first order)."""
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


# ─── Background: poll REST API for historical alerts ───────────────────────────
def _poll_rest() -> None:
    while True:
        try:
            r = requests.get(f"{ALERT_BRIDGE_HTTP}/api/v1/alerts", timeout=10)
            if r.ok:
                payload = r.json()
                items = (payload if isinstance(payload, list)
                         else payload.get("alerts", payload.get("results", [])))
                for item in reversed(items):
                    if isinstance(item, dict):
                        _upsert(item)
        except Exception:
            pass
        time.sleep(30)


# ─── Background: WebSocket listener for real-time updates ─────────────────────
def _ws_loop() -> None:
    try:
        import websocket  # optional
    except ImportError:
        return

    paths = ["/ws", "/ws/alerts", "/api/v1/ws", ""]
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


threading.Thread(target=_poll_rest, daemon=True, name="rest-poller").start()
threading.Thread(target=_ws_loop,   daemon=True, name="ws-listener").start()


# ─── Utilities ─────────────────────────────────────────────────────────────────
SEV_ICON = {
    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡",
    "LOW": "🟢", "INFORMAL": "⚪",
}
VLM_ICON = {"SUCCESS": "✅", "FAILURE": "❌"}


def _fmt_ts(ts: str) -> str:
    try:
        return (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                .strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        return ts or "—"


def _video_path(alert: dict) -> str | None:
    vp = alert.get("video_path", "")
    if not vp:
        return None
    for candidate in [vp, os.path.join(MEDIA_DIR, vp.lstrip("/"))]:
        if os.path.exists(candidate):
            return candidate
    return None


# ─── Pagination logic ──────────────────────────────────────────────────────────
def load_page(page: int, search: str):
    """Return (table_rows, page_info_str, new_page_int, page_alerts_list)."""
    alerts = _snapshot()

    q = search.strip().lower()
    if q:
        def _match(a):
            haystack = " ".join([
                a.get("id", ""),
                a.get("sensor_id", ""),
                a.get("stream_name", ""),
                (a.get("alert") or {}).get("type", ""),
                (a.get("alert") or {}).get("description", ""),
                (a.get("result") or {}).get("description", ""),
                (a.get("result") or {}).get("reasoning", ""),
            ]).lower()
            return q in haystack
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
        vlm = (res.get("status") or "").upper()
        ver = res.get("verification_result")
        rows.append([
            (a.get("id") or "")[:12],
            _fmt_ts(a.get("@timestamp", "")),
            a.get("sensor_id", "") or "—",
            a.get("stream_name", "") or "—",
            f"{SEV_ICON.get(sev, '⚫')} {sev}" if sev else "—",
            ai.get("type", "") or "—",
            ai.get("status", "") or "—",
            f"{VLM_ICON.get(vlm, '⏳')} {vlm}" if vlm else "⏳ PENDING",
            "✅" if ver is True else ("❌" if ver is False else "—"),
        ])

    info = f"Page {page} / {total_pages}   ({total} alerts)"
    return rows, info, page, chunk


# ─── Row-selection handler ────────────────────────────────────────────────────
def select_alert(evt: gr.SelectData, page_alerts: list):
    """Called when user clicks a row; returns video, detail markdown, json, ctx."""
    idx = evt.index[0]
    if not page_alerts or not (0 <= idx < len(page_alerts)):
        return None, "*Select an alert row to preview details.*", "{}", "No alert selected."

    a   = page_alerts[idx]
    ai  = a.get("alert") or {}
    res = a.get("result") or {}
    sev = (ai.get("severity") or "").upper()
    vlm = (res.get("status") or "").upper()
    ver = res.get("verification_result")

    md = f"""### Alert `{(a.get("id") or "")[:12]}…`

| | |
|:---|:---|
| **Timestamp** | {_fmt_ts(a.get("@timestamp", "—"))} |
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
        f"Type       : {ai.get('type', 'N/A')} | Severity: {sev}",
        f"Description: {ai.get('description', 'N/A')}",
        f"VLM Status : {vlm or 'PENDING'}",
        f"Analysis   : {res.get('description', 'N/A')}",
        f"Reasoning  : {res.get('reasoning', 'N/A')}",
        f"Verified   : {'Yes' if ver is True else ('No' if ver is False else 'N/A')}",
    ])

    return (
        _video_path(a),
        md,
        json.dumps(a, indent=2, ensure_ascii=False),
        ctx,
    )


# ─── Chat handler ──────────────────────────────────────────────────────────────
def send_chat(user_msg: str, history: list, ctx: str):
    if not user_msg.strip():
        return history, ""

    sys_prompt = (
        "You are a helpful video surveillance security analysis assistant. "
        "Answer questions clearly and professionally."
    )
    if ctx and ctx.strip() and ctx != "No alert selected.":
        sys_prompt += f"\n\nCurrently reviewing this alert:\n{ctx}"

    msgs = [{"role": "system", "content": sys_prompt}]
    for u, b in history:
        msgs.append({"role": "user",      "content": u})
        if b:
            msgs.append({"role": "assistant", "content": b})
    msgs.append({"role": "user", "content": user_msg})

    try:
        r = requests.post(
            f"{VSS_URL}/chat/completions",
            json={"model": "local", "messages": msgs,
                  "max_tokens": 1024, "temperature": 0.2},
            timeout=120,
        )
        if r.ok:
            bot = r.json()["choices"][0]["message"]["content"]
        else:
            bot = f"API error: HTTP {r.status_code}\n{r.text[:200]}"
    except Exception as e:
        bot = f"Connection error: {e}"

    return history + [(user_msg, bot)], ""


# ─── Status bar ───────────────────────────────────────────────────────────────
def get_status() -> str:
    parts = []
    for label, url in [
        ("Alert-Bridge", f"{ALERT_BRIDGE_HTTP}/health"),
        ("VSS Engine",   f"{VSS_URL}/health/live"),
    ]:
        try:
            r = requests.get(url, timeout=3)
            parts.append(f"🟢 {label}" if r.ok else f"🟡 {label} ({r.status_code})")
        except Exception:
            parts.append(f"🔴 {label}")
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

/* Make table rows look clickable */
.alert-tbl table tbody tr:hover { background: #e8eeff !important; cursor: pointer; }
"""


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🚨 Alert Inspector",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=CSS,
) as demo:

    # ── Shared state ──────────────────────────────────────────────────────────
    _page    = gr.State(1)
    _palerts = gr.State([])   # alerts on current page (for row-select lookup)
    _ctx     = gr.State("No alert selected.")   # chat context from selected alert

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

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ═══════════════════════════ Tab 1: Alert List ═══════════════════════
        with gr.TabItem("📋  Alert List"):
            with gr.Row():

                # ── Left column: search + table + pagination ──────────────
                with gr.Column(scale=11):
                    with gr.Row():
                        search_in   = gr.Textbox(
                            placeholder="🔍  Filter by sensor, type, description, alert ID…",
                            show_label=False, scale=5,
                        )
                        btn_search  = gr.Button("Search",      scale=1, min_width=80)
                        btn_refresh = gr.Button("🔄  Refresh", variant="secondary",
                                                scale=1, min_width=90)

                    tbl = gr.Dataframe(
                        headers=["ID", "Timestamp", "Sensor", "Stream",
                                 "Severity", "Type", "Alert Status", "VLM Result", "Verified"],
                        datatype=["str"] * 9,
                        interactive=False,
                        wrap=False,
                        height=430,
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

                # ── Right column: video preview + alert details ───────────
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

        # ═══════════════════════════ Tab 2: Chat ═════════════════════════════
        with gr.TabItem("💬  Chat with VLM"):
            with gr.Row():

                # ── Chat panel ────────────────────────────────────────────
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=460,
                        show_copy_button=True,
                        bubble_full_width=False,
                    )
                    with gr.Row():
                        chat_in  = gr.Textbox(
                            placeholder="Ask about the selected alert or any surveillance question…",
                            show_label=False, scale=5,
                        )
                        btn_send = gr.Button("Send ➤", variant="primary",
                                             scale=1, min_width=80)
                    btn_clr = gr.Button("🗑️  Clear conversation", variant="secondary")

                # ── Alert context sidebar ─────────────────────────────────
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

    # ═══ Event wiring ═════════════════════════════════════════════════════════

    def _load(page, search):
        rows, info, pg, pa = load_page(page, search)
        return rows, info, pg, pa

    # Initial load
    demo.load(
        fn=lambda: _load(1, ""),
        outputs=[tbl, pg_info, _page, _palerts],
    )

    # Refresh
    btn_refresh.click(
        fn=lambda s: _load(1, s),
        inputs=[search_in],
        outputs=[tbl, pg_info, _page, _palerts],
    )

    # Search
    btn_search.click(
        fn=lambda s: _load(1, s),
        inputs=[search_in],
        outputs=[tbl, pg_info, _page, _palerts],
    )
    search_in.submit(
        fn=lambda s: _load(1, s),
        inputs=[search_in],
        outputs=[tbl, pg_info, _page, _palerts],
    )

    # Pagination
    btn_prev.click(
        fn=lambda pg, s: _load(max(1, pg - 1), s),
        inputs=[_page, search_in],
        outputs=[tbl, pg_info, _page, _palerts],
    )
    btn_next.click(
        fn=lambda pg, s: _load(pg + 1, s),
        inputs=[_page, search_in],
        outputs=[tbl, pg_info, _page, _palerts],
    )

    # Row click → video + details + update chat context
    tbl.select(
        fn=select_alert,
        inputs=[_palerts],
        outputs=[video_out, detail_md, json_out, _ctx],
    ).then(
        fn=lambda c: c,
        inputs=[_ctx],
        outputs=[ctx_box],
    )

    # Chat
    btn_send.click(
        fn=send_chat,
        inputs=[chat_in, chatbot, _ctx],
        outputs=[chatbot, chat_in],
    )
    chat_in.submit(
        fn=send_chat,
        inputs=[chat_in, chatbot, _ctx],
        outputs=[chatbot, chat_in],
    )
    btn_clr.click(fn=lambda: [], outputs=[chatbot])


# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        allowed_paths=[MEDIA_DIR],
        share=False,
    )
