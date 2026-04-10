#!/bin/bash
# SKU Recognition - DeepStream Pipeline Startup Script

set -e

WORKDIR="/opt/nvidia/sku-recognition"
TRITON_LOG="/tmp/via-logs/triton.log"
# Triton defaults: HTTP=8000, gRPC=8001 — ต้องตรงกับ gdinoconfig_grpc.txt (url: "localhost:8001")
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_TIMEOUT="${TRITON_TIMEOUT:-300}"   # รอ Triton ได้สูงสุด 5 นาที

cd "$WORKDIR"
mkdir -p /tmp/via-logs/

# ---------------------------------------------------------------------------
# 1. Install proprietary codecs (optional)
# ---------------------------------------------------------------------------
if [ "$INSTALL_PROPRIETARY_CODECS" = "true" ]; then
    if ! command -v ffmpeg_for_overlay_video >/dev/null 2>&1; then
        echo "[SKU] Installing multimedia packages..."
        apt-get update -qq
        apt-get install -y --no-install-recommends gstreamer1.0-plugins-ugly libx264-dev
    fi
fi

# ---------------------------------------------------------------------------
# 2. Install pyservicemaker wheel
# ---------------------------------------------------------------------------
arch=$(uname -m)
echo "[SKU] Installing pyservicemaker ($arch)..."
if [ "$arch" = "aarch64" ]; then
    pip3 install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-py3-none-linux_*.whl \
        --force-reinstall --no-deps --quiet
else
    pip3 install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-cp312-cp312-linux_x86_64.whl \
        --force-reinstall --no-deps --quiet
fi

# ---------------------------------------------------------------------------
# 3. Re-ID model สำหรับ tracker
# ---------------------------------------------------------------------------
mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
mkdir -p ~/.via/ngc_model_cache/cv_models/triton_model_repo/gdino_trt/1/

if [ ! -f ~/.via/ngc_model_cache/cv_models/resnet50_market1501.etlt ]; then
    echo "[SKU] Downloading Re-ID model (tracker)..."
    wget --progress=bar:force \
        'https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt' \
        -P ~/.via/ngc_model_cache/cv_models/
fi
cp ~/.via/ngc_model_cache/cv_models/resnet50_market1501.etlt \
   /opt/nvidia/deepstream/deepstream/samples/models/Tracker/

# ---------------------------------------------------------------------------
# 4. GroundingDINO + Triton (ถ้าใช้ USE_GDINO=true)
# ---------------------------------------------------------------------------
if [ "${USE_GDINO:-true}" = "true" ]; then

    # --- Download/convert model (ทำครั้งเดียว แล้ว cache ไว้ใน volume) ---
    if [ ! -f ~/.via/ngc_model_cache/cv_models/triton_model_repo/gdino_trt/1/model.plan ]; then
        echo "[SKU] ============================================"
        echo "[SKU] GroundingDINO model not found in cache."
        echo "[SKU] Downloading + converting to TensorRT engine."
        echo "[SKU] (ครั้งแรกใช้เวลานาน 10-30 นาที ขึ้นกับ GPU)"
        echo "[SKU] ============================================"
        cp /opt/nvidia/TritonGdino/download_convert_model.sh ~/.via/ngc_model_cache/cv_models/
        cp /opt/nvidia/TritonGdino/update_model.py            ~/.via/ngc_model_cache/cv_models/
        cd ~/.via/ngc_model_cache/cv_models && bash download_convert_model.sh
        cd "$WORKDIR"
        echo "[SKU] Model ready."
    else
        echo "[SKU] GDino model found in cache, skipping download."
    fi

    cp -r ~/.via/ngc_model_cache/cv_models/triton_model_repo/gdino_trt/* \
          /opt/nvidia/TritonGdino/triton_model_repo/gdino_trt/

    # --- Start Triton ---
    echo "[SKU] Starting Triton (HTTP :${TRITON_HTTP_PORT}, gRPC :${TRITON_GRPC_PORT})..."
    tritonserver \
        --model-repository=/opt/nvidia/TritonGdino/triton_model_repo \
        --strict-model-config=false \
        --grpc-infer-allocation-pool-size=16 \
        --exit-on-error=true \
        --http-port "${TRITON_HTTP_PORT}" \
        --grpc-port "${TRITON_GRPC_PORT}" \
        > "$TRITON_LOG" 2>&1 &
    TRITON_PID=$!
    disown

    # --- Wait for Triton พร้อม progress และ timeout ---
    echo "[SKU] Waiting for Triton to be ready (timeout: ${TRITON_TIMEOUT}s)..."
    echo "[SKU] Log: $TRITON_LOG"
    elapsed=0
    last_log_line=0
    while true; do
        # เช็ค health
        if curl --silent --fail --connect-timeout 1 \
                "http://localhost:${TRITON_HTTP_PORT}/v2/health/ready" > /dev/null 2>&1; then
            echo ""
            echo "[SKU] Triton is ready! (took ${elapsed}s)"
            break
        fi

        # เช็คว่า Triton process ยังมีอยู่ไหม
        if ! kill -0 "$TRITON_PID" 2>/dev/null; then
            echo ""
            echo "[SKU] ERROR: Triton process died. Log:"
            tail -30 "$TRITON_LOG"
            exit 1
        fi

        # timeout
        if [ "$elapsed" -ge "$TRITON_TIMEOUT" ]; then
            echo ""
            echo "[SKU] ERROR: Triton did not become ready within ${TRITON_TIMEOUT}s."
            echo "[SKU] Last log lines:"
            tail -20 "$TRITON_LOG"
            exit 1
        fi

        # แสดง log ใหม่ที่เพิ่มขึ้นมา (tail -f style แบบ non-blocking)
        current_lines=$(wc -l < "$TRITON_LOG" 2>/dev/null || echo 0)
        if [ "$current_lines" -gt "$last_log_line" ]; then
            tail -n +"$((last_log_line + 1))" "$TRITON_LOG" | \
                sed 's/^/  [triton] /'
            last_log_line=$current_lines
        else
            printf "."
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done
fi

# ---------------------------------------------------------------------------
# 5. Create output folder
# ---------------------------------------------------------------------------
OUTPUT_DIR="${SKU_OUTPUT_DIR:-/tmp/sku/captures}"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 6. Run SKU pipeline
# ---------------------------------------------------------------------------
PYTHONPATH="$WORKDIR:$PYTHONPATH"

echo "[SKU] ========================================"
echo "[SKU] Starting sku_pipeline.py"
echo "[SKU]   SOURCE      = ${SKU_SOURCE}"
echo "[SKU]   OUTPUT      = ${OUTPUT_DIR}"
echo "[SKU]   PROMPT      = ${SKU_PROMPT:-product . box . bottle . can . package}"
echo "[SKU]   THRESHOLD   = ${SKU_THRESHOLD:-0.3}"
echo "[SKU]   LINE_NAME   = ${SKU_LINE_NAME:-SKU_LINE}"
echo "[SKU]   COOLDOWN    = ${SKU_COOLDOWN:-2.0}"
echo "[SKU]   USE_GDINO   = ${USE_GDINO:-true}"
echo "[SKU] ========================================"

export GST_DEBUG="GST_ELEMENT_FACTORY:3,nvinferserver:3,nvdsanalytics:5,nvtracker:3"

python3 sku_pipeline.py \
    --source      "${SKU_SOURCE}" \
    --output      "${OUTPUT_DIR}" \
    --prompt      "${SKU_PROMPT:-product . box . bottle . can . package}" \
    --threshold   "${SKU_THRESHOLD:-0.3}" \
    --line-name   "${SKU_LINE_NAME:-SKU_LINE}" \
    --cooldown    "${SKU_COOLDOWN:-2.0}" \
    --stream-name "${SKU_STREAM_NAME:-conveyor}"
