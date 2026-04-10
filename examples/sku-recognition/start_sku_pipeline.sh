#!/bin/bash
# SKU Recognition - DeepStream Pipeline Startup Script
# =====================================================
# Script นี้ทำงานเหมือนกับ start_nv_cv_event_detector.sh แต่ปรับสำหรับ SKU recognition:
#   1. ติดตั้ง codec (ถ้าต้องการ)
#   2. Download และ prepare model files
#   3. Start Triton server (สำหรับ GroundingDINO)
#   4. Run sku_pipeline.py

set -e

WORKDIR="/opt/nvidia/sku-recognition"
cd "$WORKDIR"

# ---------------------------------------------------------------------------
# 1. Install proprietary codecs (optional)
# ---------------------------------------------------------------------------
if [ "$INSTALL_PROPRIETARY_CODECS" = "true" ]; then
    if ! command -v ffmpeg_for_overlay_video >/dev/null 2>&1; then
        echo "[SKU] Installing multimedia packages..."
        apt-get update -qq
        apt-get install -y --no-install-recommends \
            gstreamer1.0-plugins-ugly libx264-dev
    fi
fi

# ---------------------------------------------------------------------------
# 2. Install pyservicemaker wheel (ต้องทำทุก container start เพราะ wheel อยู่ใน image)
# ---------------------------------------------------------------------------
arch=$(uname -m)
echo "[SKU] Installing pyservicemaker for $arch..."
if [ "$arch" = "aarch64" ]; then
    pip3 install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-py3-none-linux_*.whl \
        --force-reinstall --no-deps --quiet
else
    pip3 install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-cp312-cp312-linux_x86_64.whl \
        --force-reinstall --no-deps --quiet
fi

# ---------------------------------------------------------------------------
# 3. Prepare model directories
# ---------------------------------------------------------------------------
mkdir -p /tmp/via-logs/
mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
mkdir -p ~/.via/ngc_model_cache/cv_models/triton_model_repo/gdino_trt/1/

# Re-ID model (สำหรับ tracker)
if [ ! -f ~/.via/ngc_model_cache/cv_models/resnet50_market1501.etlt ]; then
    echo "[SKU] Downloading Re-ID model..."
    wget -q 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt' \
        -P ~/.via/ngc_model_cache/cv_models/
fi
cp ~/.via/ngc_model_cache/cv_models/resnet50_market1501.etlt \
   /opt/nvidia/deepstream/deepstream/samples/models/Tracker/

# ---------------------------------------------------------------------------
# 4. Prepare GroundingDINO model (ถ้าใช้ USE_GDINO=true)
# ---------------------------------------------------------------------------
if [ "${USE_GDINO:-true}" = "true" ]; then
    if [ ! -f ~/.via/ngc_model_cache/cv_models/triton_model_repo/gdino_trt/1/model.plan ]; then
        echo "[SKU] Downloading and converting GroundingDINO model..."
        cp /opt/nvidia/TritonGdino/download_convert_model.sh ~/.via/ngc_model_cache/cv_models/
        cp /opt/nvidia/TritonGdino/update_model.py            ~/.via/ngc_model_cache/cv_models/
        cd ~/.via/ngc_model_cache/cv_models && bash download_convert_model.sh
        cd "$WORKDIR"
    fi

    cp -r ~/.via/ngc_model_cache/cv_models/triton_model_repo/gdino_trt/* \
          /opt/nvidia/TritonGdino/triton_model_repo/gdino_trt/

    echo "[SKU] Starting Triton server for GroundingDINO..."
    nohup tritonserver \
        --model-repository=/opt/nvidia/TritonGdino/triton_model_repo \
        --strict-model-config=false \
        --grpc-infer-allocation-pool-size=16 \
        --exit-on-error=true \
        --http-port 8001 \
        > /tmp/via-logs/triton.log 2>&1 &

    disown

    echo "[SKU] Waiting for Triton to be ready..."
    until curl --silent --fail --connect-timeout 1 "http://localhost:8001/v2/health/ready" > /dev/null 2>&1; do
        echo "[SKU]   ... still waiting for Triton"
        sleep 2
    done
    echo "[SKU] Triton is ready."
fi

# ---------------------------------------------------------------------------
# 5. Create output folder
# ---------------------------------------------------------------------------
OUTPUT_DIR="${SKU_OUTPUT_DIR:-/tmp/sku/captures}"
mkdir -p "$OUTPUT_DIR"
echo "[SKU] Capture output folder: $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 6. Run SKU pipeline
# ---------------------------------------------------------------------------
PYTHONPATH="$WORKDIR:$PYTHONPATH"

echo "[SKU] Starting sku_pipeline.py..."
echo "[SKU]   SOURCE      = ${SKU_SOURCE}"
echo "[SKU]   OUTPUT      = ${OUTPUT_DIR}"
echo "[SKU]   PROMPT      = ${SKU_PROMPT:-product . box . bottle . can . package}"
echo "[SKU]   THRESHOLD   = ${SKU_THRESHOLD:-0.3}"
echo "[SKU]   LINE_NAME   = ${SKU_LINE_NAME:-SKU_LINE}"
echo "[SKU]   COOLDOWN    = ${SKU_COOLDOWN:-2.0}"
echo "[SKU]   STREAM_NAME = ${SKU_STREAM_NAME:-conveyor}"

python3 sku_pipeline.py \
    --source      "${SKU_SOURCE}" \
    --output      "${OUTPUT_DIR}" \
    --prompt      "${SKU_PROMPT:-product . box . bottle . can . package}" \
    --threshold   "${SKU_THRESHOLD:-0.3}" \
    --line-name   "${SKU_LINE_NAME:-SKU_LINE}" \
    --cooldown    "${SKU_COOLDOWN:-2.0}" \
    --stream-name "${SKU_STREAM_NAME:-conveyor}"
