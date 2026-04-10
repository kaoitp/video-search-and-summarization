FROM nvcr.io/nvidia/blueprint/nv-cv-event-detector:2.4.0

# ---- Copy SKU recognition application files ----
WORKDIR /opt/nvidia/sku-recognition

COPY sku_pipeline.py          ./
COPY simple_config_updater.py ./
COPY config_nvdsanalytics.txt ./
COPY gdinoconfig_grpc.txt     ./
COPY via_tracker_config_fast.yml ./
COPY requirements.txt         ./
COPY start_sku_pipeline.sh    ./

# ---- System packages ----
# xvimagesink สำหรับแสดงผลผ่าน X11 / SSH forwarding
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        gstreamer1.0-plugins-good \
        gstreamer1.0-x \
        libxv1 && \
    rm -rf /var/lib/apt/lists/*

# ---- Install extra Python dependencies ----
# numpy<2 required: torch in this image was compiled against NumPy 1.x
RUN pip3 install --no-cache-dir opencv-python-headless "numpy<2"

# ---- Entrypoint ----
RUN chmod +x start_sku_pipeline.sh
ENTRYPOINT ["/opt/nvidia/sku-recognition/start_sku_pipeline.sh"]
