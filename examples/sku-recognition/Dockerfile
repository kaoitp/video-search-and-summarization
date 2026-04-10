FROM nvcr.io/nvidia/blueprint/nv-cv-event-detector:2.4.0

# ---- Copy SKU recognition application files ----
WORKDIR /opt/nvidia/sku-recognition

COPY sku_pipeline.py        ./
COPY config_nvdsanalytics.txt ./
COPY requirements.txt       ./
COPY start_sku_pipeline.sh  ./

# Copy GStreamer plugin
COPY gst-plugins/gstskucapture.py \
     /opt/nvidia/deepstream/deepstream/service-maker/gst-plugins/python/

# ---- Install extra Python dependencies ----
RUN pip3 install --no-cache-dir opencv-python-headless numpy

# ---- Entrypoint ----
RUN chmod +x start_sku_pipeline.sh
ENTRYPOINT ["/opt/nvidia/sku-recognition/start_sku_pipeline.sh"]
