# SKU Recognition — DeepStream Line Crossing Pipeline

Detects items passing on a conveyor belt using NVIDIA DeepStream. When an object crosses a configured line, the pipeline captures a JPEG image (crop + full frame) for downstream SKU recognition.

## Overview

```
nvurisrcbin → nvstreammux → nvinferserver (GDino/Triton)
            → nvtracker → nvvideoconvert → nvdsanalytics
            → [line crossing probe]  ← captures here
            → nvdsosd → fakesink / xvimagesink
```

- **Detection**: GroundingDINO (open-vocabulary) via Triton Inference Server, or a custom ONNX/TRT model
- **Tracking**: NVIDIA Multi-Object Tracker with ResNet50 Re-ID
- **Line crossing**: Centroid-based geometric detection — no pyds dependency
- **Capture**: Background OpenCV thread reads the same source; JPEG files are written asynchronously

## Requirements

- NVIDIA GPU (Ampere or newer recommended)
- Docker + NVIDIA Container Toolkit
- Internet access on first run (model download)

## Quick Start

```bash
# Copy and edit environment variables
cp .env.example .env
# Edit .env: set SKU_SOURCE to your RTSP URL or video file path

# Build image
docker compose build

# Run
docker compose up
```

Captured images are saved to `./captures/` by default.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SKU_SOURCE` | *(required)* | Video source — RTSP URL or `file:///path/to/video.mp4` |
| `SKU_OUTPUT_DIR` | `/output/captures` | Output folder for captured JPEG images |
| `SKU_PROMPT` | `product . box . bottle . can . package` | GroundingDINO detection prompt |
| `SKU_THRESHOLD` | `0.3` | Detection confidence threshold |
| `SKU_LINE_NAME` | `SKU_LINE` | Line name — must match `config_nvdsanalytics.txt` |
| `SKU_COOLDOWN` | `2.0` | Minimum seconds between captures of the same object |
| `SKU_STREAM_NAME` | `conveyor` | Label used in output filenames |
| `USE_GDINO` | `true` | Use GroundingDINO (`true`) or custom model (`false`) |
| `USE_CUSTOM_MODEL` | `false` | Use custom ONNX/TRT model in `custom_model/` |
| `ENABLE_DISPLAY` | `false` | Show live output window (requires `DISPLAY` env) |
| `TRITON_TIMEOUT` | `300` | Seconds to wait for Triton to become ready |

## Configuring the Line

Edit `config_nvdsanalytics.txt`:

```ini
[line-crossing-stream-0]
enable=1
line-crossing-SKU_LINE=1800;200;1800;1500
```

Format: `line-crossing-{NAME}=x1;y1;x2;y2`

Set `config-width` and `config-height` to match your source resolution (default: 3840×2160).

The pipeline reads line coordinates from this file automatically — no code changes needed.

## Output Files

Each crossing event produces two files:

| File | Content |
|---|---|
| `sku_{timestamp}_id{N}_{label}_crop.jpg` | Cropped bounding box (padded 15%) |
| `sku_{timestamp}_id{N}_{label}_full.jpg` | Full frame with bounding box and line overlay |

## Using a Custom Model

1. Place your ONNX model in `custom_model/`
2. Edit `custom_model/nvinfer_config.yaml` with model path and input/output names
3. Edit `custom_model/labels.txt` with your class names
4. Set `USE_CUSTOM_MODEL=true` and `USE_GDINO=false` in `.env`

## First Run Notes

On first run, the startup script will:

1. Download the Re-ID model (~50 MB) from NGC
2. Download and convert the GroundingDINO model to TensorRT (~1–3 GB, 10–30 min depending on GPU)

Both are cached in a Docker volume and reused on subsequent runs.

## File Structure

```
sku-recognition/
├── sku_pipeline.py          # Main DeepStream pipeline
├── simple_config_updater.py # Updates GDino prompt in config file
├── config_nvdsanalytics.txt # Line crossing definition
├── gdinoconfig_grpc.txt     # Triton gRPC connection config
├── via_tracker_config_fast.yml  # NvMultiObjectTracker config
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .env.example
└── custom_model/            # Drop custom ONNX model here
```
