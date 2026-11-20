# CourtBird

Turns net-sport match video (volleyball first) into a top-view court diagram.

Pipeline: video -> detect/track people and ball -> court calibration (classical line
detection, manual click fallback) -> court-space record -> top-view renderer / viewer.

Status: M0 done (schema, volleyball spec, top-view renderer, record format).

Court coordinates are metres: x along the long axis (net at L/2), y across.
Perception and rendering only talk through `courtbird.schema.FrameState`.

```
pip install -e .[dev]
pytest
```

## Setup (Windows, NVIDIA GPU)

```
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev] rtmlib onnxruntime-gpu tqdm
.venv\Scripts\python -m pip install nvidia-cudnn-cu12==9.10.2.21 nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-runtime-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cufft-cu12==11.3.3.83 nvidia-curand-cu12==10.3.9.90 nvidia-nvjitlink-cu12==12.8.93
.venv\Scripts\python tools/download_models.py
```

Install only `onnxruntime-gpu` (not `onnxruntime` alongside it) or CUDA will not load. The
newest cuDNN (9.28) failed with onnxruntime 1.23.2; the pins above work.

Detector benchmark on the 1280x720 test video, RTX 3060: rtmdet-nano 105 fps, rtmdet-m 31 fps
(end-to-end per frame, including CPU preprocessing).
