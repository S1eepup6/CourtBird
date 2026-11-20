"""Person detection with RTMDet (ONNX, via rtmlib). Pixel-space output only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from rtmlib import RTMDet

try:  # load cuDNN/cuBLAS from the pip nvidia-* wheels
    ort.preload_dlls()
except Exception:
    pass

# model name -> square input size baked into the ONNX file
INPUT_SIZES = {"rtmdet-nano-person": 320, "rtmdet-m-person": 640}

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def find_onnx(name: str) -> Path:
    hits = list((MODELS_DIR / name).rglob("end2end.onnx"))
    if not hits:
        raise FileNotFoundError(f"{name} not found; run tools/download_models.py {name}")
    return hits[0]


class PersonDetector:
    """Returns an (N, 5) float array of [x1, y1, x2, y2, score] in image pixels."""

    def __init__(self, model: str = "rtmdet-m-person", device: str = "cuda",
                 score_thr: float = 0.3, nms_thr: float = 0.45):
        input_size = INPUT_SIZES[model]
        self.det = RTMDet(str(find_onnx(model)), model_input_size=(input_size, input_size),
                          backend="onnxruntime", device=device,
                          score_thr=score_thr, nms_thr=nms_thr)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = self.det(frame_bgr)
        boxes = np.asarray(out, dtype=np.float32).reshape(-1, out.shape[-1] if len(out) else 4)
        return boxes
