"""Person detection with RTMDet (ONNX, via rtmlib's session). Pixel-space output only.

The three steps are exposed separately (preprocess / run / postprocess) so the pipeline can
overlap CPU work with GPU inference. Preprocessing is our own float32 version of rtmlib's
(letterbox to a square, BGR mean/std) because rtmlib's float64 path costs ~10 ms per frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import RTMDet

try:  # load cuDNN/cuBLAS from the pip nvidia-* wheels
    ort.preload_dlls()
except Exception:
    pass

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# model name -> square input size baked into the ONNX file
INPUT_SIZES = {"rtmdet-nano-person": 320, "rtmdet-m-person": 640}


def find_onnx(name: str) -> Path:
    hits = list((MODELS_DIR / name).rglob("end2end.onnx"))
    if not hits:
        raise FileNotFoundError(f"{name} not found; run tools/download_models.py {name}")
    return hits[0]


class PersonDetector:
    """Returns an (N, 5) float array of [x1, y1, x2, y2, score] in image pixels.

    `score_thr` is deliberately low: the tracker uses the score to split detections into
    confident ones and weak ones (weak boxes can still keep an occluded player's track alive).
    """

    def __init__(self, model: str = "rtmdet-m-person", device: str = "cuda",
                 score_thr: float = 0.1, nms_thr: float = 0.45):
        self.size = INPUT_SIZES[model]
        self.det = RTMDet(str(find_onnx(model)), model_input_size=(self.size, self.size),
                          backend="onnxruntime", device=device, score_thr=score_thr, nms_thr=nms_thr)
        self.score_thr = score_thr
        self.mean = np.asarray(self.det.mean, np.float32)
        self.inv_std = (1.0 / np.asarray(self.det.std, np.float32)).astype(np.float32)
        sess = self.det.session
        self._in = sess.get_inputs()[0].name
        self._outs = [o.name for o in sess.get_outputs()]

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """BGR frame -> (1, 3, S, S) float32 tensor and the resize ratio."""
        h, w = frame_bgr.shape[:2]
        S = self.size
        ratio = min(S / h, S / w)
        nh, nw = int(h * ratio), int(w * ratio)
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.astype(np.float32)
        x -= self.mean
        x *= self.inv_std
        return np.ascontiguousarray(x.transpose(2, 0, 1))[None], ratio

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self.det.session.run(self._outs, {self._in: tensor})[0]

    def postprocess(self, out: np.ndarray, ratio: float) -> np.ndarray:
        if out.shape[-1] != 5:
            raise ValueError(f"unexpected detector output {out.shape}; expected baked-in NMS (…, 5)")
        boxes = out[0].astype(np.float32)
        boxes = boxes[boxes[:, 4] > self.score_thr]
        boxes[:, :4] /= ratio
        return boxes

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        tensor, ratio = self.preprocess(frame_bgr)
        return self.postprocess(self.run(tensor), ratio)
