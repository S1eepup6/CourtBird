"""Download detector weights (not committed to git) and print SHA-256 for the record."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
MODELS = {
    # person-only RTMDet (COCO + Objects365), Apache-2.0 code from OpenMMLab
    "rtmdet-nano-person": "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip",
    "rtmdet-m-person": "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip",
}
DEST = Path(__file__).resolve().parent.parent / "models"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> Path:
    out = DEST / name
    if (out / "end2end.onnx").exists():
        return out
    DEST.mkdir(exist_ok=True)
    zpath = DEST / MODELS[name]
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(BASE + MODELS[name], zpath)
    print(f"  zip sha256 {sha256(zpath)}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    zpath.unlink()
    return out


if __name__ == "__main__":
    for n in (sys.argv[1:] or MODELS):
        print(fetch(n))
