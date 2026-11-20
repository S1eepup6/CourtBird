# Third-party notices

CourtBird itself is licensed under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

**No third-party code, model weights or data are copied into this repository.** Python packages are
installed from PyPI by the user, model weights are downloaded by `tools/download_models.py`, and the test
video used during development is not included. This file lists what CourtBird uses and under which terms,
as far as they could be established (licence data below was read from the installed package metadata and the
upstream licence files on 2026-11-20; versions are the ones the project was developed and tested with). It is
not legal advice - check the upstream licences before redistributing anything.

## 1. Python packages CourtBird depends on directly

| Package | Tested version | Licence | Notes |
|---|---|---|---|
| [NumPy](https://numpy.org) | 2.2.6 | BSD-3-Clause | |
| [SciPy](https://scipy.org) | 1.15.3 | BSD-3-Clause | Hungarian assignment (`linear_sum_assignment`), optimisation |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | 5.1.0.95 | Apache-2.0 | The wheels bundle further libraries (e.g. FFmpeg under LGPL-2.1+, libpng, ...). Their licences are in the wheel's `LICENSE-3RD-PARTY.txt`. |
| [ONNX Runtime](https://onnxruntime.ai) (`onnxruntime-gpu` or `onnxruntime`) | 1.23.2 | MIT | The wheel ships a `ThirdPartyNotices.txt`. |
| [rtmlib](https://github.com/Tau-J/rtmlib) | 0.0.16 | Apache-2.0 | Provides the RTMDet ONNX wrapper and its image/model helpers. Installed with `--no-deps` (see README). |
| [tqdm](https://tqdm.github.io) | 4.70.1 | MPL-2.0 AND MIT | |
| [Qt for Python - PySide6-Essentials / shiboken6](https://pyside.org) | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | Qt itself is also available under these terms. CourtBird imports `QtCore`, `QtGui`, `QtWidgets` (and `QtTest` in tests) and links to them dynamically through the unmodified PyPI wheels, so the library can be replaced by installing another version. CourtBird uses Qt under the **LGPL-3.0**. |

Development only: [pytest](https://docs.pytest.org) 9.2.0 (MIT).

## 2. Packages installed alongside them (transitive, as seen in the development environment)

| Package | Version | Licence |
|---|---|---|
| colorama | 0.4.6 | BSD |
| coloredlogs | 15.0.1 | MIT |
| exceptiongroup | 1.3.2 | MIT |
| flatbuffers | 25.12.19 | Apache-2.0 |
| humanfriendly | 10.0 | MIT |
| iniconfig | 2.3.1 | MIT |
| mpmath | 1.3.0 | BSD |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.7.0 | MIT |
| protobuf | 7.36.2 | BSD-3-Clause |
| Pygments | 2.21.0 | BSD-2-Clause |
| pyreadline3 | 3.5.6 | BSD |
| sympy | 1.14.0 | BSD |
| tomli | 2.5.0 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |

`rtmlib` declares requirements on `onnxruntime` (CPU) and on two OpenCV builds (`opencv-python` and
`opencv-contrib-python`, both Apache-2.0). CourtBird does not use them that way; see the README for the install
commands that avoid duplicates.

## 3. NVIDIA libraries (optional, for GPU inference)

To run ONNX Runtime on an NVIDIA GPU the README tells users to install, from PyPI, `nvidia-cudnn-cu12`,
`nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cufft-cu12`,
`nvidia-curand-cu12` and `nvidia-nvjitlink-cu12`. These are **proprietary NVIDIA software**, distributed under
NVIDIA's own licence terms (accepted by the user on installation), and are never redistributed by CourtBird.

## 4. Model weights (downloaded, not included)

`tools/download_models.py` downloads person-detection models from OpenMMLab's servers
(`rtmdet-nano-person`, `rtmdet-m-person`: RTMDet exported to ONNX, trained on COCO and Objects365).

- RTMDet / MMDetection code: Apache-2.0 (<https://github.com/open-mmlab/mmdetection/blob/main/LICENSE>,
  copyright OpenMMLab). Paper: Lyu et al., "RTMDet: An Empirical Study of Designing Real-Time Object Detectors", 2022.
- The weights carry no separate licence text of their own; they are published under the repository's Apache-2.0
  licence. They were trained on COCO (annotations CC BY 4.0, images under their Flickr licences) and
  Objects365 (terms not reviewed). Details and the SHA-256 of the downloaded archives: [docs/licenses.md](docs/licenses.md).
- Please keep the OpenMMLab attribution and the Apache-2.0 licence text if you redistribute the weights.

## 5. Ideas and methods this project builds on (reimplemented, no code copied)

- **ByteTrack**-style two-stage association (confident detections first, then weak ones against unmatched tracks):
  Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
- Constant-velocity **Kalman filtering** and the **Hungarian algorithm** (via SciPy) for tracking.
- **Focal length and camera pose from a plane homography** using the orthogonality/equal-length constraints of the
  plane's axes: Z. Zhang, "A Flexible New Technique for Camera Calibration", IEEE TPAMI 2000.
- **Lucas-Kanade optical flow**, **RANSAC** homography estimation and **SIFT** feature matching as implemented in OpenCV
  (Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004).
- Horizon and camera height from the linear relation between a standing person's pixel height and foot row on a
  ground plane (classic single-view metrology).

Other projects were evaluated while planning (WASB-SBDT, TrackNet/TrackNetV3, the Volleyball group-activity dataset)
but **no code, weights or data from them are used**.

## 6. Test footage

The match video used during development is not part of this repository and was not authored by the project.
The README screenshot contains a single frame of it with the scoreboard area obscured.
