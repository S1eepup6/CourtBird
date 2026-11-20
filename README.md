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

## Usage (M2)

```
.venv\Scripts\python -m tools.calibrate VIDEO calib.json      # click court landmarks (once per camera position)
.venv\Scripts\python -m tools.overlay VIDEO calib.json out.png --sec 30   # check the calibration
.venv\Scripts\python -m tools.analyze VIDEO calib.json out.cbrec --start 0 --end 60
.venv\Scripts\python -m tools.debug_frame VIDEO calib.json out.png --sec 30
```

How tracking works (`courtbird/perception/`): detections are tracked in **image space**, so identity
does not depend on the court calibration (a calibration fixed later by hand does not invalidate
tracks). Kalman filter on the foot point + colour-histogram appearance + ByteTrack-style two-stage
Hungarian matching; distances are measured in person heights. Offline, tracklets are stitched
across occlusions, then projected through the calibration, non-players are dropped (outside the
court, or standing still beside it), gaps are filled/smoothed, and each track's team is the side
of the net it spends most of its time on.

Measured on the 20 s test clip: ids 193 -> 36, median track length 1.5 s -> 17 s. Speed (RTX 3060,
1280x720, decode/inference/tracking in overlapping threads): ~43 fps with rtmdet-m, ~78 fps with
rtmdet-nano; `--stride 2` halves the time again. Check a record with `python -m tools.track_metrics`.

Current limits: one fixed calibration per video (the camera may drift slightly), foot-point
projection only, no ball yet. Everything downstream of the calibration is only as good as the
calibration: click the landmarks carefully with `tools.calibrate` (a calibration eyeballed from
a still frame put far-side players outside the court in the test video).

## Viewer (M-V1)

```
.venv\Scripts\python -m courtbird.viewer [record.cbrec]
```

- File > Open record: view a saved `.cbrec` instantly (the source video is auto-linked when it sits
  next to / one folder above the record; File > Link video / calibration otherwise).
- File > Analyze video: runs detection in the background, then opens the result.
- Edit the record in the top view: drag to move, double-click to add a player, right-click to place
  the ball (right-click on the ball removes it), Delete, `T` team (`Shift+T` for all following
  frames), `I` rename id (optionally for all following frames), Ctrl+Z undo, Ctrl+S save.
- With a calibration linked, the source video panel shows the court and every recorded player
  re-projected onto the picture, which makes calibration errors easy to spot.
- "팀 영역 표시": tints each half of the court (video and top view) in the colour of the team standing
  there and labels it BLUE / RED, so the video can be compared with the top view. Records store their
  calibration, so this works as soon as a record is opened. Edit > swap teams (Ctrl+T) exchanges the
  colours for the whole record when blue turns out to be the other side.

## Calibration refinement (line/edge snapping)

`courtbird/court/refine.py` re-fits a rough calibration to the image: it builds "line evidence"
(thin bright ridges + strong colour-region boundaries, people masked out), scores a calibration by
how close the projected court lines are to that evidence, and optimises the four court corners
(each may move <= 35 px, so another sport's lines further away cannot capture the fit).
Use it from the calibration tool (`r` key) or `python -m tools.refine_demo VIDEO SEED.json out.png`.

Honest limits (measured on the test video, one venue): it tidies a hand-placed calibration and
follows slow camera drift (a 5 s calibration carried to 60-600 s kept hugging the court edges), but
it is not precise: seeds perturbed by 15 px converge to within ~23 px of each other, not to one
answer, and it cannot find the court from scratch. That venue's court is a coloured floor area
(no white volleyball lines; the strong white lines belong to another sport), so a line-only detector
fails there. A from-scratch detector would need venue-independent cues (colour region + vanishing
geometry, or a learned keypoint model).

## Camera motion (pan / tilt / zoom)

`courtbird/court/camera_motion.py` estimates the camera motion of every frame from the static
background (grid-spread corner features tracked from a *keyframe*, RANSAC homography, people
masked out). Detections are mapped back to the calibration view before tracking, so the tracker and
the single court calibration work in one fixed frame; the per-frame motion is stored in the record
(`FrameState.camera`) and the viewer carries the court overlay through it. Fixed cameras snap to the
identity (about 12% slower; `--no-camera` / the dialog checkbox turns it off). A calibration made at
another time than the analysis start is carried over by direct SIFT registration (`calibrate`
stores the frame time in the file).

What was measured, and what was not:
- Chaining frame-to-frame homographies drifted 45 px (30 s) to 289 px (10 min) on a *static* camera,
  so tracking is done against keyframes: 0.3-1 px from direct SIFT registration over 10 minutes.
- Spreading the features over a grid (so the weakly textured floor constrains the fit) cut the
  error at the court corners from ~31 px to ~4 px on known synthetic pan+zoom.
- Real match clip warped with a known pan/zoom, analysed with vs without compensation: players land
  0.17 m (median) from the fixed-camera result with it, 1.19 m without.
- The only real video available has a fixed camera, so the motion path was validated on synthetic
  motion applied to real frames, not on genuine handheld footage. Assumes mostly rotation about the
  camera centre (distant background); large camera translation, very fast pans or a background with
  no texture will degrade it.
