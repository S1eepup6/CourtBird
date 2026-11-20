# Third-party license review

Checked 2026-11-20.

| Component | License | Source |
|---|---|---|
| RTMDet (MMDetection code) | Apache-2.0 | https://github.com/open-mmlab/mmdetection/blob/main/LICENSE |
| rtmlib (RTMDet/RTMPose inference, no mmcv) | Apache-2.0 | https://github.com/Tau-J/rtmlib/blob/main/LICENSE |
| OpenCV (`opencv-python`) | Apache-2.0 | |

Apache-2.0 permits redistribution and commercial use. Unlike YOLO/ultralytics (AGPL-3.0), it
does not force this project's own license.

## Open points

- **Weights**: the official COCO checkpoints are published by OpenMMLab under the repo's
  Apache-2.0 license, but no separate per-weight license text was found. COCO annotations are
  CC BY 4.0 and the images carry Flickr licenses; this is the usual basis for COCO-pretrained
  weights, not a restriction we found, but it is not legal advice.
- **Hosting risk**: one secondary source (a Pose2Sim Zenodo record) claims the OpenMMLab weight
  servers were shut down on 2026-06-06. On 2026-11-20 the RTMDet-m weight URL still returned
  HTTP 200. Plan: download weights via script, record the SHA-256, and do not commit weights.
- Redistribute the Apache-2.0 LICENSE/NOTICE texts if weights or code are ever bundled.

## Person detector weights (added M1)

`tools/download_models.py` fetches the OpenMMLab person-only RTMDet ONNX exports
(`rtmdet-nano-person`, `rtmdet-m-person`; trained on COCO + Objects365). Zip SHA-256 at download
time (2026-11-20):

- nano: `cb8f6d545a963167acb5a7c98307dcf8a46f7b451ee7e1d8abf2a1f81c26f59d`
- m: `716262a99aec134232e2420ae2b1c5bd4a5c0c9a0bd5484067cd1920f881f32e`

Objects365 is an additional training set whose terms were not reviewed here.
