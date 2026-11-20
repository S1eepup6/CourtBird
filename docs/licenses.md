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
