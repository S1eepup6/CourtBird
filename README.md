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
