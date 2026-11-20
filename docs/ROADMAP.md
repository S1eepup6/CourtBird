# CourtBird roadmap (status as of 2026-11-20)

## The six original requirements

| # | Requirement | Status |
|---|---|---|
| 1 | Players on different sides of the net get different colours | Done. Team = side of the net the *track* spends most time on. Weak for players glued to the net. |
| 2 | People vs ball in different colours; ball thicker/lighter | People done. Ball drawing exists in the renderer; **ball detection is deferred** (nothing produces balls yet). |
| 3 | Accurate person recognition despite overlaps | Mostly done (RTMDet + Kalman/Hungarian tracker + appearance + stitching). **Not measured against ground truth.** |
| 4 | Show only what is recognised when the camera moves / part of the court is visible | Done (camera-motion compensation, visible/hidden regions, shading). **Validated on synthetic motion only.** |
| 5 | Detector and renderer are separate modules joined by positions | Done (`FrameState`; renderer takes court-space positions). |
| 6 | Recognise automatically where the video was shot from | Viewpoint estimates done (from calibration, and calibration-free). **Court calibration itself is still manual** (click tool + snapping). |

## Known gaps, by priority

### P0 - blocks trusting the output
1. **Accurate calibration of the test video.** The one in use was eyeballed (far-side players land outside the
   court). Needs careful clicks with `tools.calibrate` (then `r` to snap). Everything downstream inherits this.
2. **Ground truth for tracking.** No ID-accuracy number exists; only proxies (id count, track length). Plan: let a
   hand-corrected record (made in the viewer) serve as ground truth and add an evaluation tool (ID switches,
   position error vs corrected record).
3. **Full-length run never done.** Only 20 s clips were analysed. Run the 11.7 min match end to end and check time
   (~8 min expected with rtmdet-m), memory, record size and viewer responsiveness.

### P1 - quality and robustness
4. **Record size**: `visible_region`/`hidden_regions` are stored on every frame; a full match will bloat the file.
   Store once and only on change.
5. **Records can't be re-calibrated.** Only court coordinates are saved. Store each person's foot pixel so a
   corrected calibration can re-project an existing record without re-running detection.
6. **More footage.** One venue, fixed camera. Need other venues (different floor colours / lines) and a genuinely
   moving camera to find out what really breaks.
7. **Players at the net / same-team ID swaps.** Team by net side flips near the net; same-kit players are only
   separated by motion. Ideas: team by kit-colour clusters, stronger re-identification, pose cues.
8. **Non-player filtering** is heuristic (outside the court + barely moving). Referees/line judges that move, benches,
   liberos and substitutes need checking on more footage.

### P2 - features
9. **Automatic court detection from scratch.** Classical line detection failed on the test venue (court = coloured
   floor area, strong white lines belong to another sport). Options: learned court-keypoint model (needs labelled data
   and a licence check), or view-class templates + snapping. Per-video calibration is one-time, so speed is not the
   constraint.
10. **Ball detection (deferred by decision)** and height estimation. Researched path: WASB-SBDT (MIT code; the volleyball
    weights' training data terms are unclear) or train our own; viewer ball-labelling mode; 3D via parabola + camera pose.
11. **Broadcast footage with cuts** (scene changes): motion tracking restarts at a cut; needs cut detection and
    re-calibration per shot.

### P3 - polish and release
12. ~~Choose the project licence and add third-party notices~~ Done: Apache-2.0, see `LICENSE` and
    `THIRD_PARTY_NOTICES.md` (first public release, 0.1.0).
13. Other sports: add `SportSpec`s for badminton/tennis (court lines, singles/doubles, net height) and check landmarks.
14. Viewer: export top-view video/GIF, ball labelling, font check on Windows, packaging (installer or `pipx`).
15. Repo hygiene before going public: CI running the tests, pinned dependencies, setup instructions for non-Windows /
    CPU-only, remove machine-specific paths from docs.

## Decisions waiting on the owner
- Whether to spend time labelling (calibration points, IDs, later the ball) - items 1, 2 and 10 depend on it.
- More video: another venue and a moving-camera clip (item 6).
- Which sport after volleyball (item 13).
