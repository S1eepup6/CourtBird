"""Click-based court calibration.

    python -m tools.calibrate VIDEO OUT.json [--sec 5] [--landmarks a,b,c,...]

Click each requested landmark in the frame (a magnifier follows the cursor).
Keys:  n skip landmark   u / right-click undo   +/- zoom magnifier   r snap to lines/edges
       Enter save (needs >= 4)   q quit

`r` re-fits the four court corners to the line/colour-boundary evidence in the frame (each corner
may move <= 35 px), which tidies up hand-clicked corners. It replaces your points by the 4 refined
corners; press `u` to take your clicks back.
Once 4+ points exist the court is overlaid live and per-point errors (metres) are shown, so a
misplaced click stands out. More points than 4 are fitted by least squares.
"""
import argparse
import sys

import cv2
import numpy as np

from courtbird.court import Calibration, landmarks
from courtbird.court.refine import refine_calibration
from courtbird.sports import SPORTS
from courtbird.court.draw import draw_court

DEFAULT = ["corner_near_left", "corner_near_right", "post_right", "post_left",
           "corner_far_right", "corner_far_left"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--sec", type=float, default=5.0)
    ap.add_argument("--sport", default="volleyball")
    ap.add_argument("--landmarks", default=",".join(DEFAULT))
    a = ap.parse_args()
    spec = SPORTS[a.sport]
    names = a.landmarks.split(",")
    unknown = [n for n in names if n not in landmarks(spec)]
    if unknown:
        print("unknown landmarks:", unknown, "\navailable:", list(landmarks(spec))); return 1

    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.sec * 1000)
    ok, frame = cap.read()
    if not ok:
        print("cannot read frame"); return 1
    h, w = frame.shape[:2]
    pts: list[tuple[str, float, float]] = []
    state = {"i": 0, "mouse": (0, 0), "zoom": 4}
    win = "CourtBird calibrate"

    def current():
        return names[state["i"]] if state["i"] < len(names) else None

    def advance():
        state["i"] += 1

    def on_mouse(ev, x, y, flags, _):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN and current():
            pts.append((current(), float(x), float(y))); advance()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            undo()

    snapshot: list = []  # the points as clicked, restored by `u` right after `r`

    def undo():
        if snapshot:
            pts[:] = list(snapshot)
            snapshot.clear()
            state["i"] = len(pts) if len(pts) < len(names) else len(names)
            return
        if pts:
            name = pts.pop()[0]
            state["i"] = names.index(name)

    def render():
        img = frame.copy()
        calib = None
        if len(pts) >= 4:
            try:
                calib = Calibration.from_points(spec, pts, (w, h)); draw_court(img, calib, spec)
            except ValueError:
                pass
        for n, x, y in pts:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(img, n, (int(x) + 8, int(y) - 8), 0, 0.5, (0, 255, 0), 1)
        if calib:
            lm = landmarks(spec)
            for n, x, y in pts:
                g = calib.image_to_court([[x, y]])[0]; e = float(np.hypot(*(g - np.array(lm[n]))))
                cv2.putText(img, f"{e:.2f}m", (int(x) + 8, int(y) + 14), 0, 0.5, (0, 200, 255), 1)
            cv2.putText(img, f"mean err {calib.reprojection_error(spec):.2f} m", (10, h - 12), 0, 0.7, (0, 255, 255), 2)
        msg = f"click: {current()}" if current() else "all landmarks done - Enter to save"
        cv2.putText(img, msg, (10, 28), 0, 0.8, (0, 0, 0), 4); cv2.putText(img, msg, (10, 28), 0, 0.8, (255, 255, 255), 2)
        mx, my = state["mouse"]; z = state["zoom"]; r = 50
        x0, y0 = max(0, min(w - 2 * r, mx - r)), max(0, min(h - 2 * r, my - r))
        crop = cv2.resize(frame[y0:y0 + 2 * r, x0:x0 + 2 * r], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        cx, cy = (mx - x0) * z, (my - y0) * z
        cv2.line(crop, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1); cv2.line(crop, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)
        ph, pw = crop.shape[:2]
        img[h - ph - 40:h - 40, w - pw - 10:w - 10] = crop
        return img, calib

    cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
    while True:
        img, calib = render(); cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return 0
        if k == ord("n") and current():
            advance()
        elif k == ord("u"):
            undo()
        elif k in (ord("+"), ord("=")):
            state["zoom"] = min(8, state["zoom"] + 1)
        elif k == ord("-"):
            state["zoom"] = max(2, state["zoom"] - 1)
        elif k == ord("r") and calib:
            fit = refine_calibration(calib, frame, spec)
            snapshot[:] = list(pts)
            pts[:] = list(fit.calib.points)
            state["i"] = len(names)
            print(f"refined: cost {fit.seed_cost:.2f} -> {fit.cost:.2f} px")
        elif k in (13, 10) and calib:
            calib.save(a.out); print("saved", a.out, "mean err (m):", round(calib.reprojection_error(spec), 3)); return 0


if __name__ == "__main__":
    sys.exit(main())
