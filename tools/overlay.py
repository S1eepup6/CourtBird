"""Draw the calibrated court (and free zone) over a video frame to check a calibration."""
import argparse
import cv2
import numpy as np
from courtbird.court import Calibration
from courtbird.sports import SPORTS


def draw_court(frame, calib, spec, color=(0, 255, 255)):
    def poly(pts, c, t=2):
        p = calib.court_to_image(np.array(pts)).astype(np.int32)
        cv2.polylines(frame, [p], True, c, t, cv2.LINE_AA)
    L, W, m = spec.length, spec.width, spec.free_zone
    poly([(0, 0), (L, 0), (L, W), (0, W)], color)
    for a, b in spec.lines:
        poly([a, b], color, 1)
    poly([(L / 2, -spec.post_offset), (L / 2, W + spec.post_offset)], (0, 0, 255), 2)
    poly([(-m, -m), (L + m, -m), (L + m, W + m), (-m, W + m)], (255, 0, 255), 1)
    for n, x, y in calib.points:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), 2)
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("calib"); ap.add_argument("out")
    ap.add_argument("--sec", type=float, default=5); ap.add_argument("--sport", default="volleyball")
    a = ap.parse_args()
    spec = SPORTS[a.sport]; calib = Calibration.load(a.calib, spec)
    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.sec * 1000)
    ok, f = cap.read()
    cv2.imwrite(a.out, draw_court(f, calib, spec))
    print("reprojection error (m):", round(calib.reprojection_error(spec), 4))
