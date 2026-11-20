"""Draw the calibrated court (and free zone) over a video frame to check a calibration."""
import argparse
import cv2
import numpy as np
from courtbird.court import Calibration
from courtbird.court.draw import draw_court
from courtbird.sports import SPORTS


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
