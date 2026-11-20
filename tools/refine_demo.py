"""python -m tools.refine_demo VIDEO SEED.json OUT.png [--sec 5] : seed vs refined calibration overlay."""
import argparse
import cv2
import numpy as np
from courtbird.court import Calibration
from courtbird.court.draw import draw_court
from courtbird.court.refine import line_evidence, refine_calibration
from courtbird.perception.detector import PersonDetector
from courtbird.sports import SPORTS

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("seed"); ap.add_argument("out")
ap.add_argument("--sec", type=float, default=5); ap.add_argument("--sport", default="volleyball")
a = ap.parse_args()
spec = SPORTS[a.sport]; seed = Calibration.load(a.seed, spec)
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.sec * 1000); f = cap.read()[1]
boxes = PersonDetector("rtmdet-nano-person")(f); boxes = boxes[boxes[:, 4] > 0.3]
import time; t = time.perf_counter()
fit = refine_calibration(seed, f, spec, boxes)
print(f"refine {time.perf_counter()-t:.2f}s  cost {fit.seed_cost:.2f} -> {fit.cost:.2f} px  coverage {fit.coverage:.2f}")
ev = line_evidence(f, boxes)
left = draw_court(f.copy(), seed, spec, show_points=False)
right = draw_court(f.copy(), fit.calib, spec, show_points=False)
right[ev > 0] = (0, 255, 0)
for img, txt in ((left, "seed"), (right, "refined (green = line evidence)")):
    cv2.putText(img, txt, (20, 40), 0, 1.0, (0, 255, 255), 2)
cv2.imwrite(a.out, np.vstack([left[200:720], right[200:720]]))
