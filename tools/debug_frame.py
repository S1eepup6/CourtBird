"""Side-by-side check: source frame with detections coloured by team + top view."""
import argparse
import cv2
import numpy as np
from courtbird.court import Calibration
from courtbird.court.players import foot_points
from courtbird.perception.detector import PersonDetector
from courtbird.render import TopViewRenderer
from courtbird.render.topview import TEAM_COLORS, UNKNOWN_TEAM
from courtbird.schema import FrameState, Person
from courtbird.sports import SPORTS
from tools.overlay import draw_court

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("calib"); ap.add_argument("out")
ap.add_argument("--sec", type=float, default=5); ap.add_argument("--sport", default="volleyball")
a = ap.parse_args()
spec = SPORTS[a.sport]; calib = Calibration.load(a.calib, spec)
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.sec * 1000)
ok, f = cap.read()
boxes = PersonDetector("rtmdet-m-person")(f)
feet, okmask = foot_points(boxes, f.shape[0])
court = calib.image_to_court(feet)
state = FrameState(); n = 0
for b, (x, y), k in zip(boxes, court, okmask):
    inside = k and -8 <= x <= spec.length + 8 and -2 <= y <= spec.width + 2
    team = spec.team_of(x)
    col = TEAM_COLORS[team] if inside else (128, 128, 128)
    x1, y1, x2, y2 = map(int, b[:4]); cv2.rectangle(f, (x1, y1), (x2, y2), col, 2)
    if inside:
        n += 1; state.people.append(Person(n, float(x), float(y), team))
f = draw_court(f, calib, spec)
vis, hid = calib.visibility(spec); state.hidden_regions = hid
r = TopViewRenderer(spec); top = r.render(state)
top = cv2.resize(top, (f.shape[1], int(top.shape[0] * f.shape[1] / top.shape[1])))
cv2.imwrite(a.out, np.vstack([f, top]))
print("kept", n, "of", len(boxes))
