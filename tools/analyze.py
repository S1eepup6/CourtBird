"""python -m tools.analyze VIDEO CALIB OUT.cbrec [--start S] [--end S] [--stride N] [--model M]"""
import argparse, time
from tqdm import tqdm
from courtbird import record
from courtbird.court import Calibration
from courtbird.pipeline import analyze
from courtbird.sports import SPORTS

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("calib"); ap.add_argument("out")
ap.add_argument("--sport", default="volleyball"); ap.add_argument("--model", default="rtmdet-m-person")
ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float, default=None)
ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args()
spec = SPORTS[a.sport]; calib = Calibration.load(a.calib, spec)
bar = tqdm(total=None, unit="f")
t = time.perf_counter()
def prog(i, n):
    bar.total = n; bar.n = i; bar.refresh()
rec = analyze(a.video, calib, spec, a.model, a.stride, a.start, a.end, prog)
bar.close()
record.save(rec, a.out)
print(f"{len(rec.frames)} frames in {time.perf_counter()-t:.1f}s -> {a.out}")
