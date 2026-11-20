import sys, time
import cv2
from courtbird.perception.detector import PersonDetector

video = sys.argv[1]; model = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30 * 120)  # skip to ~2 min in
det = PersonDetector(model)
frames = [cap.read()[1] for _ in range(n)]
det(frames[0])  # warmup
t = time.perf_counter()
res = [det(f) for f in frames]
dt = time.perf_counter() - t
print(f"{model}: {n/dt:.1f} fps, mean persons/frame {sum(len(r) for r in res)/n:.1f}, out cols {res[0].shape}")
for b in res[0]:
    x1,y1,x2,y2 = map(int,b[:4]); cv2.rectangle(frames[0],(x1,y1),(x2,y2),(0,255,0),2)
if len(sys.argv) > 4:  # optional: write an annotated frame (keep it out of git: it is a video frame)
    cv2.imwrite(sys.argv[4], frames[0])
