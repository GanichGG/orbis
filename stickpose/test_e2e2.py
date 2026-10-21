"""Сквозная проверка v2: тестовое видео с известной разметкой -> stick2r6.py (окна заменены «пользователем»,
кликающим точно ±2 px) -> метрики: сколько вопросов, ошибка в долях длины торса, ошибка углов R6."""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

prefix = sys.argv[1]
weights = sys.argv[2]
extra = sys.argv[3:]
GT = np.load(prefix + "_gt.npy")          # (n, chars, 7, 2) в пикселях исходного кадра 960x540
n, nc = GT.shape[:2]
for ext in (".s2r.json", ".r6.json"):
    if os.path.exists(prefix + ext):
        os.remove(prefix + ext)

import colorseg as CS  # noqa: E402

rng = np.random.default_rng(5)
stats = {"asked": 0, "needless": 0}
J = ["head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot"]
fake = types.ModuleType("ui")
fake.busy = lambda *a, **k: None


def pick_colors(get, nf, preview, sample, start=0):
    fr = get(0)
    cols, pts = [], []
    for ci in range(nc):
        p = GT[0, ci, 2]  # таз — точно на теле
        cols.append([sample(fr, p[0], p[1], r=1).tolist()])
        pts.append((float(p[0]), float(p[1])))
    return 0, cols, pts


def label_frame(frame, cand, est, title, reason, ci, progress=""):
    stats["asked"] += 1
    f = int(title.split("кадр ")[1].split("/")[0]) - 1
    g = GT[f, ci]
    if est is not None:
        e = np.array([est[j] for j in J])
        T = np.linalg.norm(g[1] - g[2])
        err = min(np.linalg.norm(e - g, axis=1).max(), np.linalg.norm(e[[0, 1, 2, 4, 3, 6, 5]] - g, axis=1).max()) / T
        if err < 0.15:
            stats["needless"] += 1
    return "ok", {j: (g[i] + rng.normal(0, 2, 2)).tolist() for i, j in enumerate(J)}


fake.pick_colors = pick_colors
fake.label_frame = label_frame
captured = {}


def review(get, nf, results, start=0):
    captured["results"] = results
    return "export", 0, 0


fake.review = review
sys.modules["ui"] = fake
import stick2r6  # noqa: E402

sys.argv = ["stick2r6.py", prefix + ".mp4", "--weights", weights] + extra
stick2r6.main()
res = captured["results"]
print(f"\nвопросов: {stats['asked']} (из них оценка уже была верной: {stats['needless']}), кадров×персонажей: {n * nc}")
import track2  # noqa: E402

for ci in range(nc):
    P = res[ci]["P"]
    g = GT[:, ci]
    T = np.linalg.norm(g[:, 1] - g[:, 2], axis=1)
    e = np.linalg.norm(P - g, axis=2) / T[:, None]
    # углы частей R6 (направления сегментов)
    segs = [(2, 1), (1, 0), (1, 3), (1, 4), (2, 5), (2, 6)]
    ang = []
    for a, b in segs:
        v1, v2 = P[:, b] - P[:, a], g[:, b] - g[:, a]
        d = np.degrees(np.abs((np.arctan2(v1[:, 0], v1[:, 1]) - np.arctan2(v2[:, 0], v2[:, 1]) + np.pi) % (2 * np.pi) - np.pi))
        ang.append(d)
    ang = np.stack(ang, 1)
    print(f"перс {ci + 1}: ошибка точки ср {e.mean() * 100:.0f}% p95 {np.percentile(e, 95) * 100:.0f}% торса; "
          f"угол R6 ср {ang.mean():.1f}° p95 {np.percentile(ang, 95):.1f}°; кадров с ошибкой >30°: {(ang.max(1) > 30).sum()}"
          f"; размечено {len(res[ci]['anchors'])}/{n}")
