"""Наложение предсказаний сети (без разметки) на реальное видео: контактный лист + mp4.
    python overlay.py веса.pt видео.mp4 выход_префикс "x,y;x,y" [start end]
Точки «x,y» — клики пипеткой по персонажам на первом кадре (в координатах кадра 960px)."""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import colorseg as CS
import pipeline as PL
from model import load

COL = {"head": (255, 255, 255), "neck": (0, 255, 0), "hip": (0, 255, 255), "hands": (255, 200, 0), "feet": (255, 0, 255)}


def fit(f, ms=960):
    s = ms / max(f.shape[:2])
    return f if s >= 1 else cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)


def main():
    w, vid, out, pts = sys.argv[1:5]
    s0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    s1 = int(sys.argv[6]) if len(sys.argv) > 6 else 10 ** 9
    pts = [tuple(map(float, p.split(","))) for p in pts.split(";")]
    cap = cv2.VideoCapture(vid)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or i >= s1:
            break
        if i >= s0:
            frames.append(fit(f))
        i += 1
    cols = [[CS.sample_color(frames[0], x, y, r=1).tolist()] for x, y in pts]
    seg = CS.ColorSegmenter(cols)
    masks = [seg.step(frames[0], init_points=pts)] + [seg.step(f) for f in frames[1:]]
    inputs = PL.build_inputs(masks, len(pts))
    model = load(w)
    fcs = [PL.run_model(model, inputs[c]) for c in range(len(pts))]
    h, wd = frames[0].shape[:2]
    vw = cv2.VideoWriter(out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (wd, h))
    shots = []
    pick = set(np.linspace(0, len(frames) - 1, 12).astype(int).tolist())
    for f, fr in enumerate(frames):
        img = fr.copy()
        for c in range(len(pts)):
            fc = fcs[c][f]
            if not fc:
                continue
            for ch, pk in fc["peaks"].items():
                k = 2 if ch in ("hands", "feet") else 1
                for (x, y, s) in pk[:k]:
                    if s > 0.1:
                        cv2.circle(img, (int(x), int(y)), 4, COL[ch], -1)
                        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1)
        vw.write(img)
        if f in pick:
            ys, xs = np.nonzero(np.any([m for m in masks[f]], 0))
            if len(xs):
                x0, x1, y0, y1 = max(0, xs.min() - 30), xs.max() + 30, max(0, ys.min() - 30), ys.max() + 30
                t = img[y0:y1, x0:x1]
                t = cv2.resize(t, (int(220 * t.shape[1] / t.shape[0]), 220))
                cv2.putText(t, str(f + s0), (4, 18), 0, 0.6, (0, 0, 255), 2)
                shots.append(t)
    vw.release()
    rows, row, wsum = [], [], 0
    for t in shots:
        if wsum + t.shape[1] > 1800 and row:
            rows.append(np.hstack(row + [np.zeros((220, 1800 - wsum, 3), np.uint8)]))
            row, wsum = [], 0
        row.append(t[:, :1800])
        wsum += min(t.shape[1], 1800)
    if row:
        rows.append(np.hstack(row + [np.zeros((220, max(0, 1800 - wsum), 3), np.uint8)]))
    cv2.imwrite(out + ".jpg", np.vstack(rows))


if __name__ == "__main__":
    main()
