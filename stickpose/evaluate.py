"""Оценка модели: точность на отложенной синтетике (PCK) + наложение предсказаний на реальное видео.
    python evaluate.py веса.pt [видео.mp4 маски.npz выход.jpg]"""
import sys

import cv2
import numpy as np

import synth
from model import load, predict, peaks


def synth_eval(model, n=400, seed=12345):
    rng = np.random.default_rng(seed)
    X, K, _ = zip(*[synth.make_sample(rng) for _ in range(n)])
    H = predict(model, np.stack(X))
    res = {}
    for ch, (name, ids) in enumerate(zip(synth.CH, synth.GROUPS)):
        if ch >= H.shape[1]:  # модель v1 — без локтей и колен
            break
        errs = []
        for i in range(n):
            pk = peaks(H[i, ch], k=len(ids) + 1)
            pts = np.array([p[:2] for p in pk])
            for j in ids:  # для пар: ближайший из предсказанных
                errs.append(np.min(np.linalg.norm(pts[:len(ids)] - K[i][j], axis=1)) if len(pts) else 99)
        e = np.array(errs) / synth.S
        res[name] = (float((e < 0.05).mean()), float((e < 0.1).mean()), float(np.median(e)))
    return res


if __name__ == "__main__":
    m = load(sys.argv[1])
    r = synth_eval(m)
    print("синтетика (доля точек ближе 5% / 10% кропа, медиана ошибки):")
    for k, (a, b, med) in r.items():
        print(f"  {k:6s} {a * 100:5.1f}% {b * 100:5.1f}%  медиана {med * 100:.1f}%")
