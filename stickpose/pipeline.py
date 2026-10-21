"""Связка: маски персонажей -> кропы -> нейросеть -> кандидаты суставов в координатах кадра.
Плюс дообучение модели на разметках пользователя для конкретного видео."""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

import colorseg as CS
import synth
from model import predict, peaks

CHANNELS = ["head", "neck", "hip", "hands", "feet"]
FEAT = 24


def line_thickness(mask):
    """Толщина линий персонажа (медиана расстояния до края на скелете) — масштаб, не зависящий от позы."""
    if mask.sum() < 20:
        return np.nan
    dt = ndimage.distance_transform_edt(mask)
    sk = skeletonize(mask)
    v = dt[sk]
    return float(np.median(v) * 2) if len(v) else np.nan


def silhouette_feat(x):
    import cv2
    return cv2.resize(x[0], (FEAT, FEAT), interpolation=cv2.INTER_AREA).ravel()


def build_inputs(masks_per_frame, n_chars):
    """masks_per_frame[f][c] -> для каждого персонажа список (x(2,S,S), M, thick) или None."""
    out = [[None] * len(masks_per_frame) for _ in range(n_chars)]
    for f, ms in enumerate(masks_per_frame):
        for c in range(n_chars):
            other = np.zeros_like(ms[c])
            for k in range(n_chars):
                if k != c:
                    other |= ms[k]
            x, M = CS.crop_input(ms[c], other)
            if x is None:
                continue
            out[c][f] = (x, M, line_thickness(ms[c]))
    return out


def run_model(model, inputs, bs=64):
    """inputs (по одному персонажу) -> frames_c для трекера."""
    idx = [f for f, v in enumerate(inputs) if v is not None]
    res = [None] * len(inputs)
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        X = np.stack([inputs[f][0] for f in chunk])
        H = predict(model, X)
        for f, hm in zip(chunk, H):
            x, M, th = inputs[f]
            pk = {}
            for ch, name in enumerate(CHANNELS):
                pp = peaks(hm[ch], k=4 if name in ("hands", "feet") else 3)
                pts = CS.crop_to_frame([p[:2] for p in pp], M)
                pk[name] = [(float(px), float(py), s) for (px, py), (_, _, s) in zip(pts, pp)]
            res[f] = {"ok": True, "peaks": pk, "scale": th, "feat": silhouette_feat(x), "hm": hm}
    return res


def labels_to_samples(inputs, labels):
    """Разметки пользователя -> обучающие примеры (x, kp в пикселях кропа). Невидимые точки пропускаются
    (заменяются предсказанием не нужно: такие кадры берём, только если видны все 7)."""
    out = []
    for f, L in labels.items():
        f = int(f)
        if inputs[f] is None:
            continue
        if any(L.get(j) is None for j in ("head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot")):
            continue
        x, M, _ = inputs[f]
        kp = CS.frame_to_crop([L[j] for j in ("head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot")], M)
        out.append((x.astype(np.float32), kp.astype(np.float32)))
    return out


def finetune(model, samples, steps=150, lr=4e-4, log=print):
    """Дообучение на разметках этого видео (вперемешку с синтетикой, чтобы не переобучиться)."""
    if not samples:
        return model
    import train as TR
    return TR.train(model, steps, lr=lr, bs=16, real=samples, real_frac=0.6, log_every=max(steps, 1),
                    workers=0, on_log=log)
