"""Связка: маски персонажей -> кропы -> нейросеть -> кандидаты суставов в координатах кадра.
Плюс дообучение модели на разметках пользователя для конкретного видео."""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

import colorseg as CS
import synth
from model import predict, predict_tta, peaks

CHANNELS = synth.CH   # у моделей v1 только первые 5 (без локтей и колен)
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
    a = x[0].astype(np.float32) / (255.0 if x.dtype == np.uint8 else 1.0)
    return cv2.resize(a, (FEAT, FEAT), interpolation=cv2.INTER_AREA).ravel()


def build_inputs(masks_per_frame, n_chars):
    """masks_per_frame[f][c] -> для каждого персонажа список (x(4,S,S) uint8, M, thick) или None.
    Каналы: свой силуэт, чужие, свой силуэт в кадре f-1 и f+1 — в рамке кадра f (сеть видит движение)."""
    import cv2
    n, S = len(masks_per_frame), synth.S
    out = [[None] * n for _ in range(n_chars)]
    unp = lambda f: [CS.unpack(m) for m in masks_per_frame[f]] if masks_per_frame[f] is not None else None
    cur = unp(0) if n else None
    nxt = unp(1) if n > 1 else None
    prev = None
    for f in range(n):
        ms = cur
        for c in range(n_chars):
            other = np.zeros_like(ms[c])
            for k in range(n_chars):
                if k != c:
                    other |= ms[k]
            x, M = CS.crop_input(ms[c], other)
            if x is None:
                continue
            ch = [np.round(x * 255).astype(np.uint8)]
            for nb in (prev, nxt):
                if nb is not None:
                    ch.append(cv2.warpAffine(nb[c].astype(np.uint8) * 255, M, (S, S), flags=cv2.INTER_AREA)[None])
                else:
                    ch.append(np.zeros((1, S, S), np.uint8))
            out[c][f] = (np.concatenate(ch), M, line_thickness(ms[c]))
        prev, cur = cur, nxt
        nxt = unp(f + 2) if f + 2 < n else None
    return out


def run_model(model, inputs, bs=64, tta=True):
    """inputs (по одному персонажу) -> frames_c для трекера. tta — ещё и зеркальный кадр, среднее."""
    idx = [f for f, v in enumerate(inputs) if v is not None]
    res = [None] * len(inputs)
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        X = np.stack([inputs[f][0] for f in chunk])
        H = predict_tta(model, X) if tta else predict(model, X)
        for f, hm in zip(chunk, H):
            x, M, th = inputs[f]
            pk = {}
            for ch, name in enumerate(CHANNELS[:hm.shape[0]]):
                pp = peaks(hm[ch], k=4 if name in ("hands", "feet", "elbows", "knees") else 3)
                pts = CS.crop_to_frame([p[:2] for p in pp], M)
                pk[name] = [(float(px), float(py), s) for (px, py), (_, _, s) in zip(pts, pp)]
            res[f] = {"ok": True, "peaks": pk, "scale": th, "feat": silhouette_feat(x), "hm": hm}
    return res


def labels_to_samples(inputs, labels):
    """Разметки пользователя -> обучающие примеры (x, kp (11,2) в пикселях кропа). Берём кадры, где видны все точки;
    у старых разметок (7 точек) локти/колени — середины рук/ног (грубо, но направления торса/кистей/стоп верны)."""
    J = ("head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot")
    out = []
    for f, L in labels.items():
        f = int(f)
        if inputs[f] is None:
            continue
        if any(L.get(j) is None for j in J):
            continue
        if any(L.get(a) is not None and L.get(b) is not None and np.hypot(*np.subtract(L[a], L[b])) < 3.0
               for a, b in (("l_hand", "r_hand"), ("l_elbow", "r_elbow"), ("l_foot", "r_foot"), ("l_knee", "r_knee"))):
            continue  # левая и правая в одной точке — ошибка разметки, не учим
        pts = [L[j] for j in J]
        for j, (a, b) in (("l_elbow", ("neck", "l_hand")), ("r_elbow", ("neck", "r_hand")),
                          ("l_knee", ("hip", "l_foot")), ("r_knee", ("hip", "r_foot"))):
            pts.append(L[j] if L.get(j) is not None else (np.add(L[a], L[b]) / 2).tolist())
        x, M, _ = inputs[f]
        kp = CS.frame_to_crop(pts, M)
        out.append((x, kp.astype(np.float32)))
    return out


def finetune(model, samples, steps=150, lr=4e-4, log=print):
    """Дообучение на разметках этого видео (вперемешку с синтетикой, чтобы не переобучиться)."""
    if not samples:
        return model
    import train as TR
    return TR.train(model, steps, lr=lr, bs=16, real=samples, real_frac=0.6, log_every=max(steps, 1),
                    workers=0, on_log=log)
