"""Генератор синтетических стикменов в стиле Stick Nodes для обучения модели.

Один пример = кроп 128x128 с двумя каналами:
  0 — силуэт персонажа (с дырками там, где его перекрывают),
  1 — силуэты других персонажей/мусора поблизости,
и разметка: голова, шея (плечи), таз, 2 кисти, 2 стопы (лево/право не различаем).

Расширяется добавлением стилей в Style и поз в sample_pose().
"""
import math

import cv2
import numpy as np

S = 128          # размер входа
HM = 64          # размер карт вероятности
SS = 4           # суперсэмплинг при рисовании (сглаженные края)
ANG = ("torso", "chest", "head", "l_arm", "r_arm", "l_leg", "r_leg", "l_arm2", "r_arm2", "l_leg2", "r_leg2")
KP = ["head", "neck", "hip", "hand", "hand", "foot", "foot"]
CH = ["head", "neck", "hip", "hands", "feet"]


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _dir(a):
    """0 — вниз, по часовой на экране (y вниз)."""
    return np.array([-math.sin(a), math.cos(a)])


def sample_pose(rng):
    """Случайная поза: углы в радианах (0 = вниз), длины относительно торса."""
    p = {}
    r = rng.random()
    if r < 0.6:      # стоит/идёт/бьёт — около вертикали
        p["torso"] = math.pi + rng.normal(0, 0.35)
    elif r < 0.8:    # наклон, падение
        p["torso"] = math.pi + rng.normal(0, 1.0)
    else:            # что угодно (сальто, лежит)
        p["torso"] = rng.uniform(-math.pi, math.pi)
    p["chest"] = rng.normal(0, 0.25)                 # изгиб второго сегмента торса
    p["head"] = rng.normal(0, 0.3)
    side = 1 if rng.random() < 0.5 else -1           # куда «смотрит» (для сгиба коленей)
    for k in ("l_arm", "r_arm"):
        mode = rng.random()
        if mode < 0.3:   # вдоль тела
            p[k] = rng.normal(0, 0.4)
        else:
            p[k] = rng.uniform(-math.pi, math.pi)
        p[k + "2"] = rng.uniform(-2.6, 2.6) * (rng.random() < 0.85)
    for k in ("l_leg", "r_leg"):
        p[k] = rng.normal(0, 0.6) if rng.random() < 0.75 else rng.uniform(-2.2, 2.2)
        p[k + "2"] = side * abs(rng.normal(0, 0.9)) if rng.random() < 0.85 else rng.uniform(-2.5, 2.5)
    # пропорции
    p["chest_len"] = rng.uniform(0.35, 0.6)
    p["upper"] = rng.uniform(0.45, 0.75)
    p["fore"] = rng.uniform(0.45, 0.75)
    p["thigh"] = rng.uniform(0.5, 0.8)
    p["shin"] = rng.uniform(0.5, 0.8)
    # перспектива: некоторые сегменты укорочены
    for k in ("upper", "fore", "thigh", "shin"):
        if rng.random() < 0.15:
            p[k] *= rng.uniform(0.25, 0.8)
    return p


def forward(p, hip, T):
    """Поза -> точки. T — длина торса в пикселях."""
    pts = {}
    pts["hip"] = hip
    ta = p["torso"]
    mid = hip + _dir(ta) * T * (1 - p["chest_len"])
    ca = ta + p["chest"]
    neck = mid + _dir(ca) * T * p["chest_len"]
    pts["mid"], pts["neck"] = mid, neck
    pts["head_dir"] = ca + p["head"]
    segs = [("hip", "mid"), ("mid", "neck")]
    for k in ("l_arm", "r_arm"):
        a1 = ca + math.pi + p[k]
        e = neck + _dir(a1) * T * p["upper"]
        h = e + _dir(a1 + p[k + "2"]) * T * p["fore"]
        pts[k + "_e"], pts[k + "_h"] = e, h
        segs += [("neck", k + "_e"), (k + "_e", k + "_h")]
    for k in ("l_leg", "r_leg"):
        a1 = ta + math.pi + p[k]
        kn = hip + _dir(a1) * T * p["thigh"]
        f = kn + _dir(a1 + p[k + "2"]) * T * p["shin"]
        pts[k + "_k"], pts[k + "_f"] = kn, f
        segs += [("hip", k + "_k"), (k + "_k", k + "_f")]
    return pts, segs


def render(canvas, pts, segs, T, style, val=255):
    """Рисует стикмена в canvas (uint8, суперсэмплинг SS). Возвращает центр головы."""
    w = max(1, int(round(T * style["thick"] * SS)))
    for a, b in segs:
        pa = tuple(np.round(pts[a] * SS).astype(int))
        pb = tuple(np.round(pts[b] * SS).astype(int))
        cv2.line(canvas, pa, pb, val, w, cv2.LINE_8)
        if style["round"]:
            cv2.circle(canvas, pa, w // 2, val, -1)
            cv2.circle(canvas, pb, w // 2, val, -1)
    hr = T * style["head"]
    hc = pts["neck"] + _dir(pts["head_dir"]) * (hr * style["head_gap"])
    c = tuple(np.round(hc * SS).astype(int))
    r = int(round(hr * SS))
    if style["ring"]:
        cv2.circle(canvas, c, r, val, max(1, w // 2))
    else:
        cv2.circle(canvas, c, r, val, -1)
    if style["neck_line"]:
        cv2.line(canvas, tuple(np.round(pts["neck"] * SS).astype(int)), c, val, w)
    return hc


def sample_style(rng):
    return {
        "thick": rng.uniform(0.07, 0.3),
        "head": rng.uniform(0.18, 0.38),
        "head_gap": rng.uniform(0.7, 1.15),
        "round": rng.random() < 0.85,
        "ring": rng.random() < 0.08,
        "neck_line": rng.random() < 0.3,
    }


def _figure(rng, T, hip):
    p = sample_pose(rng)
    st = sample_style(rng)
    pts, segs = forward(p, hip, T)
    return p, st, pts, segs


def _keypoints(pts, head):
    return np.array([head, pts["neck"], pts["hip"], pts["l_arm_h"], pts["r_arm_h"], pts["l_leg_f"], pts["r_leg_f"]], float)


def make_sample(rng):
    """-> x (2,S,S) float32 [0..1], kp (7,2) в пикселях кропа."""
    big = S * 2
    T = rng.uniform(30, 50)
    cv_self = np.zeros((big * SS, big * SS), np.uint8)
    hip = np.array([big / 2, big / 2]) + rng.normal(0, 5, 2)
    p, st, pts, segs = _figure(rng, T, hip)

    # смаз движения: копии позы на пути к ней
    if rng.random() < 0.2:
        p0 = dict(p)
        for k in p0:
            if k in ("torso", "chest", "head", "l_arm", "r_arm", "l_leg", "r_leg", "l_arm2", "r_arm2", "l_leg2", "r_leg2"):
                p0[k] = p[k] + rng.normal(0, 0.5)
        shift = rng.normal(0, T * 0.4, 2)
        for t in np.linspace(0, 1, 6)[:-1]:
            pt = {k: (p0[k] * (1 - t) + p[k] * t) if isinstance(p[k], float) else p[k] for k in p}
            q, sg = forward(pt, hip - shift * (1 - t), T)
            st2 = dict(st, thick=st["thick"] * rng.uniform(0.6, 1.0))
            render(cv_self, q, sg, T, st2)
    # шлейф-«призраки» (послеобразы того же цвета): тоньше и рваные, сдвинуты назад по движению
    if rng.random() < 0.15:
        back = rng.normal(0, T * 0.5, 2)
        for g in range(int(rng.integers(1, 4))):
            pg = {k: (v + rng.normal(0, 0.25) if k in ANG else v) for k, v in p.items()}
            q, sg = forward(pg, hip + back * (g + 1), T)
            ghost = np.zeros_like(cv_self)
            render(ghost, q, sg, T, dict(st, thick=st["thick"] * rng.uniform(0.4, 0.9)))
            holes = (rng.random((ghost.shape[0] // 32 + 1, ghost.shape[1] // 32 + 1)) < 0.35).astype(np.uint8)
            holes = cv2.resize(holes, (ghost.shape[1], ghost.shape[0]), interpolation=cv2.INTER_NEAREST)
            ghost[holes > 0] = 0
            cv_self |= ghost
    head = render(cv_self, pts, segs, T, st)
    kp = _keypoints(pts, head)

    # другие персонажи рядом / поверх
    cv_other = np.zeros_like(cv_self)
    front = np.zeros_like(cv_self)
    for _ in range(rng.choice([0, 1, 1, 2], p=[0.35, 0.35, 0.2, 0.1])):
        T2 = T * rng.uniform(0.6, 1.5)
        hip2 = hip + rng.normal(0, T * 1.0, 2)
        _, st2, pts2, segs2 = _figure(rng, T2, hip2)
        tmp = np.zeros_like(cv_self)
        render(tmp, pts2, segs2, T2, st2)
        cv_other |= tmp
        if rng.random() < 0.5:  # этот персонаж перекрывает нашего
            front |= tmp
    self_m = cv_self.copy()
    self_m[front > 0] = 0
    # частицы / мусор
    for _ in range(rng.poisson(1.5)):
        c = tuple(rng.integers(0, big * SS, 2))
        rr = int(rng.uniform(1, 4) * SS)
        cv2.circle(cv_other if rng.random() < 0.7 else self_m, c, rr, 255, -1)

    def down(m):
        return cv2.resize(m, (big, big), interpolation=cv2.INTER_AREA)

    self_m, other_m = down(self_m), down(cv_other)
    full_self = down(cv_self)

    # кроп как в реальном пайплайне: по рамке видимой маски персонажа + случайный запас/сдвиг
    ys, xs = np.nonzero(self_m > 80)
    if len(xs) < 10:
        ys, xs = np.nonzero(full_self > 80)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(x1 - x0, y1 - y0) * rng.uniform(1.1, 1.45) + 4
    cx = (x0 + x1) / 2 + rng.normal(0, side * 0.04)
    cy = (y0 + y1) / 2 + rng.normal(0, side * 0.04)
    sc = S / side
    M = np.array([[sc, 0, S / 2 - cx * sc], [0, sc, S / 2 - cy * sc]], np.float32)
    xa = cv2.warpAffine(self_m, M, (S, S), flags=cv2.INTER_AREA)
    xb = cv2.warpAffine(other_m, M, (S, S), flags=cv2.INTER_AREA)
    kp = kp * sc + M[:, 2]

    # шум силуэта: размытие/порог, эрозия/дилатация, сжатие
    if rng.random() < 0.5:
        k = int(rng.integers(1, 3)) * 2 + 1
        xa = cv2.GaussianBlur(xa, (k, k), 0)
    if rng.random() < 0.3:
        op = cv2.erode if rng.random() < 0.5 else cv2.dilate
        xa = op(xa, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:
        xa = (xa > rng.uniform(60, 180)).astype(np.uint8) * 255
    x = np.stack([xa, xb]).astype(np.float32) / 255.0
    if rng.random() < 0.5:  # зеркало (лево/право и так не различаем)
        x = x[:, :, ::-1].copy()
        kp[:, 0] = S - 1 - kp[:, 0]
    return x, kp.astype(np.float32)


def heatmaps(kp, sigma=1.6):
    """kp (7,2) в пикселях S -> (5,HM,HM)."""
    yy, xx = np.mgrid[0:HM, 0:HM].astype(np.float32)
    k = kp * (HM / S)
    out = np.zeros((5, HM, HM), np.float32)
    groups = [[0], [1], [2], [3, 4], [5, 6]]
    for c, g in enumerate(groups):
        for i in g:
            out[c] = np.maximum(out[c], np.exp(-((xx - k[i, 0]) ** 2 + (yy - k[i, 1]) ** 2) / (2 * sigma ** 2)))
    return out


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    tiles = []
    for i in range(32):
        x, kp = make_sample(rng)
        img = np.zeros((S, S, 3), np.uint8)
        img[..., 2] = (x[0] * 255).astype(np.uint8)
        img[..., 0] = (x[1] * 160).astype(np.uint8)
        for j, (px, py) in enumerate(kp):
            col = [(255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 255)][j]
            cv2.circle(img, (int(px), int(py)), 3, col, -1)
        tiles.append(cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[r:r + 8]) for r in range(0, 32, 8)]
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "synth_grid.png", np.vstack(rows))
