"""Генератор синтетических стикменов в стиле Stick Nodes для обучения модели.

Один пример = кроп 128x128 с четырьмя каналами (Orbis v2):
  0 — силуэт персонажа (с дырками там, где его перекрывают),
  1 — силуэты других персонажей/мусора поблизости,
  2, 3 — силуэт персонажа в предыдущем и следующем кадре (в той же рамке: видно движение),
и разметка: голова, шея (плечи), таз, 2 кисти, 2 стопы (лево/право не различаем)
+ карта «скелет тела» (линии костей текущей позы без шлейфов и мусора) — учит сеть видеть тело целиком.

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
CIN = 4          # каналы входа (см. выше); старые модели v1 брали только первые 2


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


def render(canvas, pts, segs, T, style, val=255, ss=SS):
    """Рисует стикмена в canvas (uint8, суперсэмплинг ss). Возвращает центр головы."""
    SS = ss
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


def _motion(rng, p, hip, T):
    """Позы и положения таза в соседних кадрах: (p_prev, hip_prev), (p_next, hip_next).
    Скорость — от почти неподвижной до резкого удара за кадр (как в Stick Nodes на 24-30 fps)."""
    sa = rng.choice([0.02, 0.08, 0.2, 0.45], p=[0.2, 0.35, 0.3, 0.15])
    sh = T * rng.choice([0.02, 0.1, 0.3, 0.7], p=[0.3, 0.35, 0.25, 0.1])
    v = {k: rng.normal(0, sa) for k in ANG}
    hv = rng.normal(0, sh, 2)
    out = []
    for sgn in (-1, 1):
        q = dict(p)
        for k in ANG:
            q[k] = p[k] + sgn * v[k] + rng.normal(0, sa * 0.3)
        out.append((q, hip + sgn * hv + rng.normal(0, sh * 0.2, 2)))
    return out


def make_sample(rng):
    """-> x (CIN,S,S) float32 [0..1], kp (7,2) в пикселях кропа, bones (HM,HM) — карта костей."""
    big = S * 2
    T = rng.uniform(30, 50)
    cv_self = np.zeros((big * SS, big * SS), np.uint8)
    hip = np.array([big / 2, big / 2]) + rng.normal(0, 5, 2)
    p, st, pts, segs = _figure(rng, T, hip)
    # соседние кадры: та же фигура в движении; иногда кадра нет (край видео/нет маски) или там смена плана
    # (рисуются без суперсэмплинга: это грубая подсказка о движении, а так вдвое быстрее)
    nbs = []
    for q, hq in _motion(rng, p, hip, T):
        cv_n = np.zeros((big, big), np.uint8)
        r = rng.random()
        if r < 0.07:
            pass
        elif r < 0.11:
            _, st2, pts2, segs2 = _figure(rng, T * rng.uniform(0.5, 1.6), hip + rng.normal(0, T, 2))
            render(cv_n, pts2, segs2, T, st2, ss=1)
        else:
            qp, qs = forward(q, hq, T)
            render(cv_n, qp, qs, T, st, ss=1)
        nbs.append(cv_n)

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
    if front.any():  # перекрывающий тоже двигается
        fr_small = cv2.resize(front, (big, big), interpolation=cv2.INTER_AREA) > 127
        for cv_n in nbs:
            d = np.round(rng.normal(0, T * 0.15, 2)).astype(int)
            cv_n[np.roll(fr_small, tuple(d), axis=(0, 1))] = 0
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
    warp = lambda m: cv2.warpAffine(m, M, (S, S), flags=cv2.INTER_AREA)
    xa, xb = warp(self_m), warp(other_m)
    xn = [warp(m) for m in nbs]
    kp = kp * sc + M[:, 2]
    bones = bone_map({k: v * sc + M[:, 2] for k, v in pts.items() if k != "head_dir"}, segs, kp[0])

    # шум силуэта (одинаковый для всех кадров — это одна и та же маска пипетки): размытие/порог, эрозия/дилатация
    k = int(rng.integers(1, 3)) * 2 + 1 if rng.random() < 0.5 else 0
    op = (cv2.erode if rng.random() < 0.5 else cv2.dilate) if rng.random() < 0.3 else None
    thr = rng.uniform(60, 180) if rng.random() < 0.5 else None

    def noise(m):
        if k:
            m = cv2.GaussianBlur(m, (k, k), 0)
        if op is not None:
            m = op(m, np.ones((2, 2), np.uint8))
        if thr is not None:
            m = (m > thr).astype(np.uint8) * 255
        return m
    x = np.stack([noise(xa), xb] + [noise(m) for m in xn]).astype(np.float32) / 255.0
    if rng.random() < 0.5:  # зеркало (лево/право и так не различаем)
        x = x[:, :, ::-1].copy()
        bones = bones[:, ::-1].copy()
        kp[:, 0] = S - 1 - kp[:, 0]
    return x, kp.astype(np.float32), bones


def bone_map(pts, segs, head):
    """Линии костей текущей позы (без шлейфов, чужих и мусора) в разрешении карт HM."""
    m = np.zeros((HM * 4, HM * 4), np.uint8)
    k = HM / S * 4

    def p(v):
        return tuple(np.round(np.asarray(v) * k).astype(int))
    for a, b in list(segs) + [("neck", None)]:
        cv2.line(m, p(pts[a]), p(head if b is None else pts[b]), 255, 5, cv2.LINE_AA)
    return cv2.resize(m, (HM, HM), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


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
        x, kp, bones = make_sample(rng)
        img = np.zeros((S, S, 3), np.uint8)
        img[..., 2] = (x[0] * 255).astype(np.uint8)
        img[..., 0] = (x[1] * 160).astype(np.uint8)
        img[..., 1] = (np.maximum(x[2], x[3]) * 90).astype(np.uint8)
        img[..., 1] = np.maximum(img[..., 1], cv2.resize(bones, (S, S)) * 255).astype(np.uint8)
        for j, (px, py) in enumerate(kp):
            col = [(255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 255)][j]
            cv2.circle(img, (int(px), int(py)), 3, col, -1)
        tiles.append(cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[r:r + 8]) for r in range(0, 32, 8)]
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "synth_grid.png", np.vstack(rows))
