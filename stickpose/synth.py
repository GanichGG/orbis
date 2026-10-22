"""Генератор синтетических стикменов в стиле Stick Nodes для обучения модели.

Один пример = кроп 128x128 с четырьмя каналами (Orbis v2):
  0 — силуэт персонажа (с дырками там, где его перекрывают),
  1 — силуэты других персонажей/мусора поблизости,
  2, 3 — силуэт персонажа в предыдущем и следующем кадре (в той же рамке: видно движение),
и разметка: голова, шея (плечи), таз, 2 кисти, 2 стопы, 2 локтя, 2 колена (лево/право не различаем)
+ карта «скелет тела» (линии костей текущей позы без шлейфов и мусора) — учит сеть видеть тело целиком.

Расширяется добавлением стилей в Style и поз в sample_pose().
"""
import math
import os

import cv2
import numpy as np

S = 128          # размер входа
HM = 64          # размер карт вероятности
SS = 4           # суперсэмплинг при рисовании (сглаженные края)
ANG = ("torso", "chest", "head", "l_arm", "r_arm", "l_leg", "r_leg", "l_arm2", "r_arm2", "l_leg2", "r_leg2")
KP = ["head", "neck", "hip", "hand", "hand", "foot", "foot", "elbow", "elbow", "knee", "knee"]
CH = ["head", "neck", "hip", "hands", "feet", "elbows", "knees"]   # первые 5 — как у моделей v1
GROUPS = [[0], [1], [2], [3, 4], [5, 6], [7, 8], [9, 10]]
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
    P = lambda v: tuple(np.round(np.asarray(v) * SS).astype(int))
    w0 = T * style["thick"] * SS
    w = max(1, int(round(w0)))
    for a, b in segs:
        # стили Stick Nodes: торс бывает толще, нижние звенья рук/ног — тоньше (сужение)
        k = style.get("torso_w", 1.0) if b in ("mid", "neck") else (style.get("taper", 1.0) if a != "neck" and a != "hip" else 1.0)
        wi = max(1, int(round(w0 * k)))
        pa, pb = P(pts[a]), P(pts[b])
        cv2.line(canvas, pa, pb, val, wi, cv2.LINE_8)
        if style["round"]:
            cv2.circle(canvas, pa, wi // 2, val, -1)
            cv2.circle(canvas, pb, wi // 2, val, -1)
    for side in ("l", "r"):  # кулаки/ботинки — кружки на концах
        if style.get("fist") and side + "_arm_h" in pts:
            cv2.circle(canvas, P(pts[side + "_arm_h"]), max(1, int(w0 * style["fist"])), val, -1)
        if style.get("boot") and side + "_leg_f" in pts:
            cv2.circle(canvas, P(pts[side + "_leg_f"]), max(1, int(w0 * style["boot"])), val, -1)
    pr = style.get("prop")
    if pr and pr["hand"] + "_arm_h" in pts:  # оружие/палка в руке — того же цвета, попадает в маску персонажа
        h, e = pts[pr["hand"] + "_arm_h"], pts[pr["hand"] + "_arm_e"]
        d = h - e
        n = np.linalg.norm(d)
        d = _rot(d / n if n > 1e-6 else np.array([0.0, 1.0]), pr["ang"])
        cv2.line(canvas, P(h - d * T * pr["len"] * pr["back"]), P(h + d * T * pr["len"]), val,
                 max(1, int(w0 * pr["thick"])), cv2.LINE_8)
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
    hair = style.get("hair")
    if hair:  # причёска-«шипы» на голове (сторона от шеи)
        base = pts["head_dir"]
        for i in range(hair["n"]):
            a = base + hair["spread"] * (i / max(1, hair["n"] - 1) - 0.5) + hair["tilt"]
            tip = hc + _dir(a) * hr * (1 + hair["size"])
            l, r_ = hc + _dir(a - 0.35) * hr * 0.9, hc + _dir(a + 0.35) * hr * 0.9
            cv2.fillConvexPoly(canvas, np.round(np.array([l, tip, r_]) * SS).astype(np.int32), val)
    return hc


def sample_style(rng):
    if rng.random() < 0.12:  # «объёмный силуэт» (как бойцы Shadow Fight): толстые конечности, широкий торс
        return {"thick": rng.uniform(0.3, 0.48), "head": rng.uniform(0.26, 0.38), "head_gap": rng.uniform(0.7, 0.95),
                "round": True, "ring": False, "neck_line": rng.random() < 0.5, "taper": rng.uniform(0.6, 0.9),
                "torso_w": rng.uniform(1.2, 1.8), "fist": rng.uniform(0.7, 1.0) if rng.random() < 0.5 else 0.0,
                "boot": rng.uniform(0.7, 1.0) if rng.random() < 0.5 else 0.0,
                "prop": {"hand": "l" if rng.random() < 0.5 else "r", "len": rng.uniform(0.6, 1.8), "ang": rng.normal(0, 0.4),
                         "back": rng.uniform(0, 0.4), "thick": rng.uniform(0.2, 0.5)} if rng.random() < 0.3 else None,
                "hair": None}
    return {
        "thick": rng.uniform(0.07, 0.3),
        "head": rng.uniform(0.18, 0.38),
        "head_gap": rng.uniform(0.7, 1.15),
        "round": rng.random() < 0.85,
        "ring": rng.random() < 0.08,
        "neck_line": rng.random() < 0.3,
        "taper": rng.uniform(0.5, 0.95) if rng.random() < 0.35 else 1.0,
        "torso_w": rng.uniform(1.2, 2.0) if rng.random() < 0.25 else 1.0,
        "fist": rng.uniform(0.6, 1.1) if rng.random() < 0.25 else 0.0,
        "boot": rng.uniform(0.6, 1.1) if rng.random() < 0.2 else 0.0,
        "prop": {"hand": "l" if rng.random() < 0.5 else "r", "len": rng.uniform(0.5, 1.6), "ang": rng.normal(0, 0.5),
                 "back": rng.uniform(0, 0.4), "thick": rng.uniform(0.3, 0.8)} if rng.random() < 0.15 else None,
        "hair": {"n": int(rng.integers(2, 6)), "size": rng.uniform(0.3, 0.9), "spread": rng.uniform(0.8, 2.0),
                 "tilt": rng.normal(0, 0.6)} if rng.random() < 0.2 else None,
    }


def _figure(rng, T, hip):
    p = sample_pose(rng)
    st = sample_style(rng)
    pts, segs = forward(p, hip, T)
    return p, st, pts, segs


def _keypoints(pts, head):
    return np.array([head, pts["neck"], pts["hip"], pts["l_arm_h"], pts["r_arm_h"], pts["l_leg_f"], pts["r_leg_f"],
                     pts["l_arm_e"], pts["r_arm_e"], pts["l_leg_k"], pts["r_leg_k"]], float)


_W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MOCAP_PATH = os.path.join(_W, "mocap.npz")   # движения людей (CMU, tools/fetch_mocap.py)
SF2_PATH = os.path.join(_W, "sf2.npz")       # боевые анимации Shadow Fight 2 (tools/fetch_sf2.py), если есть
MOCAP_FRAC = 0.65   # доля примеров с позой/движением из записей (остальное — случайные позы, в т.ч. «дикие»)
SF2_FRAC = 0.4      # из примеров с записями — доля боевых анимаций SF2 (если файл есть)
_MOCAP = None


def _load_motion(path, min_len):
    z = np.load(path)
    ok = z["length"] >= min_len
    return {"pos": z["pos"].astype(np.float32), "root": z["root"].astype(np.float32), "start": z["start"][ok],
            "length": z["length"][ok], "J": {j: i for i, j in enumerate(z["joints"])}}


def mocap():
    """Источники движений: {"cmu": ..., "sf2": ...} (что есть) или None. Грузятся один раз на процесс."""
    global _MOCAP
    if _MOCAP is None:
        _MOCAP = {}
        if os.path.exists(MOCAP_PATH):
            _MOCAP["cmu"] = _load_motion(MOCAP_PATH, 16)
        if os.path.exists(SF2_PATH):
            _MOCAP["sf2"] = _load_motion(SF2_PATH, 8)
    return _MOCAP or None


class MocapFigure:
    """Человек из записи движения -> стикмен в 2D: случайный ракурс (чаще сбоку), пропорции Stick Nodes,
    укорочения в перспективе сохраняются. fig(o) — поза через o кадров (30 fps, можно дробно) -> (pts, segs)."""

    def __init__(self, rng, T, hip):
        src = mocap()
        key = "sf2" if "sf2" in src and ("cmu" not in src or rng.random() < SF2_FRAC) else "cmu"
        M = src[key]
        self.source = key
        self.M, self.T, self.hip = M, T, np.asarray(hip, float)
        i = int(rng.integers(len(M["start"])))
        self.step = float(rng.choice([0.5, 1, 1, 1, 2, 3], ))
        margin = int(math.ceil(self.step * 4)) + 1
        n = int(M["length"][i])
        self.s0 = int(M["start"][i])
        self.t = float(rng.integers(margin, max(margin + 1, n - margin)))
        self.n = n
        yaw = (math.pi / 2 * rng.choice([-1, 1]) + rng.normal(0, 0.45)) if rng.random() < 0.7 else rng.uniform(-math.pi, math.pi)
        pitch, roll = rng.normal(0, 0.14), rng.normal(0, 0.08)
        cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        self.R = Ry @ Rx @ Rz
        self.prop = {"chest_len": rng.uniform(0.35, 0.6), "upper": rng.uniform(0.45, 0.75), "fore": rng.uniform(0.45, 0.75),
                     "thigh": rng.uniform(0.5, 0.8), "shin": rng.uniform(0.5, 0.8)}
        J = M["J"]
        P0 = M["pos"][self.s0:self.s0 + n]
        self.torso3 = float(np.median(np.linalg.norm(P0[:, J["neck"]] - P0[:, J["hip"]], axis=1))) + 1e-6
        self.root0 = self._root(self.t)

    def _frame(self, arr, t):
        t = min(max(t, 0.0), self.n - 1.0)
        a = int(math.floor(t))
        b = min(a + 1, self.n - 1)
        w = t - a
        return arr[self.s0 + a] * (1 - w) + arr[self.s0 + b] * w

    def _root(self, t):
        return self._frame(self.M["root"], t)

    def _2d(self, v3):
        q = self.R @ v3
        return np.array([q[0], -q[2]])

    def __call__(self, o):
        t = self.t + o
        P3 = self._frame(self.M["pos"], t)
        J, T, pr = self.M["J"], self.T, self.prop

        def d2(a, b):
            v3 = P3[J[b]] - P3[J[a]]
            L = np.linalg.norm(v3)
            return self._2d(v3) / L if L > 1e-6 else np.zeros(2)
        hip = self.hip + self._2d(self._root(t) - self.root0) * (T / self.torso3)
        pts = {"hip": hip}
        pts["mid"] = hip + d2("hip", "chest") * T * (1 - pr["chest_len"])
        pts["neck"] = pts["mid"] + d2("chest", "neck") * T * pr["chest_len"]
        vh = d2("neck", "head")
        if np.linalg.norm(vh) < 0.3:  # голова смотрит на камеру — вдоль торса
            vh = pts["neck"] - hip
        pts["head_dir"] = math.atan2(-vh[0], vh[1])
        segs = [("hip", "mid"), ("mid", "neck")]
        for side in ("l", "r"):
            e = pts["neck"] + d2(side + "_sh", side + "_elbow") * T * pr["upper"]
            pts[side + "_arm_e"], pts[side + "_arm_h"] = e, e + d2(side + "_elbow", side + "_hand") * T * pr["fore"]
            kn = hip + d2(side + "_hipj", side + "_knee") * T * pr["thigh"]
            pts[side + "_leg_k"], pts[side + "_leg_f"] = kn, kn + d2(side + "_knee", side + "_foot") * T * pr["shin"]
            segs += [("neck", side + "_arm_e"), (side + "_arm_e", side + "_arm_h"),
                     ("hip", side + "_leg_k"), (side + "_leg_k", side + "_leg_f")]
        return pts, segs


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
    """-> x (CIN,S,S) float32 [0..1], kp (11,2) в пикселях кропа, bones (HM,HM) — карта костей.
    Поза и движение — из записей движений людей (mocap.npz, если есть; доля MOCAP_FRAC) или случайные."""
    big = S * 2
    T = rng.uniform(30, 50)
    cv_self = np.zeros((big * SS, big * SS), np.uint8)
    hip = np.array([big / 2, big / 2]) + rng.normal(0, 5, 2)
    use_mc = mocap() is not None and rng.random() < MOCAP_FRAC
    if use_mc:
        st = sample_style(rng)
        fig = MocapFigure(rng, T, hip)
        k = fig.step
        pts, segs = fig(0)
        nb_poses = [fig(-k), fig(k)]
        smear = [fig(t) for t in np.linspace(-k, 0, 6)[:-1]] if rng.random() < 0.2 else []
        ghosts = [fig(-k * (g + 1)) for g in range(int(rng.integers(1, 4)))] if rng.random() < 0.15 else []
    else:
        p, st, pts, segs = _figure(rng, T, hip)
        nb_poses = [forward(q, hq, T) for q, hq in _motion(rng, p, hip, T)]
        smear, ghosts = [], []
        if rng.random() < 0.2:  # смаз движения: копии позы на пути к ней
            p0 = {k: (v + rng.normal(0, 0.5) if k in ANG else v) for k, v in p.items()}
            shift = rng.normal(0, T * 0.4, 2)
            for t in np.linspace(0, 1, 6)[:-1]:
                pt = {k: (p0[k] * (1 - t) + p[k] * t) if isinstance(p[k], float) else p[k] for k in p}
                smear.append(forward(pt, hip - shift * (1 - t), T))
        if rng.random() < 0.15:  # шлейф-«призраки»
            back = rng.normal(0, T * 0.5, 2)
            for g in range(int(rng.integers(1, 4))):
                pg = {k: (v + rng.normal(0, 0.25) if k in ANG else v) for k, v in p.items()}
                ghosts.append(forward(pg, hip + back * (g + 1), T))
    # соседние кадры: та же фигура в движении; иногда кадра нет (край видео/нет маски) или там смена плана
    # (рисуются без суперсэмплинга: это грубая подсказка о движении, а так вдвое быстрее)
    nbs = []
    for qp, qs in nb_poses:
        cv_n = np.zeros((big, big), np.uint8)
        r = rng.random()
        if r < 0.07:
            pass
        elif r < 0.11:
            _, st2, pts2, segs2 = _figure(rng, T * rng.uniform(0.5, 1.6), hip + rng.normal(0, T, 2))
            render(cv_n, pts2, segs2, T, st2, ss=1)
        else:
            render(cv_n, qp, qs, T, st, ss=1)
        nbs.append(cv_n)
    for q, sg in smear:
        render(cv_self, q, sg, T, dict(st, thick=st["thick"] * rng.uniform(0.6, 1.0)))
    # шлейф-«призраки» (послеобразы того же цвета): тоньше и рваные
    for q, sg in ghosts:
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
        if use_mc and rng.random() < 0.6:
            st2 = sample_style(rng)
            pts2, segs2 = MocapFigure(rng, T2, hip2)(0)
        else:
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
    """kp (11,2) в пикселях S -> (len(CH),HM,HM)."""
    yy, xx = np.mgrid[0:HM, 0:HM].astype(np.float32)
    k = kp * (HM / S)
    out = np.zeros((len(GROUPS), HM, HM), np.float32)
    for c, g in enumerate(GROUPS):
        if np.isnan(k[g]).any():  # неуверенная точка (самообучение) — этот канал не учим
            out[c] = -1
            continue
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
            col = [(255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 255),
                   (255, 128, 0), (255, 128, 0), (128, 0, 255), (128, 0, 255)][j]
            cv2.circle(img, (int(px), int(py)), 3, col, -1)
        tiles.append(cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[r:r + 8]) for r in range(0, 32, 8)]
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "synth_grid.png", np.vstack(rows))
