"""Постобработка траекторий перед экспортом:
  - фиксация стоп на земле (без проскальзывания);
  - выбор ключевых кадров (как аниматор): по каждой части R6 оставляем только кадры,
    без которых кривая угла отклонилась бы больше допуска (Рамер—Дуглас—Пекер по времени).
"""
import numpy as np

J = {"head": 0, "neck": 1, "hip": 2, "l_hand": 3, "r_hand": 4, "l_foot": 5, "r_foot": 6,
     "l_elbow": 7, "r_elbow": 8, "l_knee": 9, "r_knee": 10}
PARTS = {  # часть R6: (откуда, куда) — направление части. Правило аниматора: рука R6 = предплечье, нога = голень
    "Torso": ("hip", "neck"), "Head": ("neck", "head"),
    "Left Arm": ("l_elbow", "l_hand"), "Right Arm": ("r_elbow", "r_hand"),
    "Left Leg": ("l_knee", "l_foot"), "Right Leg": ("r_knee", "r_foot"),
}


def facing(P, T):
    """Куда смотрит персонаж по кадрам: +1 вправо, -1 влево, 0 не понять. Колени гнутся вперёд:
    знак векторного произведения (таз->стопа) x (таз->колено) в системе тела (работает и вверх ногами)."""
    votes = np.zeros(len(P))
    for k, f in (("l_knee", "l_foot"), ("r_knee", "r_foot")):
        d = P[:, J[f]] - P[:, J["hip"]]
        q = P[:, J[k]] - P[:, J["hip"]]
        cr = (d[:, 0] * q[:, 1] - d[:, 1] * q[:, 0]) / (np.linalg.norm(d, axis=1) * T + 1e-6)
        votes += np.where(np.abs(cr) > 0.08, -np.sign(cr), 0)   # cr < 0 -> колено справа от линии -> вправо
    return np.sign(votes).astype(int)


def segments(n, cuts):
    b = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
    return list(zip(b, b[1:]))


def plant_feet(P, T, cuts, speed=0.04, height=0.12, min_len=3):
    """P (n,7,2) пиксели, T (n,) длина торса. Стопа «стоит», если почти не движется и находится
    у нижней границы своего участка. Во время контакта стопа держится в одной точке."""
    P = P.copy()
    n = len(P)
    contacts = {}
    for s0, s1 in segments(n, cuts):
        if s1 - s0 < min_len:
            continue
        ground = np.percentile(P[s0:s1, [5, 6], 1], 95)  # y земли (ниже всех стоп), экранный y вниз
        for j in (5, 6):
            y = P[s0:s1, j, 1]
            v = np.r_[np.inf, np.linalg.norm(np.diff(P[s0:s1, j], axis=0), axis=1)] / T[s0:s1]
            on = (v < speed) & ((ground - y) / T[s0:s1] < height)
            f = 0
            while f < len(on):
                if not on[f]:
                    f += 1
                    continue
                e = f
                while e + 1 < len(on) and on[e + 1]:
                    e += 1
                if e - f + 1 >= min_len:
                    a, b = s0 + f, s0 + e + 1
                    P[a:b, j] = np.median(P[a:b, j], axis=0)
                    contacts.setdefault(j, []).append((a, b - 1))
                f = e + 1
    return P, contacts


def part_angles(P):
    """Углы частей на экране (радианы, развёрнутые по времени)."""
    out = {}
    for name, (a, b) in PARTS.items():
        v = P[:, J[b]] - P[:, J[a]]
        out[name] = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    return out


def _rdp(t, y, tol):
    """Индексы точек, которые нужно оставить (Рамер—Дуглас—Пекер на кривой y(t))."""
    keep = {0, len(t) - 1}
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        frac = (t[a + 1:b] - t[a]) / (t[b] - t[a])
        if y.ndim > 1:
            lin = y[a] + (y[b] - y[a])[None] * frac[:, None]
            d = np.linalg.norm(y[a + 1:b] - lin, axis=1)
        else:
            lin = y[a] + (y[b] - y[a]) * frac
            d = np.abs(y[a + 1:b] - lin)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            keep.add(m)
            stack += [(a, m), (m, b)]
    return sorted(keep)


def keyframes(P, T, cuts, labeled=(), tol_deg=4.0, tol_root=0.06):
    """-> {часть: [кадры]} + "Root": [кадры]. Кадры на смене плана и размеченные — всегда ключи."""
    n = len(P)
    ang = part_angles(P)
    must = set(labeled) | set(cuts) | {max(0, c - 1) for c in cuts}
    keys = {}
    for name, y in ang.items():
        ks = set()
        for s0, s1 in segments(n, cuts):
            idx = _rdp(np.arange(s0, s1, dtype=float), y[s0:s1], np.radians(tol_deg))
            ks |= {s0 + i for i in idx}
        keys[name] = sorted(ks | must)
    # корень: путь таза в единицах торса
    root = P[:, J["hip"]] / T[:, None]
    ks = set()
    for s0, s1 in segments(n, cuts):
        idx = _rdp(np.arange(s0, s1, dtype=float), root[s0:s1], tol_root)
        ks |= {s0 + i for i in idx}
    keys["Root"] = sorted(ks | must)
    return keys
