"""Сравнение «видео | R6»: насколько анимация повторяет оригинал.

Запуск:   python compare.py видео.mp4 [видео.r6.json] [--start N] [--all]
Результат: <видео>_compare.mp4 — слева видео с найденными точками, справа то же видео с наложенным R6
(вид сбоку, как его поставит импорт: торс по линии таз→шея, рука по предплечью локоть→кисть,
нога по голени колено→стопа; у старых файлов без локтей/колен — шея→кисть и таз→стопа).
Между ключевыми кадрами углы и корень интерполируются линейно, как в Blender (INTERP = "LINEAR"),
так что видно и потери от прореживания ключей. --all — показать все кадры без прореживания.
"""
import argparse
import json
import os

import cv2
import numpy as np

J = {"head": 0, "neck": 1, "hip": 2, "l_hand": 3, "r_hand": 4, "l_foot": 5, "r_foot": 6,
     "l_elbow": 7, "r_elbow": 8, "l_knee": 9, "r_knee": 10}
BONES = [("hip", "neck"), ("neck", "head"), ("neck", "l_elbow"), ("l_elbow", "l_hand"), ("neck", "r_elbow"),
         ("r_elbow", "r_hand"), ("hip", "l_knee"), ("l_knee", "l_foot"), ("hip", "r_knee"), ("r_knee", "r_foot")]
# R6 в шпильках (studs): торс 2 в высоту, руки/ноги 2, голова ~1.2; вид сбоку — ширина = толщина 1
PARTS = {"torso": ("hip", "neck"), "head": ("neck", "head"), "l_arm": ("l_elbow", "l_hand"), "r_arm": ("r_elbow", "r_hand"),
         "l_leg": ("l_knee", "l_foot"), "r_leg": ("r_knee", "r_foot")}
COLORS = [(60, 60, 230), (230, 120, 40), (40, 180, 60), (200, 60, 200)]


def angles(P):
    """Углы частей (развёрнутые по времени) + положение таза. P: (n, 7, 2)."""
    A = {}
    for name, (a, b) in PARTS.items():
        v = P[:, J[b]] - P[:, J[a]]
        A[name] = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    return A


def interp_keys(A, hip, T, keys, n):
    """Линейная интерполяция углов, таза и масштаба между ключевыми кадрами (как в Blender)."""
    k = np.array(sorted(set(keys) | {0, n - 1}))
    t = np.arange(n)
    A2 = {name: np.interp(t, k, y[k]) for name, y in A.items()}
    hip2 = np.stack([np.interp(t, k, hip[k, i]) for i in range(2)], 1)
    return A2, hip2, np.interp(t, k, T[k])


def box(img, origin, ang, length, width, color, alpha):
    """Прямоугольник части R6: от origin вдоль угла ang (0 — вниз по экрану), длина/ширина в пикселях."""
    d = np.array([np.sin(ang), np.cos(ang)])
    nrm = np.array([-d[1], d[0]])
    o = np.asarray(origin, float)
    pts = np.array([o - nrm * width / 2, o + d * length - nrm * width / 2,
                    o + d * length + nrm * width / 2, o + nrm * width / 2])
    over = img.copy()
    cv2.fillConvexPoly(over, np.round(pts).astype(np.int32), color, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (20, 20, 20), 1, cv2.LINE_AA)


def draw_r6(img, a, hip, T, facing, color):
    """R6 сбоку: шпилька = T/2 (торс R6 высотой 2 = таз..шея)."""
    s = T / 2.0
    up = a["torso"]                               # угол таз->шея (π — вверх)
    d = np.array([np.sin(up), np.cos(up)])
    neck = hip + d * 2 * s
    shoulder = hip + d * 1.5 * s                  # плечевой сустав R6 на 0.5 ниже верха торса
    near = "r" if facing == "r" else "l"          # ближняя к камере сторона рисуется поверх
    far = "l" if near == "r" else "r"
    dark = tuple(int(c * 0.55) for c in color)
    box(img, shoulder, a[f"{far}_arm"], 2 * s, s, dark, 0.75)
    box(img, hip, a[f"{far}_leg"], 2 * s, s, dark, 0.75)
    box(img, hip, up, 2 * s, s, color, 0.6)
    box(img, neck, a["head"], 1.2 * s, 1.0 * s, color, 0.6)
    box(img, hip, a[f"{near}_leg"], 2 * s, s, color, 0.8)
    box(img, shoulder, a[f"{near}_arm"], 2 * s, s, color, 0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("json", nargs="?")
    ap.add_argument("--start", type=int, default=None, help="с какого кадра видео шла разметка (если не записано в файле)")
    ap.add_argument("--all", action="store_true", help="все кадры, без прореживания ключей")
    a = ap.parse_args()
    base = os.path.splitext(a.video)[0]
    data = json.load(open(a.json or base + ".r6.json", encoding="utf-8"))
    start = a.start if a.start is not None else data.get("start", 0)
    W, H = data["frame_size"]
    chars = []
    for ch in data["characters"]:
        P = np.array(ch["frames"], float)
        n = len(P)
        if P.shape[1] < len(J):  # старый файл: локти/колени — середины рук/ног (направления как раньше)
            mids = [(P[:, J["neck"]] + P[:, J[e]]) / 2 for e in ("l_hand", "r_hand")]
            mids += [(P[:, J["hip"]] + P[:, J[e]]) / 2 for e in ("l_foot", "r_foot")]
            P = np.concatenate([P[:, :7], np.stack(mids, 1)], 1)
        T = np.array(ch.get("torso_px") or np.linalg.norm(P[:, J["neck"]] - P[:, J["hip"]], axis=1), float)
        A = angles(P)
        keys = sorted(set().union(*[set(v) for v in ch["keys"].values()])) if ch.get("keys") and not a.all else range(n)
        A2, hip2, T2 = interp_keys(A, P[:, J["hip"]], T, keys, n)
        chars.append(dict(P=P, A=A2, hip=hip2, T=T2, keys=set(keys), facing=ch.get("facing", "r"), n=n))
    n = max(c["n"] for c in chars)

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out_path = base + "_compare.mp4"
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), data.get("fps", 30), (2 * W, H))
    for f in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        left, right = fr.copy(), (fr * 0.55 + 255 * 0.45).astype(np.uint8)
        for ci, c in enumerate(chars):
            if f >= c["n"]:
                continue
            col = COLORS[ci % len(COLORS)]
            p = c["P"][f]
            for u, v in BONES:
                cv2.line(left, tuple(np.round(p[J[u]]).astype(int)), tuple(np.round(p[J[v]]).astype(int)), col, 2, cv2.LINE_AA)
            for q in p:
                cv2.circle(left, tuple(np.round(q).astype(int)), 4, col, -1, cv2.LINE_AA)
            draw_r6(right, {k: v[f] for k, v in c["A"].items()}, c["hip"][f], c["T"][f], c["facing"], col)
        key = any(f in c["keys"] for c in chars)
        cv2.putText(left, f"{f + start}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(left, f"{f + start}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(right, "R6" + ("  [key]" if key and not a.all else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 2, cv2.LINE_AA)
        wr.write(np.hstack([left, right]))
    wr.release()
    print("Готово:", out_path)


if __name__ == "__main__":
    main()
