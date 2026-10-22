"""Тестовое видео в стиле Stick Nodes с известной разметкой: 2 персонажа, текстурный фон,
наезды камеры и смена плана. -> video.mp4 + gt.npy (кадры, персонажи, 7 точек, xy)."""
import math
import sys

import cv2
import numpy as np

import synth

W, H, FPS = 960, 540, 30


def keyposes(rng, k):
    return [synth.sample_pose(rng) for _ in range(k)]


def interp(p0, p1, t):
    out = {}
    for k in p0:
        a, b = p0[k], p1[k]
        if k in ("torso", "chest", "head", "l_arm", "r_arm", "l_leg", "r_leg", "l_arm2", "r_arm2", "l_leg2", "r_leg2"):
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            out[k] = a + d * t
        else:
            out[k] = a * (1 - t) + b * t
    return out


def ease(t):
    return t * t * (3 - 2 * t)


def background(rng):
    bg = np.zeros((H * 2, W * 2, 3), np.uint8)
    for y in range(H * 2):
        if y < H * 1.1:
            c = np.array([45, 27, 9]) * (0.7 + 0.3 * y / (H * 1.1))
        else:
            c = np.array([79, 91, 92]) * (0.8 + 0.2 * rng.random())
        bg[y] = c
    noise = rng.normal(0, 6, (H * 2, W * 2, 1))
    noise[: int(H * 1.1)] *= 0.3
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def sf2_track(rng, n, T, x0):
    """Боец из анимаций Shadow Fight 2: случайные приёмы подряд, вид точно сбоку, постоянные пропорции и стиль."""
    src = synth.mocap()["sf2"]
    base = synth.MocapFigure.__new__(synth.MocapFigure)
    proto = synth.MocapFigure(rng, T, np.array([x0, 0.0]))
    while proto.source != "sf2":
        proto = synth.MocapFigure(rng, T, np.array([x0, 0.0]))
    side = 1 if rng.random() < 0.5 else -1
    c, s_ = 0.0, float(side)   # yaw ±90°: ось «вперёд» SF2 -> экран
    proto.R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    out, hipx = [], x0
    while len(out) < n:
        i = int(rng.integers(len(src["start"])))
        f = base
        f.__dict__.update(proto.__dict__)
        f.M, f.s0, f.n = src, int(src["start"][i]), int(src["length"][i])
        P0 = src["pos"][f.s0:f.s0 + f.n]
        J = src["J"]
        f.torso3 = float(np.median(np.linalg.norm(P0[:, J["neck"]] - P0[:, J["hip"]], axis=1))) + 1e-6
        f.t, f.root0 = 0.0, f._root(0.0)
        f.hip = np.array([hipx, 0.0])
        for o in range(f.n):
            pts, segs = f(o)
            out.append((pts, segs))
        hipx = out[-1][0]["hip"][0]
        if abs(hipx - x0) > 200:   # не уходить за кадр: следующий приём — снова с места старта
            hipx = x0
    return out[:n], proto


def main(out_prefix, n=120, seed=7, source="random"):
    rng = np.random.default_rng(seed)
    bg = background(rng)
    chars = []
    if source == "sf2":
        tracks = [sf2_track(rng, n, 70.0, [380.0, 620.0][ci] + W / 2) for ci in range(2)]
    for ci in range(2):
        kp = keyposes(rng, n // 10 + 2)
        st = synth.sample_style(rng)
        st.update(thick=rng.uniform(0.16, 0.24), head=rng.uniform(0.24, 0.32), ring=False, neck_line=False, round=True)
        chars.append({"kp": kp, "style": st, "col": [(0, 8, 108), (0, 0, 0)][ci],
                      "x0": [380.0, 620.0][ci], "vx": [1.2, -1.0][ci]})
    # камера: зум и сдвиг меняются плавно, на кадре 70 — смена плана
    cams = []
    z, cx, cy = 1.0, W, H
    for f in range(n):
        if f == 70:
            z, cx, cy = 0.55, W + 80, H + 60
        z *= 1.0 + (0.006 if 25 <= f < 45 else 0.0)
        cams.append((z, cx, cy))
    vw = cv2.VideoWriter(out_prefix + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    GT = np.zeros((n, 2, 11, 2), np.float32)
    for f in range(n):
        z, cx, cy = cams[f]
        world = bg.copy()
        for ci, ch in enumerate(chars):
            T = 70.0
            if source == "sf2":   # анимация SF2: таз из записи, земля — по стопам
                pts, segs = tracks[ci][0][f]
                pts = {k: (v + np.array([0.0, H * 1.1 + 20]) if k != "head_dir" else v) for k, v in pts.items()}
            else:
                seg, t = divmod(f, 10)
                p = interp(ch["kp"][seg], ch["kp"][seg + 1], ease(t / 10))
                p["torso"] = math.pi + 0.6 * math.sin(f / 9 + ci)  # больше «стоячих» поз
                hip = np.array([ch["x0"] + ch["vx"] * f + W / 2, H * 1.1 + 20 + rng.normal(0, 0.5)])
                pts, segs = synth.forward(p, hip, T)
            m = np.zeros((H * 2 * synth.SS, W * 2 * synth.SS), np.uint8)
            head = synth.render(m, pts, segs, T, ch["style"])
            m = cv2.resize(m, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
            a = (m.astype(np.float32) / 255.0)[..., None]
            world = (world * (1 - a) + np.array(ch["col"], np.float32) * a).astype(np.uint8)
            GT[f, ci] = synth._keypoints(pts, head)
        # камера: world -> кадр
        M = np.array([[z, 0, W / 2 - cx * z], [0, z, H / 2 - cy * z]], np.float32)
        frame = cv2.warpAffine(world, M, (W, H), flags=cv2.INTER_AREA)
        GT[f] = GT[f] * z + M[:, 2]
        vw.write(frame)
    vw.release()
    np.save(out_prefix + "_gt.npy", GT)
    print("ok", out_prefix)


if __name__ == "__main__":
    main(sys.argv[1], source="sf2" if "--sf2" in sys.argv else "random")
