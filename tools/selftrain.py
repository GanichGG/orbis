"""Самообучение Orbis: прогон по роликам без разметки -> «уверенные» кадры -> weights/pseudo/*.npz для train.py --pseudo.

    python tools/selftrain.py --weights weights/orbis.pt видео1.mp4 видео2.mov ...
    python tools/selftrain.py --weights weights/orbis.pt папка_с_роликами/

Персонажи берутся из <видео>.s2r.json (если ролик уже открывали в stick2r6.py — там клики и разметки),
иначе открывается окно пипетки (кликнуть персонажей, Enter). Кадр попадает в обучение, только если:
  - трекер уверен во всех 11 точках и не считает кадр спорным;
  - точки плавно продолжают соседние кадры (нет рывка);
  - сеть ставит те же точки на зеркальном кадре (согласие с самой собой).
Цикл: selftrain -> train.py --pseudo weights/pseudo -> selftrain с новыми весами -> ...
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "stickpose"), ROOT]

import colorseg as CS          # noqa: E402
import pipeline as PL          # noqa: E402
import track2 as T             # noqa: E402
import stick2r6 as APP         # noqa: E402
from model import load, device, predict, peaks  # noqa: E402

EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
CH_OF = {"head": "head", "neck": "neck", "hip": "hip", "l_hand": "hands", "r_hand": "hands", "l_foot": "feet",
         "r_foot": "feet", "l_elbow": "elbows", "r_elbow": "elbows", "l_knee": "knees", "r_knee": "knees"}


def mirror_peaks(model, inputs):
    """Пики сети на зеркальном входе, возвращённые в координаты кадра: {кадр: {канал: [(x, y)]}}."""
    S = PL.synth.S
    idx = [f for f, v in enumerate(inputs) if v is not None]
    out = {}
    for i in range(0, len(idx), 64):
        chunk = idx[i:i + 64]
        X = np.stack([inputs[f][0][:, :, ::-1] for f in chunk])
        for f, hm in zip(chunk, predict(model, X)):
            M = inputs[f][1]
            d = {}
            for c, name in enumerate(PL.CHANNELS[:hm.shape[0]]):
                pk = peaks(hm[c], k=2 if name in ("hands", "feet", "elbows", "knees") else 1)
                pts = np.array([[S - 1 - p[0], p[1]] for p in pk])
                d[name] = CS.crop_to_frame(pts, M)
            out[f] = d
    return out


def select(r, frames_c, inputs, mir, tol_mirror=0.12, tol_smooth=0.15, min_gap=2):
    """Кадры-кандидаты в обучение (см. правила в начале файла)."""
    P, C, Tt, n = r["P"], r["C"], r["T"], len(r["P"])
    cuts = set(r["cuts"])
    keep, last = [], -10
    for f in range(1, n - 1):
        if inputs[f] is None or frames_c[f] is None or f in cuts or (f + 1) in cuts:
            continue
        if r["sev"][f] > 0.3 or C[f, :7].min() < 0.6 or C[f, 7:].min() < 0.7:
            continue
        acc = np.linalg.norm(P[f] - (P[f - 1] + P[f + 1]) / 2, axis=1) / Tt[f]
        if acc.max() > tol_smooth:
            continue
        m = mir.get(f)
        if m is None or any(len(m.get(CH_OF[j], [])) == 0 or
                            np.min(np.linalg.norm(m[CH_OF[j]] - P[f, i], axis=1)) / Tt[f] > tol_mirror
                            for i, j in enumerate(T.JOINTS)):
            continue
        if f - last >= min_gap:
            keep.append(f)
            last = f
    return keep


def pick(video_path, video):
    """Клики персонажей: из .s2r.json или окном пипетки."""
    proj = os.path.splitext(video_path)[0] + ".s2r.json"
    if os.path.exists(proj):
        pj = json.load(open(proj, encoding="utf-8"))
        if pj.get("n") == len(video):
            return pj["ref"], pj["colors"], pj["points"], [{int(k): v for k, v in L.items()} for L in pj["labels"]]
    import ui

    def preview(fr, colors, pts):
        return CS.ColorSegmenter(colors).step(fr, init_points=pts)
    ref, colors, pts = ui.pick_colors(lambda i: video[i], len(video), preview, CS.sample_color)
    if colors is None:
        return None
    APP.save_json(proj, {"v": 2, "n": len(video), "start": 0, "ref": ref, "colors": colors, "points": pts,
                         "labels": [{} for _ in colors], "skipped": [[] for _ in colors]})
    return ref, colors, pts, [{} for _ in colors]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "orbis.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "weights", "pseudo"))
    a = ap.parse_args()
    files = []
    for v in a.videos:
        files += sorted(os.path.join(v, f) for f in os.listdir(v) if f.lower().endswith(EXT)) if os.path.isdir(v) else [v]
    model = load(a.weights, device())
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for path in files:
        video = APP.Video(path)
        n = len(video)
        got = pick(path, video)
        if got is None:
            continue
        ref, colors, pts, labels = got
        nc = len(colors)
        masks = [None] * n
        for d in (1, -1):
            seg = CS.ColorSegmenter(colors)
            masks[ref] = seg.step(video[ref], init_points=pts)
            for f in (range(ref + 1, n) if d > 0 else range(ref - 1, -1, -1)):
                masks[f] = seg.step(video[f])
        inputs = PL.build_inputs(masks, nc)
        del masks
        X, K = [], []
        for c in range(nc):
            fc = PL.run_model(model, inputs[c])
            lab = labels[c] if c < len(labels) and labels[c] else {
                f: APP.guess_from_model(fc[f]) for f in APP.select_diverse([x["feat"] if x else None for x in fc], 6)
                if fc[f] and fc[f].get("ok")}
            r = T.track(fc, lab, n)
            if r is None:
                continue
            keep = select(r, fc, inputs[c], mirror_peaks(model, inputs[c]))
            for f in keep:
                x, M, _ = inputs[c][f]
                X.append(x)
                K.append(CS.frame_to_crop(r["P"][f], M).astype(np.float32))
            print(f"{os.path.basename(path)}: персонаж {c + 1} — уверенных кадров {len(keep)} из {n}")
        if X:
            name = os.path.splitext(os.path.basename(path))[0]
            np.savez_compressed(os.path.join(a.out, name + ".npz"), x=np.stack(X), kp=np.stack(K))
            total += len(X)
    print(f"Готово: {total} кадров в {a.out}. Дальше: cd stickpose && python train.py --resume ../weights/orbis.pt "
          f"--steps 60000 --real ../weights/real --pseudo ../weights/pseudo --out ../weights/orbis.pt")


if __name__ == "__main__":
    main()
