"""Экзамен Orbis на настоящих кадрах: одна линейка для всех изменений.

    python tools/exam.py make видео.mp4 [--n 30] [--start 0 --end 300]   # разметить экзаменационные кадры (один раз)
    python tools/exam.py run weights/orbis.pt [weights/другие.pt ...]    # сдать экзамен (сравнить модели)

make: пипетка (или клики из <видео>.s2r.json) -> N кадров равномерно по ролику -> окно разметки на каждого
персонажа: точки расставлены сетью, проверьте и поправьте КАЖДУЮ из 11 (это эталон). S — пропустить кадр.
Сохраняется в weights/exam/<видео>.npz. На этих кадрах не учимся (train.py их не берёт).

run: сеть по каждому кадру (без трекера и подсказок) -> ошибка углов частей R6 по правилу переноса
(торс таз->шея, голова шея->голова, рука локоть->кисть, нога колено->стопа; лево/право не штрафуем)
и доля точек ближе 10% длины торса. Чем меньше градусов — тем лучше.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "stickpose"), os.path.join(ROOT, "app"), ROOT]

import colorseg as CS          # noqa: E402
import pipeline as PL          # noqa: E402
import track2 as T             # noqa: E402
from model import load, predict, peaks, device  # noqa: E402

EXAM = os.path.join(ROOT, "weights", "exam")
J = {j: i for i, j in enumerate(T.JOINTS)}


def make(a):
    import stick2r6 as APP
    import ui
    video = APP.Video(a.video, a.start, a.end)
    n = len(video)
    proj = os.path.splitext(a.video)[0] + ".s2r.json"
    pj = json.load(open(proj, encoding="utf-8")) if os.path.exists(proj) else None
    if pj and pj.get("n") == n and pj.get("start", 0) == a.start:
        ref, colors, pts = pj["ref"], pj["colors"], pj["points"]
    else:
        def preview(fr, colors, pts):
            return CS.ColorSegmenter(colors).step(fr, init_points=pts)
        ref, colors, pts = ui.pick_colors(lambda i: video[i], n, preview, CS.sample_color)
        if colors is None:
            return
    nc = len(colors)
    static = CS.static_mask(lambda i: video[i], n, colors)   # интерфейс игры, земля
    masks = [None] * n
    for d in (1, -1):
        seg = CS.ColorSegmenter(colors, exclude=static)
        masks[ref] = CS.pack(seg.step(video[ref], init_points=pts))
        for f in (range(ref + 1, n) if d > 0 else range(ref - 1, -1, -1)):
            masks[f] = CS.pack(seg.step(video[f]))
    inputs = PL.build_inputs(masks, nc)
    del masks
    model = load(a.weights, device()) if os.path.exists(a.weights) else None
    frames = np.unique(np.linspace(1, n - 2, a.n).round().astype(int))
    X, M, G, meta = [], [], [], []
    for k, f in enumerate(frames):
        for c in range(nc):
            if inputs[c][f] is None:
                continue
            est = None
            if model is not None:
                fc = PL.run_model(model, [inputs[c][f]])[0]
                est = APP.guess_from_model(fc)
            st, p = ui.label_frame(video[f], None, est, f"ЭКЗАМЕН: персонаж {c + 1} из {nc}, кадр {f + 1}/{n}",
                                   "эталон — проверьте и поправьте каждую из 11 точек", c, f"[{k + 1}/{len(frames)}]")
            if st == "close":
                return
            if st == "stop":
                break
            if st != "ok" or any(p.get(j) is None for j in T.JOINTS):
                continue
            if bad_gt(np.array([p[j] for j in T.JOINTS], float)):
                print(f"кадр {f + 1}: левая и правая точка совпали — в экзамен не беру (разведите их)")
                continue
            X.append(inputs[c][f][0])
            M.append(inputs[c][f][1])
            G.append([p[j] for j in T.JOINTS])
            meta.append(f"{os.path.basename(a.video)}:{f + a.start}:{c}")
        else:
            continue
        break
    if not X:
        print("Ничего не размечено.")
        return
    os.makedirs(EXAM, exist_ok=True)
    out = os.path.join(EXAM, os.path.splitext(os.path.basename(a.video))[0] + ".npz")
    np.savez_compressed(out, x=np.stack(X), M=np.stack(M), gt=np.array(G, np.float32), meta=np.array(meta))
    print(f"Экзамен: {len(X)} кадров -> {out}")


def predict_points(model, x, Mx):
    """Сеть -> 7 «одиночных» точек + пары (кисти, стопы, локти, колени) как наборы кандидатов, в координатах кадра."""
    hm = predict(model, x[None])[0]
    out = {}
    for c, name in enumerate(PL.CHANNELS[:hm.shape[0]]):
        pk = peaks(hm[c], k=2 if name in ("hands", "feet", "elbows", "knees") else 1)
        pts = CS.crop_to_frame([p[:2] for p in pk], Mx)
        if name in ("hands", "feet", "elbows", "knees") and len(pts) == 1:
            pts = np.vstack([pts, pts])
        out[name] = pts
    return out


def angle(v):
    return np.arctan2(v[..., 0], v[..., 1])


def adiff(a, b):
    return np.degrees(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def score(pred, g):
    """Ошибки углов частей R6 (градусы) и точек (доли торса) на одном кадре; лево/право не штрафуем."""
    Tl = np.linalg.norm(g[J["neck"]] - g[J["hip"]]) + 1e-6
    res = {"Торс": adiff(angle(pred["neck"][0] - pred["hip"][0]), angle(g[J["neck"]] - g[J["hip"]])),
           "Голова": adiff(angle(pred["head"][0] - pred["neck"][0]), angle(g[J["head"]] - g[J["neck"]]))}
    pts = {j: np.linalg.norm(pred[j][0] - g[J[j]]) / Tl for j in ("head", "neck", "hip")}
    for part, (ends, mids, gl, gr, ml, mr, root) in {
            "Руки": ("hands", "elbows", "l_hand", "r_hand", "l_elbow", "r_elbow", "neck"),
            "Ноги": ("feet", "knees", "l_foot", "r_foot", "l_knee", "r_knee", "hip")}.items():
        E = pred[ends]
        Mid = pred.get(mids)
        if Mid is None:  # модель v1 без локтей/колен: направление от шеи/таза
            Mid = np.vstack([pred[root][0], pred[root][0]])
        # каждой кисти/стопе — ближайший локоть/колено; затем лучшее соответствие сторон
        segs = [(Mid[int(np.argmin(np.linalg.norm(Mid - e, axis=1)))], e) for e in E[:2]]
        best = None
        for perm in ((0, 1), (1, 0)):
            err = [adiff(angle(segs[perm[i]][1] - segs[perm[i]][0]), angle(g[J[h]] - g[J[m]]))
                   for i, (h, m) in enumerate(((gl, ml), (gr, mr)))]
            d = [np.linalg.norm(segs[perm[i]][1] - g[J[h]]) / Tl for i, h in enumerate((gl, gr))]
            dm = [np.linalg.norm(segs[perm[i]][0] - g[J[m]]) / Tl for i, m in enumerate((ml, mr))]
            if best is None or sum(err) < sum(best[0]):
                best = (err, d, dm)
        res[part] = float(np.mean(best[0]))
        res[part + "_все"] = best[0]
        pts[ends] = float(np.mean(best[1]))
        pts[mids] = float(np.mean(best[2]))
    return res, pts


def load_exam():
    files = sorted(os.path.join(EXAM, f) for f in os.listdir(EXAM) if f.endswith(".npz")) if os.path.isdir(EXAM) else []
    return [np.load(f) for f in files]


def bad_gt(g, tol=3.0):
    """Эталон с ошибкой: левая и правая кисть/локоть/стопа/колено в одной точке."""
    return any(np.linalg.norm(g[J[a]] - g[J[b]]) < tol for a, b in
               (("l_hand", "r_hand"), ("l_elbow", "r_elbow"), ("l_foot", "r_foot"), ("l_knee", "r_knee")))


def exam_score(model, data):
    """-> (оценка 0..10, ошибки частей {часть: [°]}, точки {точка: [доля торса]})."""
    parts, pts = {}, {}
    for d in data:
        for x, Mx, g in zip(d["x"], d["M"], d["gt"]):
            if bad_gt(g):
                continue
            r, p = score(predict_points(model, x, Mx), g)
            for k in ("Торс", "Голова"):
                parts.setdefault(k, []).append(r[k])
            for k in ("Руки", "Ноги"):
                parts.setdefault(k, []).extend(r[k + "_все"])
            for k, v in p.items():
                pts.setdefault(k, []).append(v)
    allp = np.concatenate([np.ravel(v) for v in parts.values()])
    return max(0.0, min(10.0, 10 * (30 - allp.mean()) / 27)), parts, pts   # 3° -> 10, 10° -> 7.4, 30° -> 0


def run(a):
    data = load_exam()
    if not data:
        sys.exit("Нет экзамена: сначала python tools/exam.py make видео.mp4")
    print(f"экзамен: {sum(len(d['x']) for d in data)} кадров из {len(data)} роликов")
    for w in a.weights:
        mark, parts, pts = exam_score(load(w, device()), data)
        allp = np.concatenate([np.ravel(v) for v in parts.values()])
        print(f"\n{os.path.basename(w)}:  ОЦЕНКА {mark:.1f}/10   (средняя ошибка {allp.mean():.1f}°, "
              f"частей точнее 15°: {(allp < 15).mean() * 100:.0f}%; Epigram — от 7)")
        print("  части R6: " + ",  ".join(f"{k} {np.mean(v):.1f}°" for k, v in parts.items()))
        print("  точки ближе 10% торса: " + ",  ".join(f"{k} {(np.array(v) < 0.1).mean() * 100:.0f}%" for k, v in pts.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("video")
    m.add_argument("--n", type=int, default=30)
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--end", type=int, default=None)
    m.add_argument("--weights", default=os.path.join(ROOT, "weights", "orbis.pt"))
    r = sub.add_parser("run")
    r.add_argument("weights", nargs="+")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else run(a)


if __name__ == "__main__":
    main()
