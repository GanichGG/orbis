"""stick2r6 — видео со стикменами -> анимация R6 для Blender (нейросеть StickPose + ваши ответы).

Запуск:   python stick2r6.py путь/к/видео.mp4
Опции:    --start 0 --end 300     кусок видео (номера кадров)
          --first 6               сколько разных поз спросить в начале у каждого персонажа
          --view side|front       вид в видео: сбоку (по умолчанию) или спереди
          --facing r,l            куда смотрит каждый персонаж в виде сбоку (r — вправо, l — влево)
          --weights путь.pt       веса модели (по умолчанию weights/stickpose.pt рядом со скриптом)
          --fresh                 начать заново (иначе продолжит сохранённую разметку)
Результат: <видео>.r6.json — его открывает blender_import_r6.py.
Ваши разметки также сохраняются в weights/real/*.npz — на них можно дообучать модель (train.py --real).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stickpose"))
sys.path.insert(0, HERE)

import colorseg as CS          # noqa: E402
import pipeline as PL          # noqa: E402
import track2 as T             # noqa: E402
import postprocess as PP       # noqa: E402
from model import load, device  # noqa: E402

JOINTS = T.JOINTS
MAX_SIDE = 960


def fit(frame):
    h, w = frame.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s >= 1:
        return frame
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


class Video:
    def __init__(self, path, start=0, end=None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"Не открывается видео: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.buf = []
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i >= start and (end is None or i < end):
                self.buf.append(cv2.imencode(".png", fit(fr))[1])
            i += 1
            if end is not None and i >= end:
                break
        cap.release()
        if not self.buf:
            sys.exit("В видео нет кадров в выбранном диапазоне")
        self._cache = {}

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, i):
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[i] = cv2.imdecode(self.buf[i], cv2.IMREAD_COLOR)
        return self._cache[i]


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


def select_diverse(feats, n, time_weight=3.0):
    idx = [i for i, f in enumerate(feats) if f is not None]
    if not idx:
        return []
    total = max(1, len(feats) - 1)
    X = np.stack([np.append(feats[i], time_weight * i / total) for i in idx])
    chosen = sorted({0, len(idx) - 1})
    d = np.min([np.linalg.norm(X - X[k], axis=1) for k in chosen], axis=0)
    while len(chosen) < min(n, len(idx)):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(X - X[k], axis=1))
    return sorted(idx[k] for k in chosen)


def guess_from_model(fc):
    """Догадка сети для окна разметки (лево/право по порядку пиков — пользователь поправит Q/W)."""
    if not fc or not fc.get("ok"):
        return None
    pk = fc["peaks"]
    g = {"head": pk["head"][0][:2], "neck": pk["neck"][0][:2], "hip": pk["hip"][0][:2]}
    hands = pk["hands"][:2] + pk["hands"][:1]
    feet = pk["feet"][:2] + pk["feet"][:1]
    g["l_hand"], g["r_hand"] = hands[0][:2], hands[1][:2]
    g["l_foot"], g["r_foot"] = feet[0][:2], feet[1][:2]
    return {k: [float(v[0]), float(v[1])] for k, v in g.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--first", type=int, default=6)
    ap.add_argument("--view", default="side", choices=["side", "front"])
    ap.add_argument("--facing", default="")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--keys", type=float, default=4.0, help="допуск ключевых кадров в градусах (0 — ключ на каждом кадре)")
    ap.add_argument("--no_plant", action="store_true", help="не фиксировать стопы на земле")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "stickpose.pt"))
    ap.add_argument("--finetune_every", type=int, default=0, help="дообучать сеть после каждых N ответов (0 — авто: 4 на видеокарте, 10 на процессоре)")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    import ui

    def log(msg, same_line=False):
        print(("\r" + msg + " " * 8) if same_line else msg, end="" if same_line else "\n", flush=True)

    if not os.path.exists(a.weights):
        sys.exit(f"Нет весов модели: {a.weights}")
    dev = device()
    if a.finetune_every == 0:
        a.finetune_every = 4 if dev == "cuda" else 10
    model = load(a.weights, dev)
    log(f"модель: {os.path.basename(a.weights)} на {dev}")

    base = os.path.splitext(a.video)[0]
    proj_path, out_path = base + ".s2r.json", base + ".r6.json"
    video = Video(a.video, a.start, a.end)
    n = len(video)
    log(f"кадров: {n}, {video.fps:.1f} fps")

    proj = None
    if not a.fresh and os.path.exists(proj_path):
        proj = json.load(open(proj_path, encoding="utf-8"))
        if proj.get("n") != n or proj.get("start") != a.start or proj.get("v") != 2:
            log("Сохранённая разметка не подходит к этому диапазону/версии — начинаю заново.")
            proj = None
    if proj is None:
        def preview(fr, colors, pts):
            seg = CS.ColorSegmenter(colors)
            return seg.step(fr, init_points=pts)
        ref, colors, pts = ui.pick_colors(lambda i: video[i], n, preview, CS.sample_color)
        if colors is None:
            return
        proj = {"v": 2, "n": n, "start": a.start, "ref": ref, "colors": colors, "points": pts,
                "labels": [{} for _ in colors], "skipped": [[] for _ in colors]}
        save_json(proj_path, proj)
    nc = len(proj["colors"])

    # ---- выделение персонажей во всех кадрах ----
    ui.busy("Выделяю персонажей... (прогресс в консоли)")
    masks = [None] * n
    for direction in (+1, -1):
        seg = CS.ColorSegmenter(proj["colors"])
        masks[proj["ref"]] = seg.step(video[proj["ref"]], init_points=proj["points"])
        rng = range(proj["ref"] + 1, n) if direction > 0 else range(proj["ref"] - 1, -1, -1)
        t0 = time.time()
        for f in rng:
            masks[f] = seg.step(video[f])
            if time.time() - t0 > 1:
                log(f"  кадр {f + 1}/{n}", same_line=True)
                t0 = time.time()
    log("")
    inputs = PL.build_inputs(masks, nc)
    del masks

    labels = [{int(k): v for k, v in L.items()} for L in proj["labels"]]
    skipped = [set(s) for s in proj["skipped"]]
    frames_c = [PL.run_model(model, inputs[c]) for c in range(nc)]
    results = [None] * nc
    since_ft = [0] * nc

    def persist():
        proj["labels"] = [{str(k): v for k, v in L.items()} for L in labels]
        proj["skipped"] = [sorted(s) for s in skipped]
        save_json(proj_path, proj)

    def refresh(c, retrain=False):
        nonlocal model
        if retrain:
            samples = sum((PL.labels_to_samples(inputs[k], labels[k]) for k in range(nc)), [])
            if len(samples) >= 3:
                ui.busy(f"Дообучаю сеть на ваших ответах ({len(samples)})...")
                ft_steps = min(300, 60 + 20 * len(samples)) if dev == "cuda" else min(80, 30 + 5 * len(samples))
                model = PL.finetune(model, samples, steps=ft_steps, log=log)
                for k in range(nc):
                    frames_c[k] = PL.run_model(model, inputs[k])
        ui.busy("Пересчитываю движение...")
        for k in (range(nc) if retrain else [c]):
            results[k] = T.track(frames_c[k], labels[k], n, skipped=skipped[k])

    def ask(c, f, reason, progress):
        est = labels[c].get(f)
        if est is None and results[c] is not None:
            est = {j: results[c]["P"][f, i].tolist() for i, j in enumerate(JOINTS)}
        if est is None:
            est = guess_from_model(frames_c[c][f])
        fc = frames_c[c][f]
        st, pts = ui.label_frame(video[f], None, est, f"Персонаж {c + 1} из {nc}, кадр {f + 1}/{n}", reason, c, progress)
        if st == "ok":
            labels[c][f] = pts
            skipped[c].discard(f)
            since_ft[c] += 1
            persist()
        elif st == "skip":
            skipped[c].add(f)
            persist()
        return st

    # ---- ШАГ 2а: разные позы ----
    for c in range(nc):
        feats = [fc["feat"] if fc else None for fc in frames_c[c]]
        todo = [f for f in select_diverse(feats, a.first) if f not in labels[c] and f not in skipped[c]]
        for k, f in enumerate(todo):
            st = ask(c, f, "новая поза — нужна для обучения", f"[поза {k + 1}/{len(todo)}]")
            if st == "close":
                return
            if st == "stop":
                break
        if not labels[c]:
            sys.exit(f"Персонаж {c + 1}: нужен хотя бы один размеченный кадр.")
    refresh(0, retrain=True)
    since_ft = [0] * nc

    # ---- ШАГ 2б: спорные кадры ----
    asked = 0
    while True:
        w = T.worst_frame(results)
        left = T.count_flagged(results)
        if w is None:
            log(f"Спорных кадров не осталось (вопросов: {asked}).")
            break
        _, c, f = w
        reason = "; ".join(dict.fromkeys(results[c]["why"][f])) or "низкая уверенность"
        st = ask(c, f, reason, f"[спорных ~{left}]")
        if st == "close":
            return
        if st == "stop":
            break
        asked += 1
        retrain = a.finetune_every > 0 and sum(since_ft) >= a.finetune_every
        if retrain:
            since_ft = [0] * nc
        refresh(c, retrain)

    # ---- ШАГ 3: проверка ----
    f0 = 0
    while True:
        act, f0, c = ui.review(lambda i: video[i], n, results, f0)
        if act == "close":
            return
        if act == "edit":
            if ask(c, f0, "правка при проверке", "") == "close":
                return
            refresh(c)
            continue
        break

    # ---- экспорт ----
    facing = [s.strip() for s in a.facing.split(",") if s.strip()]
    out = {"format": "stick2r6", "version": 2, "fps": video.fps, "view": a.view,
           "frame_size": [video[0].shape[1], video[0].shape[0]], "characters": []}
    for c in range(nc):
        r = results[c]
        P = r["P"].copy()
        if a.smooth and a.smooth >= 3 and n > a.smooth:
            from scipy.signal import savgol_filter
            w = a.smooth if a.smooth % 2 else a.smooth + 1
            cuts = [0] + r["cuts"] + [n]
            for s0, s1 in zip(cuts, cuts[1:]):  # не сглаживать через смену плана
                if s1 - s0 > w:
                    P[s0:s1] = savgol_filter(P[s0:s1], w, 2, axis=0, mode="interp")
            for f in r["anchors"]:
                P[f] = r["P"][f]
        contacts, keys = {}, None
        if not a.no_plant:
            P, contacts = PP.plant_feet(P, r["T"], r["cuts"])
        if a.keys > 0:
            keys = PP.keyframes(P, r["T"], r["cuts"], r["anchors"], tol_deg=a.keys)
        ch = {"name": f"Char{c + 1}", "facing": facing[c] if c < len(facing) else "r",
              "joints": JOINTS, "frames": np.round(P, 2).tolist(), "labeled": r["anchors"],
              "cuts": r["cuts"], "torso_px": np.round(r["T"], 2).tolist(),
              "contacts": {JOINTS[j]: v for j, v in contacts.items()}}
        if keys:
            ch["keys"] = keys
            log(f"Персонаж {c + 1}: ключей на часть ~{int(np.mean([len(v) for v in keys.values()]))} из {n} кадров")
        out["characters"].append(ch)
    save_json(out_path, out)
    # разметки -> датасет для будущего обучения модели
    try:
        os.makedirs(os.path.join(HERE, "weights", "real"), exist_ok=True)
        samples = sum((PL.labels_to_samples(inputs[k], labels[k]) for k in range(nc)), [])
        if samples:
            np.savez_compressed(os.path.join(HERE, "weights", "real", os.path.basename(base) + ".npz"),
                                x=np.stack([s[0] for s in samples]).astype(np.float16),
                                kp=np.stack([s[1] for s in samples]))
    except Exception as e:
        log(f"(не удалось сохранить датасет: {e})")
    log(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
