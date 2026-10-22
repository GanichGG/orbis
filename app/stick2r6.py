"""stick2r6 — видео со стикменами -> анимация R6 для Blender (нейросеть StickPose + ваши ответы).

Запуск:   python app/stick2r6.py путь/к/видео.mp4   (или меню: python orbis.py)
Опции:    --start 0 --end 300     кусок видео (номера кадров)
          --first 6               сколько разных поз спросить в начале у каждого персонажа
          --view side|front       вид в видео: сбоку (по умолчанию) или спереди
          --facing r,l            куда смотрит каждый персонаж в виде сбоку (r — вправо, l — влево; без него — по коленям)
          --weights путь.pt       веса модели (по умолчанию weights/orbis.pt, если есть, иначе weights/stickpose.pt)
          --quiet 2               не спрашивать кадры в ±N от уже размеченных (0 — спрашивать всё подряд)
          --ask 40                не больше N вопросов про спорные кадры на персонажа (по умолчанию кадров/12)
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # корень проекта (скрипт лежит в app/)
sys.path.insert(0, os.path.join(HERE, "stickpose"))
sys.path.insert(0, os.path.join(HERE, "app"))

import colorseg as CS          # noqa: E402
import pipeline as PL          # noqa: E402
import track2 as T             # noqa: E402
import postprocess as PP       # noqa: E402
import motion as MO            # noqa: E402
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
    torso = max(8.0, float(np.hypot(*np.subtract(g["neck"], g["hip"]))))
    for (l, r), ch in ((("l_hand", "r_hand"), "hands"), (("l_foot", "r_foot"), "feet")):
        c = [np.asarray(p[:2], float) for p in pk[ch][:2]]
        if len(c) < 2:  # сеть нашла одну — вторую рядом, но отдельно (в одну точку пользователь их не примет)
            c.append(c[0] + [0.25 * torso, 0.0])
        g[l], g[r] = c[0], c[1]
    # локти/колени: каждому концу конечности — свой пик (не один на двоих); нет пика — середина конечности
    for (jl, jr), root, (el, er), ch in ((("l_elbow", "r_elbow"), "neck", ("l_hand", "r_hand"), "elbows"),
                                         (("l_knee", "r_knee"), "hip", ("l_foot", "r_foot"), "knees")):
        mids = [(np.asarray(g[root]) + np.asarray(g[e])) / 2 for e in (el, er)]
        c = [np.asarray(p[:2], float) for p in pk.get(ch, [])[:2]]
        if len(c) == 2:
            same = np.hypot(*(c[0] - mids[0])) + np.hypot(*(c[1] - mids[1]))
            swap = np.hypot(*(c[1] - mids[0])) + np.hypot(*(c[0] - mids[1]))
            g[jl], g[jr] = (c[0], c[1]) if same <= swap else (c[1], c[0])
        elif len(c) == 1:
            k = int(np.argmin([np.hypot(*(c[0] - m)) for m in mids]))
            g[(jl, jr)[k]], g[(jl, jr)[1 - k]] = c[0], mids[1 - k]
        else:
            g[jl], g[jr] = mids
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
    ap.add_argument("--keys", type=float, default=10.0, help="допуск промежуточных ключей в градусах (0 — ключ на каждом кадре)")
    ap.add_argument("--old_keys", action="store_true", help="старый выбор ключей (RDP по углам), без событий")
    ap.add_argument("--edit", action="store_true", help="«правка кадра»: кость держит позу, пока не сдвинется заметнее шума")
    ap.add_argument("--no_plant", action="store_true", help="не фиксировать стопы на земле")
    w_def = os.path.join(HERE, "weights", "orbis.pt")
    ap.add_argument("--weights", default=w_def if os.path.exists(w_def) else os.path.join(HERE, "weights", "stickpose.pt"))
    ap.add_argument("--finetune_every", type=int, default=0, help="дообучать сеть после каждых N ответов (0 — авто: 4 на видеокарте, 10 на процессоре)")
    ap.add_argument("--quiet", type=int, default=2)
    ap.add_argument("--ask_mode", default="keys", choices=["keys", "doubt"],
                    help="keys — сначала ключевые позы (удары, замахи, крайние), потом спорные; doubt — только спорные")
    ap.add_argument("--ask", type=int, default=0, help="вопросов про спорные кадры на персонажа (0 — кадров/12, не меньше 10)")
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
    # неподвижные области цвета персонажей (интерфейс игры, тёмная земля) — не персонажи
    static = CS.static_mask(lambda i: video[i], n, proj["colors"])
    if static is not None:
        log(f"Убраны неподвижные области того же цвета (интерфейс, земля): {static.mean() * 100:.0f}% кадра")
    masks = [None] * n
    for direction in (+1, -1):
        seg = CS.ColorSegmenter(proj["colors"], exclude=static)
        masks[proj["ref"]] = CS.pack(seg.step(video[proj["ref"]], init_points=proj["points"]))
        rng = range(proj["ref"] + 1, n) if direction > 0 else range(proj["ref"] - 1, -1, -1)
        t0 = time.time()
        for f in rng:
            masks[f] = CS.pack(seg.step(video[f]))
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
            results[k] = T.track(frames_c[k], labels[k], n, cfg={"quiet": a.quiet}, skipped=skipped[k])

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

    # ---- ШАГ 2б: спорные кадры (самые важные первыми, не больше бюджета на персонажа) ----
    budget = a.ask or max(10, n // 12)
    asked_c = [0] * nc
    asked = 0

    def key_pose_queue():
        """Ключевые позы (удары, замахи, приземления, крайние...) — их точность важнее всего для анимации.
        Порядок: важность события + сомнение трекера; рядом с уже размеченным не спрашиваем."""
        Ps = [MO.fix_swaps(results[k]["P"], results[k]["T"], results[k]["anchors"])[0] for k in range(nc)]
        q = []
        for k in range(nc):
            ev, _ = MO.detect(Ps[k], results[k]["T"], results[k]["cuts"], [Ps[j] for j in range(nc) if j != k])
            for x in ev:
                if MO.IMPORTANCE[x["type"]] >= 3:
                    q.append((MO.IMPORTANCE[x["type"]] + min(3.0, float(results[k]["sev"][x["f"]])), k, x["f"],
                              MO.RU[x["type"]]))
        return sorted(q, reverse=True)

    key_share = 0.7 if a.ask_mode == "keys" else 0.0   # доля бюджета на ключевые позы, остальное — на спорные
    queue = key_pose_queue() if key_share else []
    while True:
        w, reason = None, ""
        while queue:
            sc, c, f, name = queue.pop(0)
            near = any(abs(f - g) <= max(1, a.quiet) for g in list(labels[c]) + list(skipped[c]))
            if not near and asked_c[c] < budget * key_share:
                w, reason = (sc, c, f), f"ключевая поза: {name}"
                break
        if w is None:
            w = T.worst_frame([r if asked_c[k] < budget else None for k, r in enumerate(results)])
        if w is None:
            log(f"Вопросы закончились (задано: {asked}; бюджет {budget} на персонажа, больше — --ask).")
            break
        _, c, f = w
        reason = reason or "; ".join(dict.fromkeys(results[c]["why"][f])) or "низкая уверенность"
        st = ask(c, f, reason, f"[вопрос {asked_c[c] + 1}/{budget}]")
        asked_c[c] += 1
        if st == "close":
            return
        if st == "stop":
            break
        asked += 1
        retrain = a.finetune_every > 0 and sum(since_ft) >= a.finetune_every
        if retrain:
            since_ft = [0] * nc
        refresh(c, retrain)
        if retrain and key_share:   # сеть дообучилась — позы и события могли сдвинуться
            queue = key_pose_queue()

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
    out = {"format": "stick2r6", "version": 2, "fps": video.fps, "view": a.view, "start": a.start,
           "frame_size": [video[0].shape[1], video[0].shape[0]], "characters": []}
    # события движения всех персонажей (для «удар в противника» нужны траектории остальных)
    base = []
    for c in range(nc):
        Pc, nfix = MO.fix_swaps(results[c]["P"], results[c]["T"], results[c]["anchors"])
        if nfix:
            log(f"Персонаж {c + 1}: исправлено перескоков рук/ног: {nfix}")
        base.append(Pc)
    events, holds = [], []
    for c in range(nc):
        ev, ho = MO.detect(base[c], results[c]["T"], results[c]["cuts"], [base[k] for k in range(nc) if k != c])
        events.append(ev)
        holds.append(ho)
    MO.link_hits(events, base, [results[c]["T"] for c in range(nc)])
    for c in range(nc):
        r = results[c]
        P = base[c].copy()
        if not a.old_keys and a.smooth:   # без одиночных выбросов и шума; разметка — как есть
            P = MO.clean(P, r["cuts"], r["anchors"], a.smooth if a.smooth % 2 else a.smooth + 1)
        if a.edit:
            P = MO.edit_filter(P, r["T"], r["cuts"], r["anchors"])
        elif a.smooth and a.smooth >= 3 and n > a.smooth:
            from scipy.signal import savgol_filter
            w = a.smooth if a.smooth % 2 else a.smooth + 1
            cuts = [0] + r["cuts"] + [n]
            for s0, s1 in zip(cuts, cuts[1:]):  # не сглаживать через смену плана
                if s1 - s0 > w:
                    P[s0:s1] = savgol_filter(P[s0:s1], w, 2, axis=0, mode="interp")
            for f in r["anchors"]:
                P[f] = base[c][f]
        contacts, keys = {}, None
        if not a.no_plant:
            P, contacts = PP.plant_feet(P, r["T"], r["cuts"])
        key_frames, interp = None, None
        if a.keys > 0 and a.old_keys:
            keys = PP.keyframes(P, r["T"], r["cuts"], r["anchors"], tol_deg=a.keys)
        elif a.keys > 0:  # от позы к позе: события + разметка + промежуточные по дуге
            key_frames, interp = MO.keyframes(P, r["T"], r["cuts"], events[c], holds[c], r["anchors"],
                                              {"breakdown_deg": a.keys})
            keys = {name: key_frames for name in PP.PARTS} | {"Root": key_frames}
        fz = PP.facing(P, r["T"])   # куда смотрит (по коленям) в каждом кадре
        auto = "r" if fz.sum() >= 0 else "l"
        if c >= len(facing):
            log(f"Персонаж {c + 1}: смотрит {'вправо' if auto == 'r' else 'влево'} (по коленям; задать вручную — --facing)")
        ch = {"name": f"Char{c + 1}", "facing": facing[c] if c < len(facing) else auto,
              "facing_frames": fz.tolist(),
              "joints": JOINTS, "frames": np.round(P, 2).tolist(), "labeled": r["anchors"],
              "cuts": r["cuts"], "torso_px": np.round(r["T"], 2).tolist(),
              "contacts": {JOINTS[j]: v for j, v in contacts.items()},
              "events": [dict(x, name=MO.RU[x["type"]]) for x in events[c]]}
        if key_frames is not None:
            ch["key_frames"] = key_frames
            ch["interp"] = {str(k): v for k, v in interp.items()}
            kinds = {}
            for x in events[c]:
                kinds[MO.RU[x["type"]]] = kinds.get(MO.RU[x["type"]], 0) + 1
            log(f"Персонаж {c + 1}: события — " + (", ".join(f"{k} {v}" for k, v in kinds.items()) or "нет"))
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
                                x=np.stack([s[0] for s in samples]).astype(np.uint8),
                                kp=np.stack([s[1] for s in samples]))
    except Exception as e:
        log(f"(не удалось сохранить датасет: {e})")
    log(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
