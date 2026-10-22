"""Экзамен по анимации: насколько готовая анимация (.r6.json) похожа на работу аниматора, а не на шум.

    python tools/anim_check.py видео.r6.json [--gt эталон_gt.npy]

Меряет на той кривой, которую построит Blender (ключи + интерполяция):
  дрожание (шум углов частей поверх движения, °), перескоки рук/ног, скольжение стоп на опоре,
  ключей в секунду; с эталоном (синтетический ролик) — долю настоящих крайних поз/ударов/прыжков,
  на которые пришёлся ключ (±1 кадр).
ОЦЕНКА 0–10: 10 минус штрафы (их видно в выводе). Epigram — от 7.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stickpose"))
import motion as MO  # noqa: E402
from postprocess import J  # noqa: E402

KEY_EVENTS = ("extreme", "windup", "hit", "stop", "takeoff", "landing")


def score(m, hit_rate=None):
    pen = {}
    if m["дрожание"] > 0.5:
        pen["дрожание"] = min(3.0, (m["дрожание"] - 0.5) * 2)
    if m["перескоки"]:
        pen["перескоки"] = min(3.0, float(m["перескоки"]))
    if m["скольжение_стоп"] > 0.05:
        pen["скольжение стоп"] = 1.0 if m["скольжение_стоп"] <= 0.15 else 2.0
    if not 3 <= m["ключей_в_с"] <= 15:   # бой в Stick Nodes плотный: 3-15 ключей/с — нормально
        pen["ключей в секунду"] = 1.0
    if hit_rate is not None and hit_rate < 0.8:
        pen["ключи мимо событий"] = min(4.0, (0.8 - hit_rate) * 10)
    return max(0.0, 10 - sum(pen.values())), pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--gt", default=None, help="эталон синтетического ролика (n, персонажи, 11, 2)")
    a = ap.parse_args()
    data = json.load(open(a.json, encoding="utf-8"))
    fps = data.get("fps", 30)
    GT = np.load(a.gt) if a.gt else None
    total = []
    for ci, ch in enumerate(data["characters"]):
        P = np.array(ch["frames"], float)
        if P.shape[1] < 11:
            sys.exit("Файл без локтей/колен (старый формат) — пересоберите его stick2r6.py")
        T = np.array(ch.get("torso_px") or np.linalg.norm(P[:, J["neck"]] - P[:, J["hip"]], axis=1), float)
        keys = ch.get("key_frames") or sorted(set().union(*[set(v) for v in ch.get("keys", {}).values()])) or list(range(len(P)))
        interp = {int(k): v for k, v in ch.get("interp", {}).items()}
        contacts = {j: [tuple(s) for s in v] for j, v in ch.get("contacts", {}).items()}
        m = MO.anim_metrics(P, T, keys, interp, contacts, fps)
        hit = None
        if GT is not None and ci < GT.shape[1] and GT.shape[2] >= 11:
            G = GT[:, ci].astype(float)
            Tg = np.linalg.norm(G[:, J["neck"]] - G[:, J["hip"]], axis=1)
            ev, _ = MO.detect(G, Tg, ch.get("cuts", []))
            ev = [x for x in ev if x["type"] in KEY_EVENTS]
            ks = np.array(keys)
            hit = float(np.mean([np.min(np.abs(ks - x["f"])) <= 1 for x in ev])) if ev else 1.0
        sc, pen = score(m, hit)
        total.append(sc)
        kinds = {}
        for x in ch.get("events", []):
            kinds[x.get("name", x["type"])] = kinds.get(x.get("name", x["type"]), 0) + 1
        print(f"{ch['name']}: ОЦЕНКА АНИМАЦИИ {sc:.1f}/10")
        print(f"  ключей в секунду {m['ключей_в_с']:.1f}, дрожание {m['дрожание']:.2f}°, перескоков {m['перескоки']}, "
              f"скольжение стоп {m['скольжение_стоп']:.2f} торса" + (f", ключи на событиях {hit * 100:.0f}%" if hit is not None else ""))
        if kinds:
            print("  события: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
        if pen:
            print("  штрафы: " + ", ".join(f"{k} −{v:.1f}" for k, v in pen.items()))
    if len(total) > 1:
        print(f"\nСРЕДНЯЯ ОЦЕНКА АНИМАЦИИ {np.mean(total):.1f}/10")


if __name__ == "__main__":
    main()
