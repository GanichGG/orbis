"""Скачивает движения людей из базы CMU Motion Capture (BVH-копия на GitHub, свободное использование)
и сохраняет weights/mocap.npz — позы и движения для генератора синтетики (synth.py).

    python tools/fetch_mocap.py                 # ~600 записей: бои, удары, прыжки, кувырки, танцы, бег...
    python tools/fetch_mocap.py --max 50        # быстро, для проверки

Скачивает ~1,4 ГБ BVH (во временную папку weights/mocap_bvh/, после удаляется; --keep_bvh — оставить).
Результат ~45 МБ. Нужен только numpy.
"""
import argparse
import os
import re
import sys
import urllib.request

import numpy as np

BASE = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "weights", "mocap.npz")
CACHE = os.path.join(ROOT, "weights", "mocap_bvh")
FPS = 30
# что берём: всё «активное»; обычной ходьбы — немного
KEEP = re.compile(r"fight|punch|kick|box|martial|karate|kung|taekwondo|judo|sword|jump|hop|leap|flip|cartwheel|"
                  r"somersault|roll|acrobat|dance|break|run|jog|sprint|climb|throw|swing|stretch|yoga|gymnast|attack|"
                  r"wrestl|duck|dodge|fall|crawl|push|pull|lift|spin|twist|squat|lunge|kneel|reach|bend|stumble|"
                  r"wash|sweep|shoot|basketball|soccer|golf|baseball|tennis|motorcycle|mop|chicken|monkey|bear|"
                  r"animal|elephant|frog|march|skip|zombie|drunk|sneak|tiptoe|range of motion|salsa|charleston|"
                  r"indian|walk", re.I)
# точки для стикмена: 11 суставов Prismium + грудь (изгиб торса) + плечевые и тазобедренные суставы (направления звеньев)
MJ = ["head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot", "l_elbow", "r_elbow", "l_knee", "r_knee",
      "chest", "l_sh", "r_sh", "l_hipj", "r_hipj"]
SRC = {"neck": "Neck", "hip": "Hips", "l_hand": "LeftHand", "r_hand": "RightHand", "l_foot": "LeftFoot",
       "r_foot": "RightFoot", "l_elbow": "LeftForeArm", "r_elbow": "RightForeArm", "l_knee": "LeftLeg",
       "r_knee": "RightLeg", "chest": "Spine1", "l_sh": "LeftArm", "r_sh": "RightArm", "l_hipj": "LeftUpLeg",
       "r_hipj": "RightUpLeg"}


def parse_bvh(text):
    """-> joints [(имя, родитель, offset, каналы)], end_sites {родитель: offset}, motion (F, C), frame_time."""
    lines = text.split("\n")
    joints, ends, stack, i = [], {}, [], 0
    cur = None
    while i < len(lines):
        t = lines[i].split()
        i += 1
        if not t:
            continue
        if t[0] in ("ROOT", "JOINT"):
            cur = len(joints)
            joints.append([t[1], stack[-1] if stack else -1, None, []])
        elif t[0] == "End":
            cur = ("end", stack[-1])
        elif t[0] == "{":
            stack.append(cur)
        elif t[0] == "}":
            stack.pop()
        elif t[0] == "OFFSET":
            off = np.array(list(map(float, t[1:4])))
            if isinstance(cur, tuple):
                ends[cur[1]] = off
            else:
                joints[cur][2] = off
        elif t[0] == "CHANNELS":
            joints[cur][3] = t[2:]
        elif t[0] == "MOTION":
            break
    nf = int(lines[i].split()[1])
    ft = float(lines[i + 1].split()[2])
    data = np.array([list(map(float, ln.split())) for ln in lines[i + 2:i + 2 + nf] if ln.strip()])
    return joints, ends, data, ft


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    k = 3 - i - j
    R[..., k, k] = 1
    R[..., i, i] = c
    R[..., j, j] = c
    R[..., i, j] = -s
    R[..., j, i] = s
    return R


def forward(joints, ends, data, step):
    """Мировые позиции всех суставов (и концов головы) для кадров data[::step]. -> {имя: (F,3)}."""
    data = data[::step]
    F = len(data)
    pos, rot, col = {}, {}, 0
    for k, (name, par, off, ch) in enumerate(joints):
        R = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
        t = np.zeros((F, 3))
        for c in ch:
            v = data[:, col]
            col += 1
            if c.endswith("position"):
                t[:, "XYZ".index(c[0])] = v
            else:
                R = R @ _rot(c[0], v)
        if par < 0:
            pos[k] = t + off
            rot[k] = R
        else:
            pos[k] = pos[par] + np.einsum("fij,j->fi", rot[par], off)
            rot[k] = rot[par] @ R
    out = {joints[k][0]: pos[k] for k in pos}
    hk = next(k for k, j in enumerate(joints) if j[0] == "Head")
    out["HeadTop"] = pos[hk] + np.einsum("fij,j->fi", rot[hk], ends[hk])
    return out


def to_points(P):
    """Суставы BVH (Y вверх) -> MJ (F,16,3) в системе x вправо, y вперёд, z вверх."""
    g = {j: P[s] for j, s in SRC.items()}
    g["head"] = (P["Head"] + P["HeadTop"]) / 2
    # у стикмена руки растут из «шеи», ноги — из «таза»: середина плечевых / тазобедренных суставов
    g["neck"] = (P["LeftArm"] + P["RightArm"]) / 2
    g["hip"] = (P["LeftUpLeg"] + P["RightUpLeg"]) / 2
    A = np.stack([g[j] for j in MJ], 1)
    return A[..., [0, 2, 1]] * np.array([1, -1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--keep_bvh", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    idx = urllib.request.urlopen(BASE + "/cmu-mocap-index-text.txt", timeout=60).read().decode("utf-8", "ignore")
    trials = []
    for ln in idx.splitlines():
        m = re.match(r"^(\d+)_(\d+)\s+(.*)$", ln.strip())
        if m and KEEP.search(m.group(3)):
            trials.append((m.group(1), m.group(2), m.group(3).strip()))
    walks = [t for t in trials if re.fullmatch(r"walk.*", t[2], re.I) and not re.search(r"run|jump|dance", t[2], re.I)]
    trials = [t for t in trials if t not in walks] + walks[::6]   # каждая 6-я «просто ходьба»
    if a.max:
        trials = trials[::max(1, len(trials) // a.max)][:a.max]
    print("записей:", len(trials))
    seqs, names = [], []
    for i, (s, n, desc) in enumerate(trials):
        fn = f"{s}_{n}.bvh"
        path = os.path.join(CACHE, fn)
        try:
            if not os.path.exists(path):
                url = f"{BASE}/data/{int(s):03d}/{fn}"
                urllib.request.urlretrieve(url, path + ".part")
                os.replace(path + ".part", path)
            joints, ends, data, ft = parse_bvh(open(path, encoding="utf-8", errors="ignore").read())
            step = max(1, int(round(1.0 / (ft * FPS))))
            A = to_points(forward(joints, ends, data, step)).astype(np.float32)
            if len(A) >= 8:
                seqs.append(A)
                names.append(f"{s}_{n} {desc}")
        except Exception as e:  # битый файл / нет в копии — пропускаем
            print(f"  пропуск {fn}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(trials)}, кадров {sum(len(q) for q in seqs)}", flush=True)
    lens = np.array([len(q) for q in seqs])
    A = np.concatenate(seqs)
    root = A[:, MJ.index("hip")].copy()
    # позы — относительно таза (float16 хватает), путь таза — отдельно (float32)
    np.savez_compressed(a.out, pos=(A - root[:, None]).astype(np.float16), root=root, start=np.r_[0, np.cumsum(lens)[:-1]],
                        length=lens, names=np.array(names), joints=np.array(MJ), fps=FPS)
    if not a.keep_bvh:
        import shutil
        shutil.rmtree(CACHE, ignore_errors=True)
    print(f"готово: {a.out} — {len(seqs)} записей, {lens.sum()} кадров ({lens.sum() / FPS / 60:.0f} мин движения)")


if __name__ == "__main__":
    sys.exit(main())
