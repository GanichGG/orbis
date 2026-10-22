"""Анимации Shadow Fight 2 (ассеты игры, Gymnast Tool Suite) -> weights/sf2.npz для генератора синтетики.

    python tools/fetch_sf2.py "путь/к/архиву.zip"      (или к распакованной папке «Shadow Fight 2 Assets»)

Формат .bin анимации: int32 число кадров; на кадр — 1 байт, int32 число узлов (67), узлы × 3 float32 (X, Y вверх, Z).
Порядок узлов — как в Models/Dependencies/skeleton.xml (Node, MacroNode, CenterOfMass). Берём только анимации
со скелетом бойца (67 узлов) и пишем в формате weights/mocap.npz (те же 16 точек, Z вверх), synth берёт оба файла.

Ассеты принадлежат Nekki: файл sf2.npz только для обучения у себя, в git и релизы не кладём (.gitignore).
"""
import os
import re
import struct
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "weights", "sf2.npz")
MJ = ["head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot", "l_elbow", "r_elbow", "l_knee", "r_knee",
      "chest", "l_sh", "r_sh", "l_hipj", "r_hipj"]
SRC = {"neck": ["NNeck"], "head": ["NHead", "NTop"], "hip": ["NHip_1", "NHip_2"], "l_hand": ["NWrist_1"],
       "r_hand": ["NWrist_2"], "l_foot": ["NAnkle_1"], "r_foot": ["NAnkle_2"], "l_elbow": ["NElbow_1"],
       "r_elbow": ["NElbow_2"], "l_knee": ["NKnee_1"], "r_knee": ["NKnee_2"], "chest": ["NChest"],
       "l_sh": ["NShoulder_1"], "r_sh": ["NShoulder_2"], "l_hipj": ["NHip_1"], "r_hipj": ["NHip_2"]}
FPS = 30


class Assets:
    """Чтение из zip или из папки — одинаково."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
        self.root = path
        names = self.zip.namelist() if self.zip else [os.path.relpath(os.path.join(d, f), path).replace("\\", "/")
                                                      for d, _, fs in os.walk(path) for f in fs]
        self.names = names

    def find(self, suffix):
        return [n for n in self.names if n.replace("\\", "/").lower().endswith(suffix.lower())]

    def read(self, name):
        return self.zip.read(name) if self.zip else open(os.path.join(self.root, name), "rb").read()


def skeleton_order(assets):
    p = assets.find("Models/Dependencies/skeleton.xml") or assets.find("/skeleton.xml") or assets.find("skeleton.xml")
    if not p:
        sys.exit("Не нашёл Models/Dependencies/skeleton.xml. Укажите архив .zip целиком или папку «Shadow Fight 2 Assets»\n"
                 "(в ней должны быть папки Animations и Models), а не подпапку.")
    s = assets.read(p[0]).decode("cp1251", "ignore")
    nodes = s.split("<Nodes>")[1].split("</Nodes>")[0]
    return [n for n, _ in re.findall(r'<(\S+) Type="(\w+)"', nodes)]


def load_anim(b):
    nf = struct.unpack("<i", b[:4])[0]
    o, F = 4, []
    for _ in range(nf):
        o += 1
        nn = struct.unpack("<i", b[o:o + 4])[0]
        o += 4
        F.append(np.frombuffer(b[o:o + 12 * nn], "<f4").reshape(nn, 3))
        o += 12 * nn
    return F


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1].strip('"')
    if not os.path.exists(path):
        sys.exit(f"Нет такого пути: {path}")
    if os.path.isdir(path) and os.path.basename(os.path.normpath(path)).lower() in ("animations", "models"):
        path = os.path.dirname(os.path.normpath(path))   # указали подпапку — берём папку ассетов
    assets = Assets(path)
    order = skeleton_order(assets)
    idx = {n: i for i, n in enumerate(order)}
    seqs, names = [], []
    for name in sorted(n for n in assets.names if "/Animations/" in n.replace("\\", "/") and n.endswith(".bin")):
        try:
            F = load_anim(assets.read(name))
        except Exception:
            continue
        if len(F) < 8 or any(len(x) != len(order) for x in F):
            continue  # не боец (снаряды, эффекты) или другой скелет
        A = np.stack(F).astype(np.float64)
        g = {j: A[:, [idx[s] for s in src]].mean(1) for j, src in SRC.items()}
        P = np.stack([g[j] for j in MJ], 1)               # X, Y(вверх), Z
        P = np.stack([P[..., 2], P[..., 0], P[..., 1]], -1)   # -> наши: x = глубина, y = вперёд (экран SF2), z = вверх
        torso = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
        if not np.isfinite(P).all() or torso.min() < 1e-3:
            continue
        seqs.append(P.astype(np.float32))
        names.append(os.path.basename(name)[:-4])
    if not seqs:
        sys.exit("Не нашёл анимаций бойца — проверьте путь к ассетам.")
    lens = np.array([len(q) for q in seqs])
    A = np.concatenate(seqs)
    root = A[:, MJ.index("hip")].copy()
    np.savez_compressed(OUT, pos=(A - root[:, None]).astype(np.float16), root=root, start=np.r_[0, np.cumsum(lens)[:-1]],
                        length=lens, names=np.array(names), joints=np.array(MJ), fps=FPS)
    print(f"готово: {OUT} — {len(seqs)} анимаций, {lens.sum()} кадров")


if __name__ == "__main__":
    main()
