"""Проверка импорта на риг пользователя (без окна Blender, нужен `pip install bpy`):
    python tools/test_ik.py basefile.blend [файл.r6.json]
Без json — делает синтетического персонажа (11 точек, плавная анимация, иногда «дикие» позы).
Меряет, насколько части R6 (InternalArmature) смотрят туда же, куда части стикмена по правилу переноса:
торс таз->шея, голова шея->голова, рука локоть->кисть, нога колено->стопа."""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "stickpose"))


def make_json(path, n=120, seed=3):
    import synth
    sys.path.insert(0, os.path.join(ROOT, "stickpose"))
    from make_testvideo import interp, ease
    rng = np.random.default_rng(seed)
    keys = [synth.sample_pose(rng) for _ in range(n // 10 + 2)]
    frames = []
    for f in range(n):
        seg, t = divmod(f, 10)
        p = interp(keys[seg], keys[seg + 1], ease(t / 10))
        pts, segs = synth.forward(p, np.array([480.0 + f, 300.0]), 70.0)
        head = pts["neck"] + synth._dir(pts["head_dir"]) * 20
        frames.append(synth._keypoints(pts, head).round(2).tolist())
    data = {"format": "stick2r6", "version": 2, "fps": 30, "view": "side", "start": 0, "frame_size": [960, 540],
            "characters": [{"name": "Char1", "facing": "r", "frames": frames, "cuts": [], "labeled": [],
                            "joints": ["head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot",
                                       "l_elbow", "r_elbow", "l_knee", "r_knee"]}]}
    json.dump(data, open(path, "w"))


blend = sys.argv[1]
jpath = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(blend)), "_ik_test.r6.json")
if len(sys.argv) <= 2:
    make_json(jpath)
bpy.ops.wm.open_mainfile(filepath=blend)
P = bpy.data.objects["__PrimaryArmature"]
I = bpy.data.objects["InternalArmature"]
g = {"__name__": "x"}
exec(compile(open(os.path.join(ROOT, "blender_import_ikrig.py"), encoding="utf-8").read(), "ik", "exec"), g)
t0 = time.time()
g["run"](jpath)
print("время импорта", round(time.time() - t0, 1), "с")
data = json.load(open(jpath))
ch = data["characters"][0]
F = np.array(ch["frames"])
Jn = {j: i for i, j in enumerate(ch["joints"])}
rule = "l_elbow" in Jn
segs = {"Torso": ("hip", "neck", 1), "Head": ("neck", "head", 1),
        "Left Arm": ("l_elbow" if rule else "neck", "l_hand", -1), "Right Arm": ("r_elbow" if rule else "neck", "r_hand", -1),
        "Left Leg": ("l_knee" if rule else "hip", "l_foot", -1), "Right Leg": ("r_knee" if rule else "hip", "r_foot", -1)}
W = P.matrix_world.to_3x3()
sg = 1.0 if ch.get("facing", "r") == "r" else -1.0
err = {k: [] for k in segs}
dist = {}
for f in range(len(F)):
    bpy.context.scene.frame_set(f + 1)
    e = I.evaluated_get(bpy.context.evaluated_depsgraph_get())
    Iw = I.matrix_world.to_3x3()
    for part, (a, b, s) in segs.items():
        pb, bone = e.pose.bones[part], I.data.bones[part]
        M = Iw @ (pb.matrix @ bone.matrix_local.inverted()).to_3x3() @ Iw.inverted()
        d = (M @ Vector((0, 0, s))).normalized()
        v = F[f, Jn[b]] - F[f, Jn[a]]
        gt = (W @ Vector((0, sg * v[0], -v[1]))).normalized()
        err[part].append(math.degrees(math.acos(max(-1, min(1, d.dot(gt))))))
    # конец блока R6 (низ руки/ноги) против кисти/стопы стикмена — в шпильках, в плоскости вида сбоку
    tb = I.data.bones["Torso"]
    hipw = I.matrix_world @ (e.pose.bones["Torso"].matrix @ tb.matrix_local.inverted()
                             @ (tb.head_local + I.matrix_world.inverted().to_3x3() @ Vector((0, 0, -1))))
    sc = 2.0 / np.linalg.norm(F[f, Jn["neck"]] - F[f, Jn["hip"]])
    for part, j, off in (("Left Arm", "l_hand", (-0.5, 0, -2.5)), ("Right Arm", "r_hand", (0.5, 0, -2.5)),
                         ("Left Leg", "l_foot", (0.5, 0, -2.0)), ("Right Leg", "r_foot", (-0.5, 0, -2.0))):
        pb, bone = e.pose.bones[part], I.data.bones[part]
        endw = I.matrix_world @ (pb.matrix @ bone.matrix_local.inverted() @ (bone.head_local + Vector(off)))
        v = (F[f, Jn[j]] - F[f, Jn["hip"]]) * sc
        tw = hipw + W @ Vector((0, sg * v[0], -v[1]))
        dist.setdefault(part, []).append(math.hypot(endw.y - tw.y, endw.z - tw.z))
for k, v in dist.items():
    print(f"{k:10s} конец блока от кисти/стопы: ср {np.mean(v):.2f}  макс {np.max(v):.2f} шп.")
for k, v in err.items():
    v = np.array(v)
    print(f"{k:10s} ср {v.mean():5.1f}°  p95 {np.percentile(v, 95):5.1f}°  макс {v.max():5.1f}°")
