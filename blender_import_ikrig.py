"""Импорт stick2r6 (*.r6.json) на управляющий IK-риг Roblox (__PrimaryArmature: MasterControl,
Torso_FK, LeftArm-IK/RightArm-IK, LeftLeg-IK/RightLeg-IK, LookToPoint).

Как работает: ключи ставятся только на контроллеры, как это делал бы аниматор.
  - MasterControl — перемещение персонажа (таз);
  - Torso_FK      — наклон торса;
  - *Arm-IK / *Leg-IK — кисти и стопы (сгиб локтя/колена делает IK рига, pole-цели не трогаем);
  - LookToPoint   — наклон головы.
Скрипт сам вычисляет, куда поставить контроллер, чтобы итоговая точка (с учётом всех
ограничений рига) оказалась где нужно — поэтому работает и при CHILD_OF/COPY_LOCATION.

Запуск: Scripting -> Open -> этот файл -> Run Script -> выбрать .r6.json.
Персонажей несколько — выделите столько копий рига (по порядку), иначе анимация ляжет на первый найденный.
"""
import json
import math

import bpy
from mathutils import Matrix, Quaternion, Vector

JSON_PATH = ""
CHARACTER = None          # номер персонажа (1, 2...) для одиночного импорта; None — по порядку на выделенные риги
ROOT_MOTION = True
LOOK_DIST = 3.0
LEG_MODE = "plant"        # "plant" — стопы ставятся в реальные точки, колено гнётся (R6-нога по голени, стопы не скользят)
                          # "straight" — R6-нога точно по линии таз→стопа (поза как у стикмена, без сгиба)
KEYS_ONLY = True          # ключи только на ключевых кадрах из файла (между ними — гладкие кривые)

FK = {"l_hand": "LeftArm_FK", "r_hand": "RightArm_FK", "head": "Head"}
IK_OF = {"l_hand": "LeftLowerArm", "r_hand": "RightLowerArm", "l_foot": "LeftLowerLeg", "r_foot": "RightLowerLeg"}
CTRL = {"master": "MasterControl", "torso": "Torso_FK", "look": "LookToPoint",
        "l_hand": "LeftArm-IK", "r_hand": "RightArm-IK", "l_foot": "LeftLeg-IK", "r_foot": "RightLeg-IK"}
REF = {"hip": "Spine", "neck": "UpperTorso", "head": "Head",
       "l_sh": "LeftUpperArm", "r_sh": "RightUpperArm", "l_hip": "LeftUpperLeg", "r_hip": "RightUpperLeg",
       "l_arm": ("LeftUpperArm", "LeftLowerArm"), "r_arm": ("RightUpperArm", "RightLowerArm"),
       "l_leg": ("LeftUpperLeg", "LeftLowerLeg"), "r_leg": ("RightUpperLeg", "RightLowerLeg")}


def find_rig(objs):
    for o in objs:
        if o.type == "ARMATURE" and all(n in o.pose.bones for n in CTRL.values()):
            return o
    return None


def ik_on(obj, key):
    """Включён ли IK у конечности в этом риге (влияние IK-ограничения)."""
    pb = obj.pose.bones.get(IK_OF[key])
    return bool(pb) and any(c.type == "IK" and not c.mute and c.influence > 0.5 for c in pb.constraints)


def look_on(obj):
    pb = obj.pose.bones.get(FK["head"])
    return bool(pb) and any(c.type in ("TRACK_TO", "DAMPED_TRACK") and not c.mute and c.influence > 0.5
                            for c in pb.constraints)


def aim_fk(obj, name, d, axis):
    """Повернуть FK-кость (без ограничений) так, чтобы она смотрела вдоль d (пространство арматуры).
    Поворот только в плоскости анимации — вокруг axis."""
    pb = obj.pose.bones[name]
    set_rotation(pb, Quaternion())
    update()
    M = ev(obj).pose.bones[name].matrix.copy()
    c = (M.col[1].xyz).normalized()  # ось Y кости = её направление
    cp = c - axis * c.dot(axis)
    dp = d - axis * d.dot(axis)
    if cp.length < 1e-6 or dp.length < 1e-6:
        return
    ang = math.atan2(axis.dot(cp.cross(dp)), cp.dot(dp))
    R = Quaternion(axis, ang).to_matrix().to_4x4()
    t = M.translation.copy()
    N = R @ Matrix.Translation(-t) @ M
    N.translation = t
    pb.matrix = N
    if pb.rotation_mode == "QUATERNION":
        prev = _PREV.get(name)
        q = pb.rotation_quaternion.copy()
        if prev is not None and q.dot(prev) < 0:
            q.negate()
            pb.rotation_quaternion = q
        _PREV[name] = q


_PREV = {}


def update():
    bpy.context.view_layer.update()


def ev(obj):
    return obj.evaluated_get(bpy.context.evaluated_depsgraph_get())


def solve_locations(obj, wanted, probe):
    """wanted: {кость-контроллер: Vector цели (пространство арматуры)};
    probe(eval_obj, кость) -> точка, которая должна совпасть с целью.
    Связь «location контроллера -> точка» аффинная (ограничения рига линейны по смещению),
    поэтому хватает 4 пересчётов: ноль и три единичных сдвига."""
    names = list(wanted)
    pbs = [obj.pose.bones[n] for n in names]
    for pb in pbs:
        pb.location = (0, 0, 0)
    update()
    e = ev(obj)
    h0 = {n: probe(e, n).copy() for n in names}
    cols = {n: [] for n in names}
    for k in range(3):
        for pb in pbs:
            v = Vector((0, 0, 0))
            v[k] = 1.0
            pb.location = v
        update()
        e = ev(obj)
        for n in names:
            cols[n].append(probe(e, n) - h0[n])
    for n, pb in zip(names, pbs):
        A = Matrix((cols[n][0], cols[n][1], cols[n][2])).transposed()
        try:
            pb.location = A.inverted() @ (wanted[n] - h0[n])
        except ValueError:  # контроллер выключен ограничением — оставляем как есть
            pb.location = (0, 0, 0)
    update()


def set_rotation(pb, q):
    if pb.rotation_mode == "QUATERNION":
        if pb.rotation_quaternion.dot(q) < 0:
            q = -q
        pb.rotation_quaternion = q
    elif pb.rotation_mode == "AXIS_ANGLE":
        ax, ang = q.to_axis_angle()
        pb.rotation_axis_angle = (ang, *ax)
    else:
        pb.rotation_euler = q.to_euler(pb.rotation_mode, pb.rotation_euler)


def rot_path(pb):
    return {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(pb.rotation_mode, "rotation_euler")


def apply_character(obj, ch, data, frame_start=1):
    J = {j: i for i, j in enumerate(ch["joints"])}
    F = ch["frames"]
    n = len(F)
    side = data.get("view", "side")
    sgn = 1.0 if ch.get("facing", "r") == "r" else -1.0
    arm = obj.data
    B = arm.bones

    def to3d(v, s=1.0):
        # экран: x вправо, y вниз -> риг: вперёд +Y (вид сбоку) или вправо +X (вид спереди), вверх +Z
        if side == "front":
            return Vector((-v[0] * s, 0.0, -v[1] * s))
        return Vector((0.0, sgn * v[0] * s, -v[1] * s))

    # опорные точки рига в покое
    hip_r = B[REF["hip"]].head_local.copy()
    neck_r = B[REF["neck"]].tail_local.copy()
    L_arm = sum(B[b].length for b in REF["l_arm"])
    L_leg = sum(B[b].length for b in REF["l_leg"])
    T_px = ch.get("torso_px") or [math.dist(F[f][J["neck"]], F[f][J["hip"]]) for f in range(n)]
    T_px = [max(1e-3, t) for t in T_px]

    def norm_len(a, b):
        return [math.dist(F[f][J[a]], F[f][J[b]]) / T_px[f] for f in range(n)]

    def p95(x):
        s = sorted(x)
        return max(1e-3, s[int(0.95 * (len(s) - 1))])

    ref = {"arm": p95(norm_len("neck", "l_hand") + norm_len("neck", "r_hand")),
           "leg": p95(norm_len("hip", "l_foot") + norm_len("hip", "r_foot"))}
    # масштаб корня — по ногам, чтобы стоящая на земле стопа не скользила, пока таз движется
    k_root = L_leg / ref["leg"]

    cuts = set(ch.get("cuts", []))
    root = [Vector((0, 0, 0))]
    for f in range(1, n):
        if f in cuts or not ROOT_MOTION:
            root.append(root[-1].copy())
            continue
        d = (F[f][J["hip"]][0] - F[f - 1][J["hip"]][0], F[f][J["hip"]][1] - F[f - 1][J["hip"]][1])
        root.append(root[-1] + to3d(d, k_root / T_px[f]))

    if KEYS_ONLY and ch.get("keys"):
        frames = sorted(set().union(*[set(v) for v in ch["keys"].values()]) | {0, n - 1})
    else:
        frames = list(range(n))

    obj.animation_data_create()
    if obj.animation_data.action is None:
        obj.animation_data.action = bpy.data.actions.new(f"{obj.name}_stick2r6")
    pbs = {k: obj.pose.bones[v] for k, v in CTRL.items()}
    mode = {k: ("ik" if ik_on(obj, k) else "fk") for k in IK_OF}
    mode["head"] = "look" if look_on(obj) else "fk"
    for k in ("l_foot", "r_foot"):
        if mode[k] == "fk":
            mode[k] = "ik"  # ноги без IK в этом риге не предусмотрены — ставим цель, риг сам решит
    print("stick2r6 IK: режимы", mode)
    for k in ("l_hand", "r_hand", "head"):
        set_rotation(obj.pose.bones[FK[k]], Quaternion())
    _PREV.clear()
    for pb in pbs.values():
        pb.location = (0, 0, 0)
        set_rotation(pb, Quaternion())
    tname = CTRL["torso"]
    low = obj.pose.bones.get("LowerTorso-FK")
    if low is not None and not any(c.type == "CHILD_OF" and c.subtarget == CTRL["torso"] and c.influence > 0.5
                                   for c in low.constraints):
        tname = "LowerTorso-FK"   # в этом риге наклон торса задаёт нижний торс (поворот вокруг таза)
    torso_pb = obj.pose.bones[tname]
    set_rotation(torso_pb, Quaternion())
    t_rest = B[tname].matrix_local.to_quaternion()
    print("stick2r6 IK: торс через", tname)
    up = Vector((0, 0, 1))
    axis = to3d((1.0, 0.0)).cross(up).normalized() if side == "side" else Vector((0, 1, 0))
    look_rot = Quaternion(axis, -math.pi / 2)  # «вверх» головы -> направление взгляда

    def limb_target(root_pt, a, b, L, r):
        v = (F[f][J[b]][0] - F[f][J[a]][0], F[f][J[b]][1] - F[f][J[a]][1])
        d = to3d(v)
        if d.length < 1e-6:
            d = Vector((0, 0, -1))
        k = min(0.999, max(0.3, (math.hypot(*v) / T_px[f]) / r))
        if LEG_MODE == "straight" and b.endswith("foot"):
            k = 0.999
        return root_pt + d.normalized() * (L * k)

    for f in frames:
        # 1) наклон торса: поворот в плоскости вокруг нормали
        dt = to3d((F[f][J["neck"]][0] - F[f][J["hip"]][0], F[f][J["neck"]][1] - F[f][J["hip"]][1])).normalized()
        ang = math.atan2(axis.dot(up.cross(dt)), up.dot(dt))
        R = Quaternion(axis, ang)
        set_rotation(torso_pb, t_rest.inverted() @ R @ t_rest)
        # 2) MasterControl так, чтобы таз оказался в нужной точке
        solve_locations(obj, {CTRL["master"]: hip_r + root[f]}, lambda e, nm: e.pose.bones[REF["hip"]].head)
        e = ev(obj)
        sh = {s: e.pose.bones[REF[s + "_sh"]].head.copy() for s in ("l", "r")}
        hp = {s: e.pose.bones[REF[s + "_hip"]].head.copy() for s in ("l", "r")}
        head_pt = e.pose.bones[REF["head"]].head.copy()
        # 3) IK-цели кистей/стоп и точка взгляда
        uh = to3d((F[f][J["head"]][0] - F[f][J["neck"]][0], F[f][J["head"]][1] - F[f][J["neck"]][1]))
        uh = uh.normalized() if uh.length > 1e-6 else dt
        wanted = {}
        for key, root_pt, a, L, r in (("l_hand", sh["l"], "neck", L_arm, ref["arm"]),
                                      ("r_hand", sh["r"], "neck", L_arm, ref["arm"]),
                                      ("l_foot", hp["l"], "hip", L_leg, ref["leg"]),
                                      ("r_foot", hp["r"], "hip", L_leg, ref["leg"])):
            if mode[key] == "ik":
                wanted[CTRL[key]] = limb_target(root_pt, a, key, L, r)
        if mode["head"] == "look":
            wanted[CTRL["look"]] = head_pt + (look_rot @ uh) * LOOK_DIST
        if wanted:
            solve_locations(obj, wanted, lambda e, nm: e.pose.bones[nm].head)
        for key in ("l_hand", "r_hand"):
            if mode[key] == "fk":
                v = (F[f][J[key]][0] - F[f][J["neck"]][0], F[f][J[key]][1] - F[f][J["neck"]][1])
                aim_fk(obj, FK[key], to3d(v).normalized() if math.hypot(*v) > 1e-6 else Vector((0, 0, -1)), axis)
        if mode["head"] == "fk":
            aim_fk(obj, FK["head"], uh, axis)
        fr = frame_start + f
        pbs["master"].keyframe_insert("location", frame=fr)
        torso_pb.keyframe_insert(rot_path(torso_pb), frame=fr)
        for key in ("l_hand", "r_hand", "l_foot", "r_foot"):
            if mode[key] == "ik":
                pbs[key].keyframe_insert("location", frame=fr)
            else:
                fb = obj.pose.bones[FK[key]]
                fb.keyframe_insert(rot_path(fb), frame=fr)
        if mode["head"] == "look":
            pbs["look"].keyframe_insert("location", frame=fr)
        else:
            hb = obj.pose.bones[FK["head"]]
            hb.keyframe_insert(rot_path(hb), frame=fr)
    return frames


def run(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") != "stick2r6":
        raise RuntimeError("Это не файл stick2r6 (.r6.json)")
    scn = bpy.context.scene
    fps = data.get("fps", 30)
    scn.render.fps = max(1, int(round(fps)))
    scn.render.fps_base = scn.render.fps / fps
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    rigs = [o for o in bpy.context.selected_objects if find_rig([o])]
    if not rigs:
        r = find_rig(bpy.data.objects)
        if r is None:
            raise RuntimeError("Не найден IK-риг (нужны кости MasterControl, Torso_FK, *-IK, LookToPoint)")
        rigs = [r]
    chars = data["characters"]
    if CHARACTER:
        chars = [chars[CHARACTER - 1]]
    done = []
    for rig, ch in zip(rigs, chars):
        apply_character(rig, ch, data)
        done.append(f"{ch['name']} -> {rig.name}")
    scn.frame_start = 1
    scn.frame_end = max(len(c["frames"]) for c in chars)
    scn.frame_set(1)
    print("stick2r6 IK:", done)
    return done


class STICK2R6_OT_import_ik(bpy.types.Operator):
    """Импорт stick2r6 на IK-риг Roblox"""
    bl_idname = "import_anim.stick2r6_ik"
    bl_label = "Импорт stick2r6 на IK-риг (.r6.json)"
    bl_options = {"REGISTER", "UNDO"}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            done = run(self.filepath)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"stick2r6: {type(e).__name__}: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, "stick2r6: " + ", ".join(done))
        return {"FINISHED"}


def main():
    try:
        bpy.utils.unregister_class(STICK2R6_OT_import_ik)
    except Exception:
        pass
    bpy.utils.register_class(STICK2R6_OT_import_ik)
    import os
    if JSON_PATH and os.path.exists(JSON_PATH):
        return run(JSON_PATH)
    bpy.ops.import_anim.stick2r6_ik("INVOKE_DEFAULT")


if __name__ == "__main__":
    main()
