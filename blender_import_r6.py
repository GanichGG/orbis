"""Импорт анимации stick2r6 (*.r6.json) на R6-риг в Blender (4.x / 5.x).

Как пользоваться:
  1. Blender -> вкладка Scripting -> Open -> этот файл.
  2. (не обязательно) впишите путь к .r6.json в JSON_PATH — иначе при запуске откроется окно выбора файла.
  3. Если у вас уже есть R6-риг (арматура с костями Torso, Head, Left Arm, Right Arm,
     Left Leg, Right Leg — имена можно с пробелами/без, регистр не важен) — выделите его.
     Иначе скрипт сам создаст R6-риг с кубиками для каждого персонажа.
  4. Run Script.
Каждый персонаж получает свой Action с ключами на каждом кадре.
"""
import json
import math
import re

import bpy
from mathutils import Matrix, Quaternion, Vector

JSON_PATH = ""          # можно оставить пустым — откроется окно выбора файла
CHARACTERS = None        # None — все; или список номеров, например [1]
SPACING = 6.0            # расстояние между созданными ригами по X
ROOT_MOTION = True       # двигать персонажа по сцене (иначе — на месте). Движение камеры в видео тоже попадёт сюда
KEYS_ONLY = True         # ставить ключи только на ключевых позах (иначе — на каждом кадре)
KEEP_HEIGHT = True       # вертикальное движение таза (прыжки, приседания)

PARTS = ["Torso", "Head", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]
# какой отрезок 2D-скелета задаёт направление части и куда часть смотрит в покое (вверх/вниз)
SEG = {
    "Torso": ("hip", "neck", +1),
    "Head": ("neck", "head", +1),
    "Left Arm": ("neck", "l_hand", -1),
    "Right Arm": ("neck", "r_hand", -1),
    "Left Leg": ("hip", "l_foot", -1),
    "Right Leg": ("hip", "r_foot", -1),
}


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def find_bones(arm_obj):
    names = {_norm(b.name): b.name for b in arm_obj.data.bones}
    alias = {
        "Torso": ["torso", "uppertorso", "body"],
        "Head": ["head"],
        "Left Arm": ["leftarm", "larm", "armleft", "arml"],
        "Right Arm": ["rightarm", "rarm", "armright", "armr"],
        "Left Leg": ["leftleg", "lleg", "legleft", "legl"],
        "Right Leg": ["rightleg", "rleg", "legright", "legr"],
    }
    out = {}
    for p, keys in alias.items():
        for k in keys:
            if k in names:
                out[p] = names[k]
                break
    missing = [p for p in PARTS if p not in out]
    if missing:
        raise RuntimeError(f"В арматуре {arm_obj.name} не найдены кости: {missing}")
    return out


def build_r6(name, x_offset):
    """R6 в студах: ноги 2, торс 2, руки 2, голова 1. Смотрит в -Y, Z вверх."""
    arm = bpy.data.armatures.new(name + "_Rig")
    obj = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (x_offset, 0, 0)
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    spec = {  # голова кости = точка вращения (сустав), хвост — вдоль части
        "Torso": ((0, 0, 2), (0, 0, 4), None),
        "Head": ((0, 0, 4), (0, 0, 5), "Torso"),
        "Left Arm": ((1.5, 0, 3.5), (1.5, 0, 1.5), "Torso"),
        "Right Arm": ((-1.5, 0, 3.5), (-1.5, 0, 1.5), "Torso"),
        "Left Leg": ((0.5, 0, 2), (0.5, 0, 0), "Torso"),
        "Right Leg": ((-0.5, 0, 2), (-0.5, 0, 0), "Torso"),
    }
    for bn, (h, t, par) in spec.items():
        b = eb.new(bn)
        b.head, b.tail = Vector(h), Vector(t)
        b.roll = 0
        if par:
            b.parent = eb[par]
    bpy.ops.object.mode_set(mode="OBJECT")
    # кубики для предпросмотра
    boxes = {"Torso": ((0, 0, 3), (2, 1, 2)), "Head": ((0, 0, 4.6), (1.2, 1.2, 1.2)),
             "Left Arm": ((1.5, 0, 3), (1, 1, 2)), "Right Arm": ((-1.5, 0, 3), (1, 1, 2)),
             "Left Leg": ((0.5, 0, 1), (1, 1, 2)), "Right Leg": ((-0.5, 0, 1), (1, 1, 2))}
    for bn, (c, s) in boxes.items():
        me = bpy.data.meshes.new(f"{name}_{bn}")
        hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
        v = [(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        me.from_pydata(v, [], f)
        mo = bpy.data.objects.new(f"{name}_{bn}", me)
        bpy.context.scene.collection.objects.link(mo)
        mo.parent = obj
        mo.parent_type = "BONE"
        mo.parent_bone = bn
        bone = arm.bones[bn]
        # родитель-кость крепит к хвосту кости; компенсируем, чтобы кубик стоял на месте части
        mo.matrix_parent_inverse = (obj.matrix_world @ bone.matrix_local @
                                    Matrix.Translation((0, bone.length, 0))).inverted()
        mo.matrix_basis = Matrix.Translation(obj.matrix_world @ Vector(c))
    return obj


def plane_axes(view, facing):
    """Экран (x вправо, y вниз) -> арматура. Возвращает (u — экранное «вправо», нормаль плоскости)."""
    if view == "front":
        u = Vector((1, 0, 0))
    else:
        u = Vector((0, -1, 0)) if facing == "r" else Vector((0, 1, 0))
    z = Vector((0, 0, 1))
    n = u.cross(z).normalized()
    return u, n


def to3d(v2, u):
    return u * v2[0] + Vector((0, 0, -v2[1]))


def signed_angle(a, b, n):
    return math.atan2(n.dot(a.cross(b)), a.dot(b))


def apply_character(obj, ch, data, frame_start=1):
    bones = find_bones(obj)
    J = {j: i for i, j in enumerate(ch["joints"])}
    F = ch["frames"]
    nfr = len(F)
    u, nrm = plane_axes(data.get("view", "side"), ch.get("facing", "r"))
    arm = obj.data
    rest = {p: arm.bones[bones[p]].matrix_local.copy() for p in PARTS}
    parent_part = {}
    for p in PARTS:
        par = arm.bones[bones[p]].parent
        parent_part[p] = next((q for q in PARTS if par is not None and bones[q] == par.name), None)

    # масштаб: торс R6 (от таза до шеи) в единицах рига / в пикселях
    torso_rig = (arm.bones[bones["Head"]].head_local - (arm.bones[bones["Left Leg"]].head_local +
                                                       arm.bones[bones["Right Leg"]].head_local) / 2).length
    # пикселей на длину торса в каждом кадре (камера может приближаться/отдаляться)
    if "torso_px" in ch:
        tpx = [max(1e-3, t) for t in ch["torso_px"]]
    else:
        tl = sorted(math.dist(F[f][J["neck"]], F[f][J["hip"]]) for f in range(nfr))
        tpx = [tl[len(tl) // 2] or 1.0] * nfr
    cuts = set(ch.get("cuts", []))
    # смещение корня копим по шагам: на смене плана шаг не считаем
    root = [(0.0, 0.0)]
    for f in range(1, nfr):
        if f in cuts:
            root.append(root[-1])
            continue
        k = torso_rig / tpx[f]
        dx = (F[f][J["hip"]][0] - F[f - 1][J["hip"]][0]) * k
        dy = (F[f][J["hip"]][1] - F[f - 1][J["hip"]][1]) * k
        root.append((root[-1][0] + dx, root[-1][1] + dy))

    obj.animation_data_create()
    act = bpy.data.actions.new(f"{obj.name}_stick2r6")
    obj.animation_data.action = act
    for p in PARTS:
        pb = obj.pose.bones[bones[p]]
        pb.rotation_mode = "QUATERNION"

    prev_q = {p: None for p in PARTS}
    # ключевые кадры (если программа их выбрала): между ними Blender строит гладкие кривые
    keys = None
    if KEYS_ONLY and ch.get("keys"):
        keys = {k: set(v) for k, v in ch["keys"].items()}
    for f in range(nfr):
        P = F[f]
        Rw = {}
        for p in PARTS:
            a, b, sgn = SEG[p]
            d2 = (P[J[b]][0] - P[J[a]][0], P[J[b]][1] - P[J[a]][1])
            tgt = to3d(d2, u)
            if tgt.length < 1e-6:
                tgt = Vector((0, 0, sgn))
            ang = signed_angle(Vector((0, 0, sgn)), tgt.normalized(), nrm)
            Rw[p] = Quaternion(nrm, ang).to_matrix().to_4x4()
        # смещение корня
        off = Vector((0, 0, 0))
        if ROOT_MOTION:
            off = to3d((root[f][0], root[f][1] if KEEP_HEIGHT else 0.0), u)
        M = {}
        for p in PARTS:  # Torso первым — PARTS упорядочены от родителя
            R = rest[p]
            pivot = R.translation.copy()
            par = parent_part[p]
            if par is None:
                piv_new = pivot + off
            else:
                piv_new = M[par] @ rest[par].inverted() @ pivot
            M[p] = Matrix.Translation(piv_new) @ Rw[p] @ Matrix.Translation(-pivot) @ R
        for p in PARTS:
            pb = obj.pose.bones[bones[p]]
            par = parent_part[p]
            if par is None:
                pbone_parent = arm.bones[bones[p]].parent
                if pbone_parent is not None:  # например HumanoidRootPart выше Torso (не анимируется)
                    basis = (pbone_parent.matrix_local.inverted() @ rest[p]).inverted() @ \
                        pbone_parent.matrix_local.inverted() @ M[p]
                else:
                    basis = rest[p].inverted() @ M[p]
            else:
                basis = (rest[par].inverted() @ rest[p]).inverted() @ M[par].inverted() @ M[p]
            loc, q, _ = basis.decompose()
            if prev_q[p] is not None and q.dot(prev_q[p]) < 0:
                q.negate()
            prev_q[p] = q
            pb.rotation_quaternion = q
            if keys is None or f in keys[p]:
                pb.keyframe_insert("rotation_quaternion", frame=frame_start + f)
            if par is None:
                pb.location = loc
                if keys is None or f in keys["Root"] or f in keys[p]:
                    pb.keyframe_insert("location", frame=frame_start + f)
    return act


def run(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") != "stick2r6":
        raise RuntimeError("Это не файл stick2r6 (.r6.json)")
    scn = bpy.context.scene
    fps = data.get("fps", 30)
    scn.render.fps = max(1, int(round(fps)))
    scn.render.fps_base = scn.render.fps / fps if fps else 1
    chars = data["characters"]
    sel = [o for o in bpy.context.selected_objects if o.type == "ARMATURE"]
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    done = []
    for i, ch in enumerate(chars):
        if CHARACTERS and (i + 1) not in CHARACTERS:
            continue
        obj = sel[i] if i < len(sel) else build_r6(ch["name"], i * SPACING)
        apply_character(obj, ch, data)
        done.append(obj.name)
    scn.frame_start = 1
    scn.frame_end = max(len(c["frames"]) for c in chars)
    scn.frame_set(1)
    print("stick2r6: анимация применена к", done)
    return done


class STICK2R6_OT_import(bpy.types.Operator):
    """Импорт анимации stick2r6 (.r6.json) на R6-риг"""
    bl_idname = "import_anim.stick2r6"
    bl_label = "Импорт stick2r6 (.r6.json)"
    bl_options = {"REGISTER", "UNDO"}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            done = run(self.filepath)
        except Exception as e:  # покажем понятную ошибку в Blender, а детали — в консоли
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"stick2r6: {type(e).__name__}: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"stick2r6: готово — {', '.join(done)}")
        return {"FINISHED"}


def main():
    try:
        bpy.utils.unregister_class(STICK2R6_OT_import)
    except Exception:
        pass
    bpy.utils.register_class(STICK2R6_OT_import)
    import os
    if JSON_PATH and os.path.exists(JSON_PATH):
        return run(JSON_PATH)
    bpy.ops.import_anim.stick2r6("INVOKE_DEFAULT")


if __name__ == "__main__":
    main()
