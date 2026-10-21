import sys, json, math, time
import numpy as np, bpy
from mathutils import Vector
D = '/tmp/claude-0/-home-claude/5040af63-de2a-5d67-ae9e-0a4475d6ce34/scratchpad'
bpy.ops.wm.open_mainfile(filepath="basefile.blend")
P = bpy.data.objects["__PrimaryArmature"]; I = bpy.data.objects["InternalArmature"]
def infl(o, b): return [(c.type, c.subtarget, round(c.influence,2), c.mute) for c in o.pose.bones[b].constraints]
print("Positioner R arm", infl(P,"RightArm_Positioner"), "R leg", infl(P,"RightLeg_Positioner"), "FK", infl(P,"RightArm_FK")); print("obj rot", tuple(round(x,2) for x in P.matrix_world.to_euler()), tuple(round(x,2) for x in I.matrix_world.to_euler()))
for b in ("Right Arm","Right Leg","Torso"): print('I', b, infl(I,b))
print('rot modes', {b: P.pose.bones[b].rotation_mode for b in ("MasterControl","Torso_FK","LeftArm-IK")})
src = open('./blender_import_ikrig.py', encoding='utf-8').read()
g = {"__name__": "x"}; exec(compile(src, "ik", "exec"), g); g["LEG_MODE"] = "straight"
J = 'keys_4.0.r6.json'
t0 = time.time(); fr = g["run"](D + '/' + J); print('время импорта', round(time.time()-t0,1), 'с')
data = json.load(open(D + '/' + J)); ch = data["characters"][0]; F = np.array(ch["frames"]); Jn = {j:i for i,j in enumerate(ch["joints"])}
segs = {"Torso": ("hip","neck",1), "Head": ("neck","head",1), "Left Arm": ("neck","l_hand",-1), "Right Arm": ("neck","r_hand",-1),
        "Left Leg": ("hip","l_foot",-1), "Right Leg": ("hip","r_foot",-1)}
W = P.matrix_world.to_3x3()
err = {k: [] for k in segs}
for f in range(len(F)):
    bpy.context.scene.frame_set(f + 1)
    e = I.evaluated_get(bpy.context.evaluated_depsgraph_get())
    for part, (a, b, s) in segs.items():
        pb = e.pose.bones[part]; bone = I.data.bones[part]
        Iw = I.matrix_world.to_3x3()
        M = Iw @ (pb.matrix @ bone.matrix_local.inverted()).to_3x3() @ Iw.inverted()
        d = (M @ Vector((0,0,s))).normalized()
        v = F[f, Jn[b]] - F[f, Jn[a]]
        gt = (W @ Vector((0, v[0], -v[1]))).normalized()
        err[part].append(math.degrees(math.acos(max(-1,min(1,d.dot(gt))))))
for k, v in err.items():
    v = np.array(v); print(f'{k:10s} ср {v.mean():5.1f}°  p95 {np.percentile(v,95):5.1f}°  макс {v.max():5.1f}°')
bpy.ops.wm.save_as_mainfile(filepath=D + '/ik_check.blend')
