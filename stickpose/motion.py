"""Понимание движения, как у аниматора: события (крайние позы, удары, замахи, рывки, прыжки, паузы),
исправление перескоков рук/ног и ключи «от позы к позе» с интерполяцией по смыслу.

Все величины — в длинах торса T (масштаб персонажа в кадре) и градусах; кадры — индексы ролика.
"""
import numpy as np

from postprocess import J, PARTS, segments

ENDS = {"l_hand": "l_elbow", "r_hand": "r_elbow", "l_foot": "l_knee", "r_foot": "r_knee"}
PAIRS = ((("l_hand", "l_elbow"), ("r_hand", "r_elbow")), (("l_foot", "l_knee"), ("r_foot", "r_knee")))
RU = {"extreme": "крайняя поза", "hit": "удар", "stop": "резкая остановка", "windup": "замах", "dash": "рывок",
      "dash_end": "конец рывка", "takeoff": "отрыв", "apex": "высшая точка", "landing": "приземление",
      "hold": "пауза", "hold_end": "конец паузы", "got_hit": "получил удар"}
# важность события: что спрашивать у пользователя в первую очередь и что никогда не терять в ключах
IMPORTANCE = {"hit": 5, "got_hit": 4, "windup": 4, "landing": 4, "takeoff": 3, "stop": 3, "extreme": 3, "apex": 2,
              "dash": 2, "dash_end": 2, "hold": 1, "hold_end": 1}

CFG = dict(
    ext_min_deg=25.0,     # крайняя поза: часть повернулась хотя бы на столько с прошлой крайней
    hit_speed=0.22,       # удар/остановка: конец конечности летел быстрее (торсов за кадр)...
    hit_drop=0.4,         # ...и за 2 кадра потерял скорость хотя бы до этой доли
    contact=0.35,         # удар «в противника»: конец конечности ближе этого к его скелету (торсов)
    windup_win=8,         # замах — крайняя поза этой же конечности не дальше стольких кадров до удара
    dash_speed=0.28,      # рывок: таз движется быстрее (торсов за кадр) хотя бы 2 кадра
    air=0.25,             # прыжок: обе стопы выше земли на столько торсов
    hold_deg=1.5,         # пауза: все части поворачиваются медленнее (°/кадр)...
    hold_root=0.03,       # ...таз почти стоит...
    hold_len=3,           # ...хотя бы столько кадров
    breakdown_deg=10.0,   # промежуточный ключ, если без него дуга части уйдёт дальше этого
    breakdown_root=0.08,  # ...или путь таза (торсов)
    max_gap=10,           # ключ не реже, чем раз в столько кадров
    merge=2,              # события в пределах стольких кадров — одна поза (один ключ)
)


def _zigzag(y, thr):
    """Крайние точки кривой, «значимые» как у аниматора: размах до соседних крайних не меньше thr.
    Мелкие зигзаги (шум) убираются по одному, начиная с самого маленького."""
    w = np.gradient(y)
    ext = [i for i in range(1, len(y) - 1) if w[i - 1] * w[i + 1] < 0 or (w[i] == 0 and w[i - 1] * w[i + 1] <= 0)]
    pts = [0] + ext + [len(y) - 1]
    while len(pts) > 2:
        sw = [min(abs(y[pts[i]] - y[pts[i - 1]]), abs(y[pts[i + 1]] - y[pts[i]])) for i in range(1, len(pts) - 1)]
        if not sw:
            break
        k = int(np.argmin(sw))
        if sw[k] >= thr:
            break
        del pts[k + 1]
        # после удаления соседние могут перестать быть крайними (одно направление) — склеить
        i = 1
        while i < len(pts) - 1:
            if (y[pts[i]] - y[pts[i - 1]]) * (y[pts[i + 1]] - y[pts[i]]) > 0:
                del pts[i]
            else:
                i += 1
    return pts[1:-1]


def _smooth(P, cuts, w=7):
    """Сглаживание для поиска событий: сначала убрать одиночные выбросы (медиана по 3 кадрам), потом шум."""
    from scipy.signal import savgol_filter
    S = P.copy()
    for a, b in segments(len(P), cuts):
        if b - a >= 3:
            X = P[a:b]
            med = np.median(np.stack([np.r_[X[:1], X[:-1]], X, np.r_[X[1:], X[-1:]]]), axis=0)
            S[a:b] = med
        if b - a > w:
            S[a:b] = savgol_filter(S[a:b], w, 2, axis=0, mode="interp")
    return S


def clean(P, cuts, labeled=(), w=5):
    """Позы для ключей: без одиночных выбросов и мелкого шума; размеченные кадры — как есть (они закон)."""
    S = _smooth(P, cuts, w)
    for f in labeled:
        if 0 <= int(f) < len(P):
            S[int(f)] = P[int(f)]
    return S


def angles(P):
    """Углы частей R6 по правилу переноса (рад, развёрнуты по времени): {часть: (n,)}."""
    out = {}
    for name, (a, b) in PARTS.items():
        v = P[:, J[b]] - P[:, J[a]]
        out[name] = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    return out


def fix_swaps(P, T, labeled=(), jump=0.3, ratio=0.5):
    """Перескоки: если конечности «поменялись местами» за кадр (с обменом левая/правая расстояние до прошлого
    кадра вдвое меньше и скачок заметный) — меняем обратно до конца ролика. Размеченные кадры не трогаем."""
    P = P.copy()
    labeled = set(labeled)
    fixed = 0
    for (la, lb), (ra, rb) in PAIRS:
        L, R = [J[la], J[lb]], [J[ra], J[rb]]
        for f in range(1, len(P)):
            if f in labeled:
                continue
            same = np.linalg.norm(P[f, L] - P[f - 1, L], axis=1).sum() + np.linalg.norm(P[f, R] - P[f - 1, R], axis=1).sum()
            swap = np.linalg.norm(P[f, R] - P[f - 1, L], axis=1).sum() + np.linalg.norm(P[f, L] - P[f - 1, R], axis=1).sum()
            if same / T[f] > jump and swap < ratio * same:
                nxt = min([g for g in labeled if g > f] + [len(P)])
                P[f:nxt, L + R] = P[f:nxt, R + L]
                fixed += 1
    return P, fixed


def _skeleton_dist(p, Q):
    """Расстояние от точки p до скелета Q (11 точек): до ближайшего отрезка кости."""
    segs = [("hip", "neck"), ("neck", "head"), ("neck", "l_elbow"), ("l_elbow", "l_hand"), ("neck", "r_elbow"),
            ("r_elbow", "r_hand"), ("hip", "l_knee"), ("l_knee", "l_foot"), ("hip", "r_knee"), ("r_knee", "r_foot")]
    best = 1e9
    for a, b in segs:
        A, B = Q[J[a]], Q[J[b]]
        d = B - A
        t = np.clip(np.dot(p - A, d) / (np.dot(d, d) + 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(p - (A + t * d))))
    return best


def detect(P, T, cuts=(), others=(), cfg=None):
    """События движения персонажа. others — траектории (n,11,2) других персонажей (для «удар в противника»).
    -> список {"f", "type", "part"} по времени и отрезки пауз [(a, b)]."""
    cfg = {**CFG, **(cfg or {})}
    n = len(P)
    cuts = sorted(set(cuts))
    S = _smooth(P, cuts)
    ev = []
    cutset = set(cuts)

    def add(f, t, part=None):
        if 0 <= f < n:
            ev.append({"f": int(f), "type": t, "part": part})

    A = angles(S)
    hip = S[:, J["hip"]]
    for a, b in segments(n, cuts):
        if b - a < 3:
            continue
        # крайние позы: часть развернулась, и размах в обе стороны значимый (не шум)
        for name, y in A.items():
            for i in _zigzag(np.degrees(y[a:b]), cfg["ext_min_deg"]):
                add(a + i, "extreme", name)
        # удары и резкие остановки: конец конечности летел и резко встал
        for e in ENDS:
            rel = (S[a:b, J[e]] - hip[a:b]) / T[a:b, None]
            sp = np.r_[0, np.linalg.norm(np.diff(rel, axis=0), axis=1)]
            for i in range(2, b - a - 2):
                fast_in = min(sp[i - 1], sp[i]) > cfg["hit_speed"] * 0.7 and sp[i] > cfg["hit_speed"]
                slow_out = max(sp[i + 1], sp[i + 2]) < cfg["hit_drop"] * sp[i]
                if fast_in and slow_out:
                    f = a + i
                    near = [_skeleton_dist(S[f, J[e]], O[f]) / T[f] for O in others if O is not None]
                    add(f, "hit" if near and min(near) < cfg["contact"] else "stop", e)
        # рывки: таз быстро едет
        v = np.r_[0, np.linalg.norm(np.diff(hip[a:b], axis=0), axis=1)] / T[a:b]
        fast = v > cfg["dash_speed"]
        i = 0
        while i < len(fast):
            if fast[i]:
                j = i
                while j + 1 < len(fast) and fast[j + 1]:
                    j += 1
                if j - i + 1 >= 2:
                    add(a + i - 1, "dash")
                    add(a + j, "dash_end")
                i = j + 1
            else:
                i += 1
        # прыжки: обе стопы выше земли
        feet_y = S[a:b][:, [J["l_foot"], J["r_foot"]], 1]
        ground = np.percentile(feet_y, 95)
        air = feet_y.max(1) < ground - cfg["air"] * T[a:b]
        i = 0
        while i < len(air):
            if air[i]:
                j = i
                while j + 1 < len(air) and air[j + 1]:
                    j += 1
                if j - i + 1 >= 3:
                    add(a + i - 1, "takeoff")
                    add(a + i + int(np.argmin(hip[a + i:a + j + 1, 1])), "apex")
                    add(a + j + 1, "landing")
                i = j + 1
            else:
                i += 1
    # паузы: всё почти стоит
    speed = np.max(np.abs(np.degrees(np.stack([np.gradient(y) for y in A.values()]))), axis=0)
    vroot = np.r_[0, np.linalg.norm(np.diff(hip, axis=0), axis=1)] / T
    still = (speed < cfg["hold_deg"]) & (vroot < cfg["hold_root"])
    holds, i = [], 0
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1] and (j + 1) not in cutset:
                j += 1
            if j - i + 1 >= cfg["hold_len"]:
                holds.append((i, j))
                add(i, "hold")
                add(j, "hold_end")
            i = j + 1
        else:
            i += 1
    # замах: крайняя поза той же руки/ноги незадолго до удара
    part_of = {"l_hand": "Left Arm", "r_hand": "Right Arm", "l_foot": "Left Leg", "r_foot": "Right Leg"}
    for h in [x for x in ev if x["type"] in ("hit", "stop")]:
        cand = [x for x in ev if x["type"] == "extreme" and x["part"] == part_of[h["part"]]
                and 1 <= h["f"] - x["f"] <= cfg["windup_win"]]
        if cand:
            cand[-1]["type"] = "windup"
    # убрать дубли (одно событие одного типа в соседних кадрах)
    ev.sort(key=lambda x: (x["f"], -IMPORTANCE[x["type"]]))
    out = []
    for x in ev:
        if any(y["type"] == x["type"] and y["part"] == x["part"] and x["f"] - y["f"] <= 2 for y in out[-6:]):
            continue
        out.append(x)
    return out, holds


def link_hits(events, Ps, Ts, contact=CFG["contact"]):
    """«Получил удар»: удар персонажа i, пришедшийся рядом со скелетом j, -> событие got_hit у j в тот же кадр."""
    for i, ev in enumerate(events):
        for x in ev:
            if x["type"] != "hit":
                continue
            f, p = x["f"], Ps[i][x["f"], J[x["part"]]]
            for j, Q in enumerate(Ps):
                if j != i and Q is not None and _skeleton_dist(p, Q[f]) / Ts[j][f] < contact:
                    events[j].append({"f": f, "type": "got_hit", "part": None})
    for ev in events:
        ev.sort(key=lambda x: x["f"])
    return events


def edit_filter(P, T, cuts=(), labeled=(), k=2.5, min_deg=3.0, root_tol=0.03):
    """«Правка кадра», как у аниматора: каждая кость держит прошлое положение, пока её изменение не превысит
    обычный шум сети; сдвинулась только рука — правится только рука. Порог по каждой кости свой: k × шум
    (оценён по отклонению от медианы соседних кадров), но не меньше min_deg. Разметка и смены плана — как есть."""
    n = len(P)
    if n < 3:
        return P.copy()
    lab, cutset = set(int(f) for f in labeled), set(int(c) for c in cuts)
    hip = P[:, J["hip"]].copy()
    ang, ln = [], []
    for a, b in CHAIN:
        v = P[:, J[b]] - P[:, J[a]]
        ang.append(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))
        ln.append(np.linalg.norm(v, axis=1))
    ang, ln = np.stack(ang, 1), np.stack(ln, 1)

    def noise(y):
        med = np.median(np.stack([np.r_[y[:1], y[:-1]], y, np.r_[y[1:], y[-1:]]]), 0)
        return 1.4826 * np.median(np.abs(y - med), 0)
    thr_a = np.maximum(np.radians(min_deg), k * noise(ang))
    thr_l = np.maximum(0.05 * np.median(ln, 0), k * noise(ln))
    out_a, out_l, out_h = ang.copy(), ln.copy(), hip.copy()
    for f in range(1, n):
        if f in lab or f in cutset:
            continue
        keep = np.abs(ang[f] - out_a[f - 1]) < thr_a
        out_a[f, keep] = out_a[f - 1, keep]
        keep = np.abs(ln[f] - out_l[f - 1]) < thr_l
        out_l[f, keep] = out_l[f - 1, keep]
        if np.linalg.norm(hip[f] - out_h[f - 1]) < root_tol * T[f]:
            out_h[f] = out_h[f - 1]
    out = np.zeros_like(P)
    out[:, J["hip"]] = out_h
    for c, (a, b) in enumerate(CHAIN):
        out[:, J[b]] = out[:, J[a]] + np.stack([np.cos(out_a[:, c]), np.sin(out_a[:, c])], 1) * out_l[:, c:c + 1]
    return out


def keyframes(P, T, cuts, events, holds, labeled=(), cfg=None):
    """Ключи «от позы к позе»: события + разметка + границы планов, затем промежуточные ключи там, где без них
    дуга части уйдёт дальше допуска, и не реже max_gap. -> (кадры, {кадр: интерполяция сегмента после него})."""
    cfg = {**CFG, **(cfg or {})}
    n = len(P)
    S = _smooth(P, cuts)
    A = angles(S)
    root = S[:, J["hip"]] / T[:, None]
    keys = {0, n - 1} | set(int(c) for c in cuts) | {max(0, int(c) - 1) for c in cuts} | set(int(x) for x in labeled)
    # поза всего тела: события разных частей в соседних кадрах (±merge) -> один ключ (самое важное событие)
    evs = sorted(events, key=lambda x: x["f"])
    i = 0
    while i < len(evs):
        j = i
        while j + 1 < len(evs) and evs[j + 1]["f"] - evs[i]["f"] <= cfg["merge"]:
            j += 1
        grp = evs[i:j + 1]
        top = max(IMPORTANCE[x["type"]] for x in grp)
        fs = [x["f"] for x in grp if IMPORTANCE[x["type"]] == top]
        keys.add(int(round(np.median(fs))) if top < IMPORTANCE["stop"] else fs[0])
        i = j + 1
    keys = sorted(k for k in keys if 0 <= k < n)
    cutset = set(cuts)

    def worst(a, b):
        if b - a < 2:
            return None, 0.0
        t = (np.arange(a + 1, b) - a) / (b - a)
        best, bf = 0.0, None
        for y in A.values():
            d = np.degrees(np.abs(y[a + 1:b] - (y[a] + (y[b] - y[a]) * t))) / cfg["breakdown_deg"]
            k = int(np.argmax(d))
            if d[k] > best:
                best, bf = d[k], a + 1 + k
        d = np.linalg.norm(root[a + 1:b] - (root[a] + (root[b] - root[a]) * t[:, None]), axis=1) / cfg["breakdown_root"]
        k = int(np.argmax(d))
        if d[k] > best:
            best, bf = d[k], a + 1 + k
        return bf, best

    out, stack = set(keys), list(zip(keys, keys[1:]))
    while stack:
        a, b = stack.pop()
        if any(c in cutset for c in range(a + 1, b + 1)):
            continue  # через смену плана не тянем
        f, sc = worst(a, b)
        if f is not None and (sc > 1.0 or b - a > cfg["max_gap"]):
            if sc <= 1.0:
                f = (a + b) // 2
            out.add(f)
            stack += [(a, f), (f, b)]
    keys = sorted(out)
    # интерполяция сегмента «после ключа»: по смыслу события
    interp = {k: "BEZIER" for k in keys}
    hits = {x["f"] for x in events if x["type"] in ("hit", "stop")}
    dash = [(x["f"], y["f"]) for x, y in zip(events, events[1:]) if x["type"] == "dash" and y["type"] == "dash_end"]
    for k0, k1 in zip(keys, keys[1:]):
        if k1 in hits:
            interp[k0] = "LINEAR"            # в удар — без замедления
        if any(a <= k0 < b for a, b in dash):
            interp[k0] = "LINEAR"            # рывок держит скорость
        if any(a <= k0 and k1 <= b for a, b in holds):
            interp[k0] = "CONSTANT"          # пауза — держим позу
    return keys, interp


def anim_metrics(P, T, keys, interp, contacts=None, fps=30.0):
    """Метрики анимации (на кривой, которую увидит Blender): дрожание, перескоки, скольжение стоп, ключей/с."""
    n = len(P)
    sim = simulate(P, keys, interp)
    A = angles(sim)
    # дрожание — шум поверх движения: отклонение угла от его же сглаженной кривой (резкий удар шумом не считается)
    from scipy.signal import savgol_filter
    jitter = float(np.mean([np.mean(np.abs(np.degrees(y - savgol_filter(y, 7, 2)))) for y in A.values()])) if n > 7 else 0.0
    _, flips = fix_swaps(sim, T)
    slide = []
    for j, spans in (contacts or {}).items():
        for a, b in spans:
            pts = sim[a:b + 1, J[j]]
            slide.append(float(np.linalg.norm(pts - pts[0], axis=1).max() / T[a]))
    return {"ключей_в_с": len(keys) / max(1e-6, n / fps), "дрожание": jitter, "перескоки": flips,
            "скольжение_стоп": max(slide) if slide else 0.0}


CHAIN = [("hip", "neck"), ("neck", "head"), ("neck", "l_elbow"), ("l_elbow", "l_hand"), ("neck", "r_elbow"),
         ("r_elbow", "r_hand"), ("hip", "l_knee"), ("l_knee", "l_foot"), ("hip", "r_knee"), ("r_knee", "r_foot")]


def simulate(P, keys, interp):
    """Кривая между ключами, как её построит Blender: кости ПОВОРАЧИВАЮТСЯ (интерполируются угол и длина каждой
    кости и путь таза), BEZIER — с авто-ручками (плавно через промежуточные ключи, замедление на крайних),
    LINEAR — прямо, CONSTANT — держать."""
    keys = list(keys)
    if len(keys) < 2:
        return P.copy()
    n = len(P)
    feats = [P[:, J["hip"], 0], P[:, J["hip"], 1]]
    for a, b in CHAIN:
        v = P[:, J[b]] - P[:, J[a]]
        feats += [np.unwrap(np.arctan2(v[:, 1], v[:, 0])), np.linalg.norm(v, axis=1)]
    V = np.stack(feats, 1)
    kt = np.array(keys, float)
    kv = V[keys]
    m = np.zeros_like(kv)   # авто-ручки: наклон по соседям, на крайней точке — ноль (как auto-clamped)
    for i in range(1, len(keys) - 1):
        d0, d1 = kv[i] - kv[i - 1], kv[i + 1] - kv[i]
        m[i] = np.where(d0 * d1 > 0, (kv[i + 1] - kv[i - 1]) / (kt[i + 1] - kt[i - 1]), 0.0)
    R = V.copy()
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        if b - a < 2:
            continue
        t = ((np.arange(a + 1, b) - a) / (b - a))[:, None]
        mode = interp.get(a, "BEZIER")
        if mode == "CONSTANT":
            R[a + 1:b] = kv[i]
        elif mode == "LINEAR":
            R[a + 1:b] = kv[i] + (kv[i + 1] - kv[i]) * t
        else:
            h = b - a
            R[a + 1:b] = ((2 * t ** 3 - 3 * t ** 2 + 1) * kv[i] + (t ** 3 - 2 * t ** 2 + t) * h * m[i]
                          + (-2 * t ** 3 + 3 * t ** 2) * kv[i + 1] + (t ** 3 - t ** 2) * h * m[i + 1])
    out = np.zeros_like(P)
    out[:, J["hip"]] = R[:, :2]
    for c, (a, b) in enumerate(CHAIN):
        ang, ln = R[:, 2 + 2 * c], R[:, 3 + 2 * c]
        out[:, J[b]] = out[:, J[a]] + np.stack([np.cos(ang), np.sin(ang)], 1) * ln[:, None]
    return out
