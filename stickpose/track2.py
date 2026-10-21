"""Трекер v2: кандидаты суставов от нейросети -> плавные траектории по всему видео.

Масштаб нормируется по толщине линий персонажа (она меняется вместе с зумом камеры),
поэтому скачки камеры и смена плана не ломают трекинг.
Лево/право для рук и ног решается Витерби 2-го порядка (плавность + память поз + ваши разметки).
"""
import numpy as np

JOINTS = ["head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot"]
JI = {j: i for i, j in enumerate(JOINTS)}
PAIRS = [(("l_hand", "r_hand"), "neck", "hands"), (("l_foot", "r_foot"), "hip", "feet")]
RU = {"head": "голова", "neck": "шея", "hip": "таз", "l_hand": "левая рука", "r_hand": "правая рука",
      "l_foot": "левая нога", "r_foot": "правая нога"}

CFG = dict(
    w_vel=0.15, w_acc=0.5,     # в единицах «длина торса» за кадр
    w_score=1.0,               # доверие нейросети: -log(score)
    hold_cost=3.0,             # «не нашёлся» (берём предсказание/интерполяцию)
    share_cost=0.4,            # обе руки в одной точке
    w_anchor=1.0,              # притяжение к разметке рядом с ней
    w_mem=0.8,                 # память поз (похожий силуэт)
    amb_gap=0.6,               # если вторая версия хуже меньше чем на это — спросить
    amb_dist=0.25,             # «другая версия» = отличается больше чем на 0.25 торса
    low_score=0.15,            # слабый сигнал нейросети -> спросить
    max_jump=0.9,              # скачок больше 0.9 торса за кадр (не на смене плана) -> спросить
    cut_jump=4.0,              # смена плана: всё тело сдвинулось сильнее -> разрыв, плавность не требуем
    flag_level=1.0,
)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _viterbi2(X, U, cuts, wv, wa):
    """Общий Витерби 2-го порядка. X[f]: (S_f, D) позиции, U[f]: (S_f,) цена.
    cuts[f]=True — между f-1 и f разрыв (смена плана). Возвращает путь и мин-маргиналы."""
    n = len(X)

    def vel(a, b, f):
        if cuts[f]:
            return np.zeros((len(a), len(b)))
        return wv * np.linalg.norm(b[None] - a[:, None], axis=2)

    def acc(a, b, c, f):
        if cuts[f] or cuts[f - 1]:
            return np.zeros((len(a), len(b), len(c)))
        return wa * np.linalg.norm(c[None, None] - 2 * b[None, :, None] + a[:, None, None], axis=3)

    if n == 1:
        i = int(np.argmin(U[0]))
        return [i], [U[0]]
    alpha = [None] * n
    alpha[1] = U[0][:, None] + U[1][None] + vel(X[0], X[1], 1)
    for f in range(2, n):
        alpha[f] = (alpha[f - 1][:, :, None] + acc(X[f - 2], X[f - 1], X[f], f)).min(0) + vel(X[f - 1], X[f], f) + U[f][None]
    beta = [None] * n
    beta[n - 1] = np.zeros_like(alpha[n - 1])
    for f in range(n - 2, 0, -1):
        nxt = vel(X[f], X[f + 1], f + 1) + U[f + 1][None]
        beta[f] = (acc(X[f - 1], X[f], X[f + 1], f + 1) + nxt[None] + beta[f + 1][None]).min(2)
    marg = [None] * n
    marg[0] = (alpha[1] + beta[1]).min(1)
    for f in range(1, n):
        marg[f] = (alpha[f] + beta[f]).min(0)
    path = [int(np.argmin(m)) for m in marg]
    return path, marg


def _interp_anchor(vals, anchors, n):
    """Линейная интерполяция векторов между размеченными кадрами (для «не нашёлся»)."""
    out = np.zeros((n,) + vals[anchors[0]].shape)
    for f in range(n):
        if f <= anchors[0]:
            out[f] = vals[anchors[0]]
        elif f >= anchors[-1]:
            out[f] = vals[anchors[-1]]
        else:
            k = int(np.searchsorted(anchors, f))
            a, b = anchors[k - 1], anchors[k]
            w = (f - a) / (b - a)
            out[f] = (1 - w) * vals[a] + w * vals[b]
    return out


def track(frames_c, labels, n, cfg=None, skipped=()):
    """frames_c[f]: dict из pipeline: {"ok", "peaks": {канал: [(x,y,score)...]}, "scale": толщина линии,
    "feat": вектор силуэта} — координаты в пикселях кадра.
    labels: {кадр: {сустав: [x,y] | None}}."""
    cfg = {**CFG, **(cfg or {})}
    labels = {int(k): v for k, v in labels.items() if 0 <= int(k) < n}
    if not labels:
        return None
    anchors = sorted(labels)
    skipped = set(int(s) for s in skipped)

    # ---- масштаб: толщина линии -> «длина торса» (по разметкам) ----
    thick = np.array([fc["scale"] if fc and fc.get("ok") else np.nan for fc in frames_c], float)
    if np.isnan(thick).all():
        thick[:] = 10.0
    ok = ~np.isnan(thick)
    thick = np.interp(np.arange(n), np.nonzero(ok)[0], thick[ok])
    ratios = []
    for a, L in labels.items():
        if L.get("neck") is not None and L.get("hip") is not None:
            ratios.append(np.linalg.norm(np.subtract(L["neck"], L["hip"])) / thick[a])
    ratio = float(np.median(ratios)) if ratios else 6.0
    T = thick * ratio                      # длина торса в пикселях в каждом кадре

    # ---- смены плана: центр тела прыгнул сильно ----
    cen = np.array([list(fc["peaks"]["hip"][0][:2]) if fc and fc.get("ok") else [np.nan, np.nan]
                    for fc in frames_c], float)
    cuts = np.zeros(n, bool)
    for f in range(1, n):
        if not np.isnan(cen[f]).any() and not np.isnan(cen[f - 1]).any():
            if np.linalg.norm(cen[f] - cen[f - 1]) / T[f] > cfg["cut_jump"] or abs(np.log(thick[f] / thick[f - 1])) > 0.5:
                cuts[f] = True

    P = np.zeros((n, 7, 2))
    C = np.zeros((n, 7))
    sev = np.zeros(n)
    why = [[] for _ in range(n)]

    def flag(f, s, msg):
        if f in labels or f in skipped:
            return
        sev[f] += s
        why[f].append(msg)

    def lab_pt(a, j):
        v = labels[a].get(j)
        return None if v is None else np.asarray(v, float)

    # заполнить невидимые на разметке точки торса соседними
    for a in anchors:
        for j in ("head", "neck", "hip"):
            if labels[a].get(j) is None:
                fc = frames_c[a]
                labels[a][j] = list(fc["peaks"][j][0][:2]) if fc and fc.get("ok") else [0.0, 0.0]

    # ---- торс: голова, шея, таз — каждый своим Витерби (в нормированных единицах) ----
    for j in ("head", "neck", "hip"):
        anc = {a: lab_pt(a, j) for a in anchors}
        prior = _interp_anchor(anc, anchors, n)
        X, U = [], []
        for f in range(n):
            if f in labels:
                X.append(anc[f][None] / T[f])
                U.append(np.zeros(1))
                continue
            fc = frames_c[f]
            xs = [prior[f]]
            us = [cfg["hold_cost"]]
            if fc and fc.get("ok"):
                for (x, y, s) in fc["peaks"][j][:3]:
                    xs.append(np.array([x, y]))
                    us.append(-cfg["w_score"] * np.log(max(s, 1e-3)))
            dA = np.min(np.abs(np.array(anchors) - f))
            wa_ = cfg["w_anchor"] * np.exp(-(dA - 1) / 2.0)
            us = [u + wa_ * np.linalg.norm(x - prior[f]) / T[f] for u, x in zip(us, xs)]
            X.append(np.array(xs) / T[f])
            U.append(np.array(us))
        path, marg = _viterbi2(X, U, cuts, cfg["w_vel"], cfg["w_acc"])
        for f in range(n):
            s = path[f]
            P[f, JI[j]] = X[f][s] * T[f]
            if f in labels:
                C[f, JI[j]] = 1.0
                continue
            far = np.linalg.norm(X[f] - X[f][s], axis=1) > cfg["amb_dist"]
            gap = float(marg[f][far].min() - marg[f][s]) if far.any() else 9.0
            sc = np.exp(-U[f][s] / cfg["w_score"]) if s > 0 else 0.0
            C[f, JI[j]] = min(1.0, sc)
            if s == 0:
                flag(f, 1.2, f"{RU[j]}: не найдена")
            elif gap < cfg["amb_gap"]:
                flag(f, 1.0 + (cfg["amb_gap"] - gap), f"{RU[j]}: два варианта")

    # ---- память поз ----
    fa = [a for a in anchors if frames_c[a] and frames_c[a].get("ok")]
    nn_k, nn_s = np.zeros(n, int), np.zeros(n)
    if len(fa) >= 2:
        FA = np.stack([frames_c[a]["feat"] for a in fa])
        dd = np.linalg.norm(FA[:, None] - FA[None], axis=2) + np.eye(len(fa)) * 1e9
        sig = float(np.median(dd.min(1))) + 1e-6
        for f in range(n):
            if frames_c[f] and frames_c[f].get("ok"):
                d = np.linalg.norm(FA - frames_c[f]["feat"], axis=1)
                nn_k[f] = int(np.argmin(d))
                nn_s[f] = float(np.exp(-(d[nn_k[f]] / (0.5 * sig)) ** 2))

    # ---- конечности парами ----
    for (jl, jr), par, ch in PAIRS:
        rel_a = {}
        for a in anchors:
            pa = P[a, JI[par]]
            v = []
            for j in (jl, jr):
                q = lab_pt(a, j)
                if q is None:  # не видно на разметке — берём ближайший пик сети
                    fc = frames_c[a]
                    q = np.asarray(fc["peaks"][ch][0][:2]) if fc and fc.get("ok") else pa
                v.append((q - pa) / T[a])
            rel_a[a] = np.array(v)
        prior = _interp_anchor(rel_a, anchors, n)
        X, U = [], []
        for f in range(n):
            if f in labels:
                X.append(rel_a[f].reshape(1, 4))
                U.append(np.zeros(1))
                continue
            fc = frames_c[f]
            opts = [(prior[f][0], cfg["hold_cost"]), ]
            cand = []
            if fc and fc.get("ok"):
                for (x, y, s) in fc["peaks"][ch][:4]:
                    cand.append(((np.array([x, y]) - P[f, JI[par]]) / T[f], -cfg["w_score"] * np.log(max(s, 1e-3))))
            o0 = [(prior[f][0], cfg["hold_cost"])] + cand
            o1 = [(prior[f][1], cfg["hold_cost"])] + cand
            dA = np.min(np.abs(np.array(anchors) - f))
            wa_ = cfg["w_anchor"] * np.exp(-(dA - 1) / 2.0)
            mq = rel_a[fa[nn_k[f]]] if len(fa) >= 2 else None
            xs, us = [], []
            for i, (r0, u0) in enumerate(o0):
                for k, (r1, u1) in enumerate(o1):
                    u = u0 + u1
                    if i > 0 and k > 0 and i == k:
                        u += cfg["share_cost"]
                    u += wa_ * (np.linalg.norm(r0 - prior[f][0]) + np.linalg.norm(r1 - prior[f][1]))
                    if mq is not None and nn_s[f] > 0:
                        dm = min(np.linalg.norm(r0 - mq[0]) + np.linalg.norm(r1 - mq[1]),
                                 np.linalg.norm(r0 - mq[1]) + np.linalg.norm(r1 - mq[0]))
                        u += cfg["w_mem"] * nn_s[f] * dm
                    xs.append(np.concatenate([r0, r1]))
                    us.append(u)
            X.append(np.array(xs))
            U.append(np.array(us))
        path, marg = _viterbi2(X, U, cuts, cfg["w_vel"], cfg["w_acc"])
        for f in range(n):
            s = path[f]
            v = X[f][s].reshape(2, 2)
            for side, j in enumerate((jl, jr)):
                P[f, JI[j]] = P[f, JI[par]] + v[side] * T[f]
                if f in labels:
                    C[f, JI[j]] = 1.0
                    continue
                cols = slice(2 * side, 2 * side + 2)
                far = np.linalg.norm(X[f][:, cols] - X[f][s, cols], axis=1) > cfg["amb_dist"]
                gap = float(marg[f][far].min() - marg[f][s]) if far.any() else 9.0
                C[f, JI[j]] = 1.0 / (1.0 + np.exp(-(gap - cfg["amb_gap"]) * 4))
                if gap < cfg["amb_gap"]:
                    flag(f, 1.0 + (cfg["amb_gap"] - gap), f"{RU[j]}: два варианта")

    # ---- рывки (не на смене плана) ----
    for f in range(1, n):
        if cuts[f]:
            continue
        d = np.linalg.norm(P[f] - P[f - 1], axis=1) / T[f]
        rel = d - d[JI["hip"]]
        k = int(np.argmax(rel))
        if rel[k] > cfg["max_jump"] and (f - 1) not in labels:
            flag(f, rel[k] / cfg["max_jump"], f"{RU[JOINTS[k]]}: рывок")

    return {"P": P, "C": C, "sev": sev, "why": why, "anchors": anchors, "cuts": np.nonzero(cuts)[0].tolist(),
            "T": T, "flag_level": cfg["flag_level"]}


def worst_frame(results):
    best = None
    for ci, r in enumerate(results):
        if r is None:
            continue
        f = int(np.argmax(r["sev"]))
        s = r["sev"][f]
        if s >= r["flag_level"] and (best is None or s > best[0]):
            best = (s, ci, f)
    return best


def count_flagged(results):
    return sum(int((r["sev"] >= r["flag_level"]).sum()) for r in results if r is not None)
