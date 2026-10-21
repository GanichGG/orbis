"""Окна: выбор персонажей, разметка суставов, просмотр результата (OpenCV + текст через Pillow)."""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JOINTS = ["head", "neck", "hip", "l_hand", "r_hand", "l_foot", "r_foot", "l_elbow", "r_elbow", "l_knee", "r_knee"]
KEYS = "1234567890-"   # клавиша -> точка (по порядку JOINTS)
BONES = [("neck", "head"), ("neck", "hip"), ("neck", "l_elbow"), ("l_elbow", "l_hand"), ("neck", "r_elbow"),
         ("r_elbow", "r_hand"), ("hip", "l_knee"), ("l_knee", "l_foot"), ("hip", "r_knee"), ("r_knee", "r_foot")]

WIN = "stick2r6"
DISP_W, DISP_H = 1280, 720
PANEL = 92

JCOL = {"head": (255, 255, 255), "neck": (200, 200, 200), "hip": (200, 200, 200),
        "l_hand": (255, 170, 40), "l_foot": (255, 110, 0),      # левая сторона — синие оттенки (BGR)
        "r_hand": (40, 170, 255), "r_foot": (0, 90, 255),       # правая — оранжевые
        "l_elbow": (255, 210, 120), "l_knee": (255, 160, 60), "r_elbow": (120, 210, 255), "r_knee": (60, 150, 255)}
JNAME = {"head": "Голова", "neck": "Шея (плечи)", "hip": "Таз", "l_hand": "Левая кисть",
         "r_hand": "Правая кисть", "l_foot": "Левая стопа", "r_foot": "Правая стопа",
         "l_elbow": "Левый локоть", "r_elbow": "Правый локоть", "l_knee": "Левое колено", "r_knee": "Правое колено"}
CHAR_COL = [(80, 220, 80), (230, 80, 230), (80, 220, 230), (230, 200, 60), (60, 120, 255), (200, 200, 200)]

_FONT = {}


def font(sz):
    if sz in _FONT:
        return _FONT[sz]
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            _FONT[sz] = ImageFont.truetype(p, sz)
            return _FONT[sz]
    _FONT[sz] = ImageFont.load_default()
    return _FONT[sz]


def put_lines(img, lines, x=10, y=6, sz=17):
    """Текст с кириллицей. lines: список (строка, цвет RGB)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for s, col in lines:
        d.text((x, y), s, font=font(sz), fill=col)
        y += sz + 5
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class View:
    """Масштабирование кадра в окно + панель подсказок сверху."""

    def __init__(self, shape):
        h, w = shape[:2]
        self.s = min(DISP_W / w, (DISP_H - PANEL) / h, 2.0)
        self.w, self.h = int(w * self.s), int(h * self.s)

    def compose(self, frame, lines):
        img = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA if self.s < 1 else cv2.INTER_LINEAR)
        canvas = np.full((self.h + PANEL, max(self.w, 900), 3), 28, np.uint8)
        canvas[PANEL:PANEL + self.h, :self.w] = img
        return put_lines(canvas, lines)

    def to_img(self, x, y):
        return (x / self.s, (y - PANEL) / self.s)

    def to_disp(self, p):
        return (int(round(p[0] * self.s)), int(round(p[1] * self.s + PANEL)))


def _wait(delay=20):
    k = cv2.waitKeyEx(delay)
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        return "close"
    return k


def _key(k):
    if k == -1 or k == "close":
        return k
    if k in (2424832, 65361):
        return "left"
    if k in (2555904, 65363):
        return "right"
    c = k & 0xFF
    if c in (13, 10):
        return "enter"
    if c == 27:
        return "esc"
    if c == 8:
        return "back"
    if c == 9:
        return "tab"
    if c == 32:
        return "space"
    ch = chr(c).lower() if c < 128 else ""
    ru = {"ф": "a", "в": "d", "й": "q", "ц": "w", "ы": "s", "я": "z", "у": "e", "т": "n", "к": "r", "ч": "x", "з": "p"}
    return ru.get(ch, ch)


def ensure_window():
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)


def pick_characters(get_frame, n_frames, start=0):
    """Пользователь выбирает кадр (A/D, стрелки) и кликает по каждому персонажу."""
    ensure_window()
    f = start
    clicks = []
    view = View(get_frame(f).shape)
    state = {"click": None}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and y > PANEL:
            state["click"] = view.to_img(x, y)

    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        fr = get_frame(f).copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(fr, (int(x), int(y)), 7, CHAR_COL[i % len(CHAR_COL)], -1)
            cv2.putText(fr, str(i + 1), (int(x) + 9, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CHAR_COL[i % len(CHAR_COL)], 2)
        img = view.compose(fr, [
            (f"ШАГ 1. Кадр {f + 1}/{n_frames}.  Кликните по каждому персонажу (по туловищу) — по одному клику.", (255, 255, 255)),
            ("A/D или ←/→ — выбрать кадр, где все видны и не перекрываются;  Backspace — отменить клик;", (190, 190, 190)),
            (f"Enter — готово (персонажей: {len(clicks)});  Esc — выход.", (190, 190, 190)),
        ])
        cv2.imshow(WIN, img)
        if state["click"] is not None:
            clicks.append(state["click"])
            state["click"] = None
        k = _key(_wait())
        if k == "close" or k == "esc":
            return None, None
        if k in ("a", "left") and not clicks:
            f = max(0, f - 1)
        elif k in ("d", "right") and not clicks:
            f = min(n_frames - 1, f + 1)
        elif k == "back" and clicks:
            clicks.pop()
        elif k == "enter" and clicks:
            return f, clicks


def pick_colors(get_frame, n_frames, segment_preview, sample, start=0):
    """Пипетка: для каждого персонажа клик(и) по его цвету. Живой предпросмотр выделения.
    segment_preview(frame, colors_lab_lists, points) -> список масок.
    Возвращает (кадр, [список Lab-цветов по персонажам], [точка клика по персонажам])."""
    ensure_window()
    f = start
    chars = [[]]          # по персонажам: список (x, y)
    view = View(get_frame(f).shape)
    st = {"click": None, "dirty": True}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and y > PANEL:
            st["click"] = view.to_img(x, y)

    cv2.setMouseCallback(WIN, on_mouse)
    masks = None
    while True:
        if st["click"] is not None:
            x, y = st["click"]
            chars[-1].append((x, y, f, sample(get_frame(f), x, y)))
            st["click"] = None
            st["dirty"] = True
        if st["dirty"]:
            fr = get_frame(f)
            active = [c for c in chars if c]
            if active:
                masks = segment_preview(fr, [[p[3] for p in c] for c in active],
                                        [next(((p[0], p[1]) for p in reversed(c) if p[2] == f), None) for c in active])
            show = fr.copy()
            if masks is not None and active:
                dim = (show * 0.45).astype(np.uint8)
                for i, m in enumerate(masks):
                    dim[m] = CHAR_COL[i % len(CHAR_COL)]
                show = dim
            for i, pts in enumerate(chars):
                for (x, y, ff, _) in pts:
                    if ff != f:
                        continue
                    cv2.drawMarker(show, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            cur = len(chars)
            img = view.compose(show, [
                (f"ШАГ 1. Пипетка. Кадр {f + 1}/{n_frames}.  Кликните по персонажу {cur} (можно несколько кликов по разным оттенкам).",
                 (255, 255, 255)),
                ("N — следующий персонаж,  Backspace — убрать последний клик,  A/D — другой кадр (проверить выделение),  Enter — готово",
                 (190, 190, 190)),
                (f"Персонажей: {len(active)}.  Цветная заливка = что программа считает персонажем.  Esc — выход.", (190, 190, 190)),
            ])
            cv2.imshow(WIN, img)
            st["dirty"] = False
        k = _key(_wait())
        if k == -1:
            continue
        st["dirty"] = True
        if k in ("close", "esc"):
            return None, None, None
        if k in ("a", "left"):
            f = max(0, f - 1)
        elif k in ("d", "right"):
            f = min(n_frames - 1, f + 1)
        elif k == "n" and chars[-1]:
            chars.append([])
        elif k == "back":
            if chars[-1]:
                chars[-1].pop()
            elif len(chars) > 1:
                chars.pop()
        elif k == "enter":
            active = [c for c in chars if c]
            if active:
                pts = [next(((p[0], p[1]) for p in reversed(c) if p[2] == f), None) for c in active]
                return f, [[p[3].tolist() for p in c] for c in active], pts


def _draw_skel(img, view, P, col, thick=2, labels=False):
    for a, b in BONES:
        pa, pb = P.get(a), P.get(b)
        if pa is not None and pb is not None:
            cv2.line(img, view.to_disp(pa), view.to_disp(pb), col, thick, cv2.LINE_AA)
    for j in JOINTS:
        p = P.get(j)
        if p is not None:
            cv2.circle(img, view.to_disp(p), 5, JCOL[j], -1, cv2.LINE_AA)
            cv2.circle(img, view.to_disp(p), 5, (0, 0, 0), 1, cv2.LINE_AA)


def label_frame(frame, cand, est, title, reason, char_idx, progress=""):
    """Разметка одного кадра. est — подсказка (dict сустав->(x,y) или None).
    Возвращает ("ok", pts) / ("skip", None) / ("stop", None) / ("close", None)."""
    ensure_window()
    view = View(frame.shape)
    P = {j: (None if est is None or est.get(j) is None else tuple(map(float, est[j]))) for j in JOINTS}
    have_est = est is not None and any(v is not None for v in P.values())
    cur = 0 if not have_est else -1
    hist = []
    st = {"drag": None, "dirty": True}

    def nearest(x, y, r=14):
        best, bd = None, r
        for j in JOINTS:
            if P[j] is not None:
                dx, dy = view.to_disp(P[j])
                d = np.hypot(dx - x, dy - y)
                if d < bd:
                    best, bd = j, d
        return best

    def on_mouse(ev, x, y, flags, *_):
        nonlocal cur
        if y < PANEL:
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            hist.append(dict(P))
            if cur >= 0:
                P[JOINTS[cur]] = view.to_img(x, y)
                if have_est:
                    cur = -1
                else:  # режим «по порядку»: к следующей неотмеченной точке
                    cur = next((i for i in range(cur + 1, len(JOINTS)) if P[JOINTS[i]] is None), -1)
            else:
                j = nearest(x, y)
                if j is not None:
                    st["drag"] = j
                else:
                    hist.pop()
            st["dirty"] = True
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            P[st["drag"]] = view.to_img(x, y)
            st["dirty"] = True
        elif ev == cv2.EVENT_LBUTTONUP:
            st["drag"] = None

    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        if st["dirty"]:
            img_src = frame.copy()
            if cand and cand.get("ok") and len(cand["skel"]):
                sk = cand["skel"]
                img_src[sk[:, 1], sk[:, 0]] = CHAR_COL[char_idx % len(CHAR_COL)]
            lines = [
                (f"{title}   {progress}", (255, 255, 255)),
                (f"Почему спрашиваю: {reason}" if reason else "Отметьте суставы персонажа (подсвечен цветом).", (255, 210, 120)),
            ]
            if cur >= 0:
                lines.append((f"Кликните: {JNAME[JOINTS[cur]]}.   Tab — следующая точка, Backspace — «не видно», Z — отмена", (120, 255, 160)))
            else:
                lines.append(("Перетащите неверные точки.  Enter — верно.  1-0,- — переставить точку кликом.  Q — поменять руки, W — ноги,  S — пропустить кадр,  Esc — хватит вопросов", (120, 255, 160)))
            img = view.compose(img_src, lines)
            _draw_skel(img, view, P, (255, 255, 255))
            # легенда
            x0 = view.w - 170 if view.w > 400 else 10
            leg = img.copy()
            for i, j in enumerate(JOINTS):
                y = PANEL + 10 + i * 20
                cv2.circle(leg, (x0, y + 7), 6, JCOL[j], -1)
                mark = "▶ " if i == cur else ""
                if P[j] is None:
                    mark += "(нет) "
                leg = put_lines(leg, [(f"{KEYS[i]} {mark}{JNAME[j]}", (230, 230, 230))], x=x0 + 12, y=y - 2, sz=14)
            cv2.imshow(WIN, leg)
            st["dirty"] = False
        k = _key(_wait(15))
        if k == -1:
            continue
        st["dirty"] = True
        if k == "close":
            return "close", None
        if k == "esc":
            return "stop", None
        if k == "s":
            return "skip", None
        if k == "z" and hist:
            P.update(hist.pop())
        elif k == "tab":
            cur = (cur + 1) % len(JOINTS)
        elif k == "back" and cur >= 0:
            hist.append(dict(P))
            P[JOINTS[cur]] = None
            cur = cur + 1 if cur + 1 < len(JOINTS) else -1
        elif k in list(KEYS):
            cur = KEYS.index(k)
        elif k == "q":  # рука целиком: кисть и локоть
            hist.append(dict(P))
            P["l_hand"], P["r_hand"] = P["r_hand"], P["l_hand"]
            P["l_elbow"], P["r_elbow"] = P["r_elbow"], P["l_elbow"]
        elif k == "w":  # нога целиком: стопа и колено
            hist.append(dict(P))
            P["l_foot"], P["r_foot"] = P["r_foot"], P["l_foot"]
            P["l_knee"], P["r_knee"] = P["r_knee"], P["l_knee"]
        elif k in ("enter", "space"):
            if P["neck"] is None or P["hip"] is None:
                cur = JOINTS.index("neck") if P["neck"] is None else JOINTS.index("hip")
                continue
            return "ok", {j: (None if P[j] is None else [float(P[j][0]), float(P[j][1])]) for j in JOINTS}


def review(get_frame, n, results, start=0):
    """Просмотр: скелеты всех персонажей поверх видео. Возвращает ("export"|"edit"|"close", кадр, персонаж)."""
    ensure_window()
    f = start
    play = False
    sel = 0
    view = View(get_frame(0).shape)
    cv2.setMouseCallback(WIN, lambda *a: None)
    while True:
        img = view.compose(get_frame(f), [
            (f"ШАГ 3. Проверка.  Кадр {f + 1}/{n}.  Пробел — играть/пауза,  A/D — кадр,  ←/→ — по 10", (255, 255, 255)),
            (f"E — поправить этот кадр (персонаж {sel + 1}; TAB — сменить персонажа),  Enter — экспорт,  Esc — выход без экспорта", (190, 190, 190)),
        ])
        for ci, r in enumerate(results):
            if r is None:
                continue
            P = {j: r["P"][f, i] for i, j in enumerate(JOINTS)}
            _draw_skel(img, view, P, CHAR_COL[ci % len(CHAR_COL)], 3 if ci == sel else 2)
            if f in r["anchors"]:
                cv2.putText(img, "*", view.to_disp(P["head"]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WIN, img)
        k = _key(_wait(40 if play else 30))
        if k == "close":
            return "close", f, sel
        if play and k == -1:
            f = (f + 1) % n
            continue
        if k == "space":
            play = not play
        elif k == "a":
            f = max(0, f - 1)
        elif k == "d":
            f = min(n - 1, f + 1)
        elif k == "left":
            f = max(0, f - 10)
        elif k == "right":
            f = min(n - 1, f + 10)
        elif k == "tab":
            sel = (sel + 1) % len(results)
        elif k == "e":
            return "edit", f, sel
        elif k == "enter":
            return "export", f, sel
        elif k == "esc":
            return "close", f, sel


def busy(msg, shape=(360, 640, 3)):
    ensure_window()
    img = np.full((max(200, shape[0] // 2), max(700, shape[1] // 2), 3), 28, np.uint8)
    cv2.imshow(WIN, put_lines(img, [(msg, (255, 255, 255))], 20, 20, 20))
    cv2.waitKey(1)
