"""Выделение персонажей по цвету «пипеткой».

Цвет персонажа берётся кликами (можно несколько оттенков). В каждом кадре заново строится
палитра фона (k-means), и пиксель отдаётся персонажу, только если он ближе к цвету персонажа,
чем к любому цвету фона. Поэтому работает при движущейся камере, текстуре и затемнении.
Если у двух персонажей одинаковый цвет — делит их по положению в прошлом кадре.
"""
import cv2
import numpy as np


def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def sample_color(frame, x, y, r=3):
    """Цвет под пипеткой: медиана маленького окна (сглаживает края и шум сжатия)."""
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = max(0, int(x) - r), min(w, int(x) + r + 1), max(0, int(y) - r), min(h, int(y) + r + 1)
    return np.median(to_lab(frame[y0:y1, x0:x1]).reshape(-1, 3), axis=0)


def min_dist(lab, centers):
    """Расстояние каждого пикселя до ближайшего из centers (через матричное умножение — быстро)."""
    h, w = lab.shape[:2]
    x = lab.reshape(-1, 3)
    c = np.asarray(centers, np.float32).reshape(-1, 3)
    d2 = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]
    return np.sqrt(np.maximum(d2.min(1), 0)).reshape(h, w)


class ColorSegmenter:
    def __init__(self, colors, max_dist=38.0, k_bg=10):
        """colors: список (по персонажам) списков Lab-цветов."""
        self.colors = [np.array(c, np.float32).reshape(-1, 3) for c in colors]
        self.base = [c.copy() for c in self.colors]
        self.n = len(self.colors)
        self.max_dist = max_dist
        self.k_bg = k_bg
        self.prev = [None] * self.n
        self.size = [None] * self.n
        # у каких персонажей похожие цвета (тогда нужен пространственный разбор)
        self.twin = np.zeros((self.n, self.n), bool)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    d = np.min(np.linalg.norm(self.colors[i][:, None] - self.colors[j][None], axis=2))
                    self.twin[i, j] = d < 20

    def _bg_palette(self, lab, dchar_min):
        px = lab.reshape(-1, 3)
        dm = dchar_min.reshape(-1)
        cand = px[dm > 20]
        if len(cand) < 50:
            return np.zeros((0, 3), np.float32)
        rng = np.random.default_rng(0)
        sel = cand[rng.integers(0, len(cand), min(6000, len(cand)))]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(self.k_bg, len(sel))
        _, _, centers = cv2.kmeans(sel, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        return centers

    def dists(self, frame):
        lab = to_lab(frame)
        dc = np.stack([min_dist(lab, c) for c in self.colors])
        bg = self._bg_palette(lab, dc.min(0))
        if len(bg):
            db = min_dist(lab, bg)
        else:
            db = np.full(lab.shape[:2], 1e3, np.float32)
        self._lab = lab
        return dc, db

    def step(self, frame, init_points=None):
        """-> список масок (bool) по персонажам. init_points — клики (для первого кадра)."""
        dc, db = self.dists(frame)
        H, W = db.shape
        masks = []
        for i in range(self.n):
            m = (dc[i] < self.max_dist) & (dc[i] < 0.8 * db)
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
            masks.append(m)
        # близнецы по цвету: делим пиксели по близости к прошлым маскам
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.twin[i, j]:
                    both = masks[i] & masks[j]
                    if both.any() and self.prev[i] is not None and self.prev[j] is not None:
                        di = cv2.distanceTransform((~self.prev[i]).astype(np.uint8), cv2.DIST_L2, 3)
                        dj = cv2.distanceTransform((~self.prev[j]).astype(np.uint8), cv2.DIST_L2, 3)
                        masks[i] = masks[i] & ~(both & (dj < di))
                        masks[j] = masks[j] & ~(both & (di <= dj))
        # оставить связные куски рядом с персонажем (отсечь частицы и посторонние объекты того же цвета)
        out = []
        for i, m in enumerate(masks):
            nl, comp, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            if nl <= 1:
                out.append(m)
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            if init_points is not None and init_points[i] is not None and m.any():
                x, y = init_points[i]
                ys, xs = np.nonzero(m)
                k = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                seed = comp[ys[k], xs[k]]
                ref = comp == seed
            elif self.prev[i] is not None and self.prev[i].any():
                ref = self.prev[i]
            else:
                ref = comp == (1 + int(np.argmax(areas)))
            ys, xs = np.nonzero(ref)
            size = max(xs.max() - xs.min(), ys.max() - ys.min(), 8)
            near = cv2.dilate(ref.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=1)
            dist = cv2.distanceTransform((near == 0).astype(np.uint8), cv2.DIST_L2, 3)
            from scipy import ndimage as ndi
            big = areas.max()
            idx = np.arange(1, nl)
            dmin = np.asarray(ndi.minimum(dist, comp, idx))
            ar = stats[1:, cv2.CC_STAT_AREA]
            # кусок рядом с прошлым положением, либо крупный (камера резко сместилась)
            good = ((dmin < 0.6 * size) | (ar > 0.5 * big)) & (ar >= max(12, 0.01 * big))
            keep = idx[good].tolist()
            mm = np.isin(comp, keep) if keep else (comp == (1 + int(np.argmax(areas))))
            out.append(mm)
        self.prev = [m.copy() if m.sum() > 20 else p for m, p in zip(out, self.prev)]
        # подстройка цвета под общий тон кадра (вспышки, цветокоррекция): плавно, в пределах разумного
        for i, m in enumerate(out):
            if m.sum() > 50:
                med = np.median(self._lab[m], axis=0)
                if np.min(np.linalg.norm(self.base[i] - med, axis=1)) < 30:
                    self.colors[i] = np.vstack([self.base[i], med[None]])
        return out


def crop_input(mask_self, mask_other, S=128, margin=1.25, box=None):
    """Маски кадра -> вход модели (2,S,S) и аффинное преобразование кадр->кроп."""
    if box is None:
        ys, xs = np.nonzero(mask_self)
        if len(xs) < 5:
            return None, None
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * margin + 4
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sc = S / side
    M = np.array([[sc, 0, S / 2 - cx * sc], [0, sc, S / 2 - cy * sc]], np.float32)
    a = cv2.warpAffine(mask_self.astype(np.uint8) * 255, M, (S, S), flags=cv2.INTER_AREA)
    b = cv2.warpAffine(mask_other.astype(np.uint8) * 255, M, (S, S), flags=cv2.INTER_AREA)
    return np.stack([a, b]).astype(np.float32) / 255.0, M


def crop_to_frame(pts, M):
    pts = np.asarray(pts, float)
    sc = M[0, 0]
    return (pts - M[:, 2]) / sc


def frame_to_crop(pts, M):
    pts = np.asarray(pts, float)
    return pts * M[0, 0] + M[:, 2]
