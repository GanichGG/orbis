"""Модель StickPose (Orbis): силуэты (4 канала, 128x128: свой, чужие, свой в прошлом и следующем кадре)
-> 5 карт вероятности (64x64): голова, шея, таз, кисти, стопы + 6-я карта «скелет тела» (вспомогательная).
Небольшая U-Net, работает и на CPU, и на видеокарте. Веса v1 (2 канала -> 5 карт) тоже загружаются."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synth import S, HM, CH, CIN

NOUT = len(CH) + 1   # + карта костей


def block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class StickPose(nn.Module):
    VERSION = 2

    def __init__(self, w=32, cin=CIN, cout=NOUT):
        super().__init__()
        self.cfg = {"w": w, "cin": cin, "cout": cout}
        c = [w, w * 2, w * 3, w * 4, w * 5]
        # вход 128 -> сразу 64 (stride 2): на силуэтах мелкие детали 128 не нужны, а считать в 4 раза дешевле
        self.e0 = nn.Sequential(nn.Conv2d(cin, c[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c[0]),
                                nn.ReLU(inplace=True))  # 64
        self.e1 = block(c[0], c[1])     # 64
        self.e2 = block(c[1], c[2])     # 32
        self.e3 = block(c[2], c[3])     # 16
        self.e4 = block(c[3], c[4])     # 8 — видит всю фигуру целиком
        self.d3 = block(c[4] + c[3], c[3])
        self.d2 = block(c[3] + c[2], c[2])
        self.d1 = block(c[2] + c[1], c[1])
        self.out = nn.Conv2d(c[1], cout, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        d3 = self.d3(up(e4, e3))
        d2 = self.d2(up(d3, e2))
        d1 = self.d1(up(d2, e1))
        return self.out(d1)  # логиты, 64x64


def save(model, path, extra=None):
    torch.save({"cfg": model.cfg, "state": model.state_dict(), "version": StickPose.VERSION, "extra": extra or {}}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {"cin": 2, "cout": len(CH), **ck["cfg"]}   # у весов v1 в cfg могло не быть cin/cout
    m = StickPose(**cfg)
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def upgrade(old, cin=CIN, cout=NOUT):
    """Модель v1 (2 канала -> 5 карт) -> v2 с теми же знаниями: новые входы и выход костей начинают с нуля,
    так что сначала она отвечает как старая, а обучение учит её пользоваться соседними кадрами."""
    m = StickPose(w=old.cfg["w"], cin=cin, cout=cout).to(next(old.parameters()).device)
    sd = m.state_dict()
    for k, v in old.state_dict().items():
        t = sd[k]
        if t.shape == v.shape:
            t.copy_(v)
        elif k == "e0.0.weight":
            t.zero_()
            t[:, :v.shape[1]] = v
        elif k.startswith("out."):
            t.zero_() if k.endswith("weight") else t.fill_(-4.0)
            t[:v.shape[0]] = v
    m.load_state_dict(sd)
    return m.train(old.training)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(model, x, dev=None):
    """x (B,C,S,S) numpy (0..1 или uint8 0..255) -> вероятности (B,cout,HM,HM) numpy.
    Лишние каналы отбрасываются, недостающие соседние кадры заменяются текущим (как будто стоит на месте)."""
    dev = dev or next(model.parameters()).device
    x = fit_channels(x, model.cfg["cin"])
    t = torch.from_numpy(np.ascontiguousarray(x)).float().to(dev)
    if x.dtype == np.uint8:
        t = t / 255.0
    return torch.sigmoid(model(t)).cpu().numpy()


def fit_channels(x, cin):
    """(B,C,S,S) или (C,S,S) -> ровно cin каналов (для смеси моделей/данных v1 и v2)."""
    c = x.shape[-3]
    if c > cin:
        return x[..., :cin, :, :]
    if c < cin:
        extra = [x[..., :1, :, :]] * (cin - c)
        return np.concatenate([x] + extra, axis=-3)
    return x


def peaks(hm, k=4, thr=0.08):
    """Локальные максимумы одной карты -> [(x, y, score)] в пикселях входа S (с субпиксельным уточнением)."""
    H, W = hm.shape
    pad = np.pad(hm, 1, constant_values=0)
    nb = np.max(np.stack([pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]), 0)
    ys, xs = np.nonzero((hm >= nb) & (hm > thr))
    if len(xs) == 0:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        ys, xs = np.array([y]), np.array([x])
    order = np.argsort(-hm[ys, xs])[:k]
    out = []
    for i in order:
        y, x = ys[i], xs[i]
        y0, y1, x0, x1 = max(0, y - 1), min(H, y + 2), max(0, x - 1), min(W, x + 2)
        w = hm[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        sx = float((w * gx).sum() / (w.sum() + 1e-9))
        sy = float((w * gy).sum() / (w.sum() + 1e-9))
        sc = S / HM
        out.append(((sx + 0.5) * sc - 0.5, (sy + 0.5) * sc - 0.5, float(hm[y, x])))
    return out
