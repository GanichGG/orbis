"""Обучение StickPose на синтетике (и, при наличии, на размеченных кадрах реальных видео).

    python train.py --steps 20000 --out stickpose.pt            # с нуля
    python train.py --resume stickpose.pt --steps 10000          # продолжить
    python train.py --real labels_dir --steps 5000               # добавить реальные разметки (*.npz)

Сам использует видеокарту, если есть (CUDA).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import synth
from model import StickPose, save, load, device


class SynthStream(torch.utils.data.IterableDataset):
    def __init__(self, seed=0, real=None, real_frac=0.0):
        self.seed, self.real, self.real_frac = seed, real, real_frac

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()))
        while True:
            if self.real is not None and rng.random() < self.real_frac:
                x, kp = augment_real(rng, *self.real[rng.integers(len(self.real))])
            else:
                x, kp = synth.make_sample(rng)
            yield torch.from_numpy(x), torch.from_numpy(synth.heatmaps(kp))


def augment_real(rng, x, kp):
    """Лёгкая аугментация реального примера: поворот/масштаб/сдвиг/зеркало."""
    import cv2
    a = rng.normal(0, 12)
    s = rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((synth.S / 2, synth.S / 2), a, s)
    M[:, 2] += rng.normal(0, 4, 2)
    out = np.stack([cv2.warpAffine(c, M, (synth.S, synth.S), flags=cv2.INTER_LINEAR) for c in x])
    k = kp @ M[:, :2].T + M[:, 2]
    if rng.random() < 0.5:
        out = out[:, :, ::-1].copy()
        k[:, 0] = synth.S - 1 - k[:, 0]
    return out.astype(np.float32), k.astype(np.float32)


def loss_fn(logits, target):
    p = torch.sigmoid(logits)
    w = 1.0 + 20.0 * target
    return (w * (p - target) ** 2).mean()


def train(model, steps, lr=2e-3, bs=32, real=None, real_frac=0.0, log_every=200, workers=None, ckpt=None,
          on_log=None):
    dev = next(model.parameters()).device
    ds = SynthStream(real=real, real_frac=real_frac)
    if workers is None:
        workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, persistent_workers=workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    it = iter(dl)
    t0, run = time.time(), 0.0
    for step in range(1, steps + 1):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % log_every == 0 or step == steps:
            msg = f"шаг {step}/{steps}  loss {run:.4f}  {(time.time() - t0) / step:.2f} с/шаг"
            (on_log or print)(msg)
            if ckpt:
                save(model, ckpt, {"steps": step})
    model.eval()
    return model


def load_real(d):
    out = []
    if d and os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".npz"):
                z = np.load(os.path.join(d, fn))
                out += list(zip(z["x"].astype(np.float32), z["kp"].astype(np.float32)))
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="stickpose.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--real", default=None, help="папка с реальными разметками *.npz")
    ap.add_argument("--real_frac", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    dev = device()
    print("устройство:", dev)
    model = load(a.resume, dev).train() if a.resume else StickPose(w=a.width).to(dev)
    print("параметров:", sum(p.numel() for p in model.parameters()))
    real = load_real(a.real)
    if real:
        print("реальных примеров:", len(real))
    train(model, a.steps, a.lr, a.bs, real, a.real_frac if real else 0.0, workers=a.workers, ckpt=a.out)
    save(model, a.out, {"steps": a.steps})
    print("сохранено:", a.out)


if __name__ == "__main__":
    main()
