"""Обучение StickPose (Prismium) на синтетике (и, при наличии, на размеченных кадрах реальных видео).

    python train.py --steps 20000 --out prismium.pt              # с нуля
    python train.py --resume старые.pt --steps 10000 --out prismium.pt   # дообучить (веса v1 сами станут v2)
    python train.py --resume prismium.pt --out prismium.pt       # если обучение оборвалось — продолжит с того же шага
    python train.py --real labels_dir --steps 5000               # добавить реальные разметки (*.npz)

Сам использует видеокарту, если есть (CUDA). Каждые 200 шагов сохраняет веса и состояние обучения в --out.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import synth
from model import StickPose, save, load, device, upgrade, fit_channels


class SynthStream(torch.utils.data.IterableDataset):
    def __init__(self, seed=0, real=None, real_frac=0.0, pseudo=None, pseudo_frac=0.0):
        self.seed, self.real, self.real_frac = seed, real, real_frac
        self.pseudo, self.pseudo_frac = pseudo, pseudo_frac

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()))
        while True:
            r = rng.random()
            pool = (self.real if self.real is not None and r < self.real_frac else
                    self.pseudo if self.pseudo is not None and r < self.real_frac + self.pseudo_frac else None)
            if pool is not None:  # реальные разметки / уверенные кадры самообучения
                x, kp = augment_real(rng, *pool[rng.integers(len(pool))])
                bones = np.full((synth.HM, synth.HM), -1, np.float32)   # костей у реальных нет — не учить
            else:
                x, kp, bones = synth.make_sample(rng)
            y = np.concatenate([synth.heatmaps(kp), bones[None]])
            yield torch.from_numpy(x), torch.from_numpy(y)


def augment_real(rng, x, kp):
    """Лёгкая аугментация реального примера: поворот/масштаб/сдвиг/зеркало."""
    import cv2
    x = fit_channels(np.asarray(x), synth.CIN)
    x = x.astype(np.float32) / (255.0 if x.dtype == np.uint8 else 1.0)
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
    nk = len(synth.CH)
    pk, tk = p[:, :nk], target[:, :nk]
    mk = (tk >= 0).float()   # -1 — точка неизвестна (у кадров самообучения), не учим
    loss = (mk * (1.0 + 20.0 * tk) * (pk - tk) ** 2).sum() / (mk.sum() + 1.0)
    if logits.shape[1] > nk:  # карта костей (там, где она известна: у синтетики)
        pb, tb = p[:, nk], target[:, nk]
        m = (tb >= 0).float()
        loss = loss + 0.5 * (m * (1.0 + 5.0 * tb) * (pb - tb) ** 2).sum() / (m.sum() + 1.0)
    return loss


def train(model, steps, lr=2e-3, bs=32, real=None, real_frac=0.0, log_every=200, workers=None, ckpt=None,
          on_log=None, state=None, pseudo=None, pseudo_frac=0.0):
    """state — состояние оборванного обучения из чекпойнта (шаг, оптимизатор, расписание): продолжить с него."""
    dev = next(model.parameters()).device
    ds = SynthStream(real=real, real_frac=real_frac, pseudo=pseudo, pseudo_frac=pseudo_frac if pseudo else 0.0)
    if workers is None:  # синтетику рисует процессор — это узкое место, отдаём ему почти все потоки
        workers = max(1, min(10, (os.cpu_count() or 2) - 2))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, persistent_workers=workers > 0,
                                     pin_memory=dev.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    first = 1
    if state:
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        first = state["steps"] + 1
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.train()
    it = iter(dl)
    t0, run = time.time(), 0.0
    for step in range(first, steps + 1):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        x, y = x[:, :model.cfg["cin"]], y[:, :model.cfg["cout"]]   # модель v1 (дообучение в программе) берёт меньше
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % log_every == 0 or step == steps:
            msg = f"шаг {step}/{steps}  loss {run:.4f}  {(time.time() - t0) / (step - first + 1):.2f} с/шаг"
            (on_log or print)(msg)
            if ckpt and step < steps:
                save(model, ckpt, {"steps": step, "total": steps, "lr": lr, "bs": bs,
                                   "opt": opt.state_dict(), "sched": sched.state_dict()})
    model.eval()
    return model


def load_real(d):
    out = []
    if d and os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".npz"):
                z = np.load(os.path.join(d, fn))
                x = z["x"] if z["x"].dtype == np.uint8 else z["x"].astype(np.float32)
                kp = z["kp"].astype(np.float32)
                if kp.shape[1] == 7:  # старые разметки: локти/колени — середины рук/ног
                    mid = lambda a, b: (kp[:, a] + kp[:, b]) / 2
                    kp = np.concatenate([kp, np.stack([mid(1, 3), mid(1, 4), mid(2, 5), mid(2, 6)], 1)], 1)
                out += list(zip(x, kp))
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
    ap.add_argument("--pseudo", default=None, help="папка с кадрами самообучения (tools/selftrain.py)")
    ap.add_argument("--pseudo_frac", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    dev = device()
    print("устройство:", dev)
    state = None
    if a.resume:
        model = load(a.resume, dev).train()
        if model.cfg["cin"] != synth.CIN or model.cfg["cout"] != len(synth.CH) + 1:
            print("веса v1 -> Prismium v2 (соседние кадры + карта костей), знания старой модели сохраняются")
            model = upgrade(model).train()
        ex = torch.load(a.resume, map_location="cpu", weights_only=False).get("extra", {})
        if "opt" in ex and ex["steps"] < ex["total"]:
            state = ex
            a.steps, a.lr, a.bs = ex["total"], ex["lr"], ex["bs"]
            print(f"продолжаю оборванное обучение с шага {ex['steps']} из {ex['total']}")
    else:
        model = StickPose(w=a.width).to(dev)
    print("параметров:", sum(p.numel() for p in model.parameters()))
    real = load_real(a.real)
    if real:
        print("реальных примеров:", len(real))
    pseudo = load_real(a.pseudo)
    if pseudo:
        print("кадров самообучения:", len(pseudo))
    train(model, a.steps, a.lr, a.bs, real, a.real_frac if real else 0.0, workers=a.workers, ckpt=a.out, state=state,
          pseudo=pseudo, pseudo_frac=a.pseudo_frac)
    save(model, a.out, {"steps": a.steps})
    print("сохранено:", a.out)


if __name__ == "__main__":
    main()
