"""Mixture-density inverter for stratigraphic position.

Every direct attack on this problem failed for the same measured reason: the
GR-vs-typewell correlation surface is multi-modal, and a free search finds
trajectories cheaper than the truth (8.14 vs 8.44) at RMSE 12.5. So the model
must not optimise that surface -- it must learn, from 773 labelled wells, which
of the competing dark bands is geologically plausible.

Architecture follows Alyaev & Elsheikh (2022): a CNN over the correlation image
emits M candidate trajectories plus their logits, trained with the
multiple-trajectory-prediction loss so only the best-matching mode receives the
regression gradient (which is what stops the modes collapsing onto their mean).

Scoring uses the same pooled well-level RMSE as the rest of the project, so the
number is directly comparable with the GBM pipeline (currently CV 9.167).
mdn_train.pkl is written by mdn_data.py, our own code, so unpickling is safe.
"""
import numpy as np, pickle, time, sys, json
import torch, torch.nn as nn, torch.nn.functional as F

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
K, L = 96, 96
M_MODES = 8
ALPHA_CLS = 1.0


class MDN(nn.Module):
    def __init__(self, n_meta=6, m=M_MODES, l=L):
        super().__init__()
        self.m, self.l = m, l
        ch = (3, 32, 64, 96, 128)
        blocks = []
        for i in range(4):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.GELU(),
                       nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.GELU(),
                       nn.MaxPool2d(2)]
        self.enc = nn.Sequential(*blocks)          # -> (128, 6, 6)
        # collapse the stratigraphic axis, keep the along-well axis
        self.head_pool = nn.AdaptiveAvgPool2d((1, 6))
        feat = 128 * 6
        self.mlp = nn.Sequential(
            nn.Linear(feat + n_meta, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1))
        self.traj = nn.Linear(512, m * l)
        self.logit = nn.Linear(512, m)

    def forward(self, x, meta):
        h = self.enc(x)
        h = self.head_pool(h).flatten(1)
        h = self.mlp(torch.cat([h, meta], 1))
        return self.traj(h).view(-1, self.m, self.l), self.logit(h)


def mtp_loss(traj, logit, y, alpha=ALPHA_CLS):
    """Only the closest mode is pulled toward the target; the rest keep their
    diversity. The logits are trained to pick that mode."""
    d = (traj - y[:, None, :]).abs().mean(2)          # (B, M) 1-norm per mode
    best = d.argmin(1)
    reg = d.gather(1, best[:, None]).mean()
    cls = F.cross_entropy(logit, best)
    return reg + alpha * cls, reg.item(), cls.item()


def predict(traj, logit, mode='mean'):
    p = torch.softmax(logit, 1)
    if mode == 'top':
        return traj.gather(1, p.argmax(1)[:, None, None].expand(-1, 1, traj.shape[2])).squeeze(1)
    return (traj * p[:, :, None]).sum(1)              # RMSE-optimal point estimate


def main():
    with open('mdn_train.pkl', 'rb') as f:
        ds = pickle.load(f)
    print(f'samples={len(ds)} device={DEV}', flush=True)

    wids = np.array([d['wid'] for d in ds])
    uw = np.unique(wids)
    rng = np.random.RandomState(0)
    val_w = set(uw[rng.permutation(len(uw))[:len(uw) // 5]])
    tr = [i for i, w in enumerate(wids) if w not in val_w]
    # validate only on the organiser's own mask, which is what the metric uses
    va = [i for i, w in enumerate(wids) if w in val_w and ds[i].get('is_real', True)]
    print(f'train={len(tr)} val={len(va)} (val wells={len(val_w)})', flush=True)

    Xc = torch.from_numpy(np.stack([d['ch'] for d in ds]))
    Xm = torch.from_numpy(np.stack([d['meta'] for d in ds]))
    Y = torch.from_numpy(np.stack([d['y'] for d in ds]))

    model = MDN().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=EPOCHS * max(1, len(tr) // 64 + 1))
    best = 1e9
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tot = n = 0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xc, xm, y = Xc[b].to(DEV), Xm[b].to(DEV), Y[b].to(DEV)
            traj, logit = model(xc, xm)
            loss, reg, cls = mtp_loss(traj, logit, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            tot += reg * len(b); n += len(b)
        model.eval()
        with torch.no_grad():
            errs_mean, errs_top, errs_best = [], [], []
            for i in range(0, len(va), 256):
                b = va[i:i + 256]
                xc, xm, y = Xc[b].to(DEV), Xm[b].to(DEV), Y[b].to(DEV)
                traj, logit = model(xc, xm)
                errs_mean.append(((predict(traj, logit) - y) ** 2).mean(1).cpu())
                errs_top.append(((predict(traj, logit, 'top') - y) ** 2).mean(1).cpu())
                errs_best.append(((traj - y[:, None, :]) ** 2).mean(2).min(1).values.cpu())
            rm = torch.cat(errs_mean).mean().sqrt().item()
            rt = torch.cat(errs_top).mean().sqrt().item()
            rb = torch.cat(errs_best).mean().sqrt().item()
        if rm < best:
            best = rm
            torch.save(model.state_dict(), 'mdn_best.pt')
        print(f'ep{ep:3d} trainMAE={tot/max(n,1):.3f}  val RMSE mean={rm:.3f} top={rt:.3f} '
              f'oracle-mode={rb:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    print(f'BEST val pooled-ish RMSE = {best:.3f}')
    print('(compare: hold-last baseline ~15.9, GBM pipeline CV 9.167, oracle line 6.75)')


if __name__ == '__main__':
    main()
