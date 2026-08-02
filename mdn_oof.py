"""Cross-fitted MDN predictions, to be consumed by the GBM as features.

On its own the mixture-density inverter loses to the GBM (14.05 vs 8.76): its
modes contain good trajectories but the probability head cannot pick them, and
"best of 8 modes" is an optimistically biased statistic, not skill. What it does
carry is an independent read of the GR/typewell correlation image, so the useful
move is to hand the GBM the mode summary and let it decide how far to trust it --
the same pattern that worked for the structural surface.

Folds are over WELLS, so a well's features never come from a model that saw it.
mdn_train.pkl comes from our own mdn_data.py, so unpickling is safe.
"""
import numpy as np, pickle, torch, time, json
from sklearn.model_selection import GroupKFold
import mdn_train as T

EPOCHS = 45


def train_fold(ds, tr_idx, Xc, Xm, Y):
    model = T.MDN().to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=EPOCHS * max(1, len(tr_idx) // 64 + 1))
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            traj, logit = model(Xc[b].to(T.DEV), Xm[b].to(T.DEV))
            loss, _, _ = T.mtp_loss(traj, logit, Y[b].to(T.DEV))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
    return model


def main():
    ds = pickle.load(open('mdn_train.pkl', 'rb'))
    wids = np.array([d['wid'] for d in ds])
    real = np.array([d['is_real'] for d in ds])
    Xc = torch.from_numpy(np.stack([d['ch'] for d in ds]))
    Xm = torch.from_numpy(np.stack([d['meta'] for d in ds]))
    Y = torch.from_numpy(np.stack([d['y'] for d in ds]))
    print(f'samples={len(ds)} wells={len(np.unique(wids))} dev={T.DEV}', flush=True)

    out = {}
    t0 = time.time()
    for f, (tr, va) in enumerate(GroupKFold(5).split(np.zeros(len(ds)), None, wids)):
        model = train_fold(ds, tr, Xc, Xm, Y)
        model.eval()
        va_real = [i for i in va if real[i]]
        with torch.no_grad():
            for i in range(0, len(va_real), 256):
                b = va_real[i:i + 256]
                traj, logit = model(Xc[b].to(T.DEV), Xm[b].to(T.DEV))
                p = torch.softmax(logit, 1)
                mean = (traj * p[:, :, None]).sum(1).cpu().numpy()
                top = traj.gather(1, p.argmax(1)[:, None, None]
                                  .expand(-1, 1, traj.shape[2])).squeeze(1).cpu().numpy()
                spread = traj.std(1).cpu().numpy()
                ent = (-(p * torch.log(p + 1e-9)).sum(1)).cpu().numpy()
                for n, j in enumerate(b):
                    out[ds[j]['wid']] = dict(mean=mean[n].astype(np.float32),
                                             top=top[n].astype(np.float32),
                                             spread=spread[n].astype(np.float32),
                                             ent=float(ent[n]),
                                             bmd=ds[j]['bmd'], ev=ds[j]['ev'],
                                             t0=ds[j]['t0'])
        # honest per-fold score on the real mask
        yv = Y[va_real].numpy()
        mv = np.stack([out[ds[j]['wid']]['mean'] for j in va_real])
        print(f'  fold{f}: wells={len(va_real)} mean-RMSE={np.sqrt(np.mean((mv-yv)**2)):.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    with open('mdn_oof.pkl', 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    print(f'wrote mdn_oof.pkl for {len(out)} wells')


if __name__ == '__main__':
    main()
