# rogii-wellbore-geology-prediction

My entry for the Kaggle "ROGII — Wellbore Geology Prediction" competition:
predict TVT (true vertical thickness to a stratigraphic marker) over the
unknown zone of horizontal wells, given the known prefix (GR log + TVT) and
gamma ray on the whole trajectory.

Final leaderboard: **7.932 RMSE** (down from an 8.407 first submission).
Full writeup, what worked, what didn't, and why: [SOLUTION.md](SOLUTION.md).

## Approach, short version

- `TVT = anchor + E(X,Y) - Z`, so `U = TVT + Z` is the underlying
  stratigraphic surface — a lot of the gains came from doing post-processing
  in `U` instead of raw `TVT`.
- Main model: 5-model stack (LightGBM x3 variants + CatBoost x2) ->
  positive-Ridge meta, `GroupKFold` by well so no leakage between train/val.
- A likelihood-weighted particle filter over the GR log is blended in as a
  second, decorrelated estimator (`proj.py`, `legbuild.py`). It's *worse*
  standalone than the learned model but drags the blend down because it's
  wrong in different places — the point that took longest to actually get.
- Everything is validated on two disjoint halves of the wells before it's
  allowed anywhere near a submission (`pfcv.py`). Burned a chunk of budget
  early trusting a single-slice CV number that didn't hold up.
- It's a Kaggle **code competition**: the model runs inside a pinned kernel
  (`kernel_inferB*/full.py`), not a plain CSV upload — 3 visible wells vs
  ~200 hidden at inference means wall-clock scales ~60x, which capped how
  much PF ensembling was actually affordable.

## Layout

- `kernel_inferB*/` — the kernel pushed to Kaggle for each scored submission,
  numbered by iteration. `full.py` is the actual inference script.
- `retrain*.py` — stack training scripts (the model behind each iteration).
- `pfcv.py`, `gseval.py`, `retune.py` — the validation harnesses everything
  got checked against before touching a submission.
- Root `.py` files not under a `kernel_*` dir are earlier pipeline versions
  and dead ends kept for reference (see SOLUTION.md for which).

Competition data and trained model binaries are not included (data isn't
mine to redistribute, models are multi-hundred-MB and regenerate from the
scripts + `data/` if you have competition access).
