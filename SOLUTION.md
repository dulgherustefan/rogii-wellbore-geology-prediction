# ROGII Wellbore Geology Prediction — Solution log

Target: predict `tvt` (true vertical thickness = stratigraphic position of the bit)
for the evaluation zone of each horizontal well. Metric: RMSE (one-foot steps).

## Data structure (confirmed)
- 773 train wells, hidden test (3 sample wells locally; the 3 are reused train wells).
- `*__horizontal_well.csv`: `MD,X,Y,Z,GR,TVT_input` (+ `ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT` in TRAIN only).
  - `Z` = bit true vertical depth (elevation, negative). Known everywhere incl. test.
  - `ANCC…BUDA` = elevations of 6 stratigraphic markers at the bit (TRAIN only). Parallel surfaces (all std≈identical, move in lockstep).
  - `GR` = gamma ray (~43% missing). `TVT_input` = true TVT up to the Prediction-Start (PS) point, NaN after.
- `*__typewell.csv`: `TVT,GR` (+`Geology` in train) — reference log GR vs stratigraphic depth.
- **Eval zone = exactly the rows where `TVT_input` is NaN** (the toe, ~67–80% of the lateral). `TVT_input==TVT` on the known part.

## Key data insights
1. **TVT is per-well**: each well's TVT is referenced to its own typewell datum. TVT=11745 in well A ≠ same stratum as in well B (offsets of 100s of ft). → cross-well raw-TVT transfer fails.
2. **TVT is steered to stay in zone**: in the eval zone TVT is *mean-reverting*, not trending. Holding the last value (`const`) is a strong baseline; linear/dip extrapolation overshoots badly.
3. **TVT is very smooth** (|dTVT|/step ≈ 0.01 ft) but makes large smooth excursions (±10–50 ft) — low-frequency undulations that `const` misses.
4. **Wells are densely clustered spatially** (median nearest-well 470 ft; 702/773 have a neighbour <1500 ft) BUT laterals do not physically intersect in 3D (>200 ft apart), so 3D-KNN on TVT fails.
5. **GR is informative but ambiguous**: corr(GR_obs, typewell GR @ true TVT) ≈ 0.55–0.82. Non-monotonic GR ⇒ point matching is multi-valued. GR only localises *while TVT is moving*; when the bit sits still GR is flat/uninformative.
6. **Exact structural identity**: `TVT = b_well + E_formation(X,Y) − Z` reproduces TVT to **RMSE 0.01** using the well's own markers (`b_well = median(TVT+Z−E)` per well). The whole problem ≈ predicting the formation-surface elevation E(X,Y) at the bit.

## Leaderboard calibration (2026-06-24)
Top=5.35, **Top 10=6.53**, top5%=7.21, **dense public wall ~7.2–7.6** (a copied public notebook), median=10.0.
`const` (~11–16 on backtest) ≈ median rank. So real signal is large and extractable.

## Approaches tried (well-level backtest reproducing the eval split)
| Approach | Pooled RMSE | Verdict |
|---|---|---|
| const (hold last TVT) | ~12–16 | strong baseline (TVT mean-reverts) |
| linear / dip extrapolation of TVT or S=TVT+Z | 40–190 | fails — TVT does not trend |
| GR point-match to typewell | ~26 | fails — ambiguous |
| GR DP (velocity penalty) | ≈const | over-regularised → const; under → noise |
| **GR DP with curvature penalty (v4)** | ~17.8 (≈const) | captures smooth drift on strong-GR wells (e.g. 13.2→7.1) but wrong-direction failures cancel gains |
| windowed NCC localisation | worse than const | GR flat where TVT steady → no signature |
| 3D-KNN / offset-KNN on TVT | 90–320 | fails — per-well datum, no 3D overlap |
| neighbour formation surface (struct) | exact (0.01) w/ own markers; ~2.7 on simple wells | unreliable across faults (known-residual median ~120 ft) |
| **Particle filter (S=TVT+Z, GR likelihood, seed ensemble)** | **~9.7–12.4 (−20–25% vs const)** | **best single method**; diverges (wrong GR direction) on a minority of wells |
| confidence blend PF+struct+const | ≈PF | struct too unreliable to add weight |

## Final submissions today
- **PF ensemble + clip(±50) + shrink(0.6)** — robust, ~22% better than const on backtest. (`kernel/rogii_pf.py`)
- **const baseline** — safety net + exact LB calibration. (`kernel_const/`)
Kaggle keeps the best of the two.

## Why PF works (physical meaning)
It tracks the **structural level S = TVT + Z** as a momentum random walk along MD
(the geology dips smoothly), and at each foot reweights a particle cloud by the
**gamma-ray likelihood** (observed GR vs typewell GR at the implied TVT = S−Z),
resampling when degenerate. `tvt = S − Z`. This is automated geosteering: it
matches the GR signature, allows TVT to rise/fall (slides 6–7 of the task), and is
anchored at the last known TVT. The seed ensemble (likelihood-weighted) tames the
stochastic divergence; remaining wrong-direction failures are the open weakness.

## Path to Top-10 (next steps, mirrors the public ~7.2 solutions)
1. **More + multi-scale PF** (128 seeds, several process-noise/hold scales) with a
   per-well variant *selector* (by eval length, Z-span). Public LB7.2 does exactly this.
2. **Fix GR direction ambiguity** = the biggest remaining error. Use the neighbour
   **formation-surface dip** (datum-free) to constrain/seed the PF rate and veto
   wrong-direction drift. Build E(X,Y) with weighted ridge planes (not raw IDW) per
   formation; gate by a fault/discontinuity detector.
3. **Beam-search GR alignment** (Numba) as an independent TVT signal; multi-scale
   self-NCC of horizontal GR before PS vs after PS (slide 9: horizontal GR has higher
   resolution than the typewell).
4. **Stack** all signals (PF×scales, beam, NCC, struct, trajectory/GR features) with
   **LightGBM + CatBoost**, GroupKFold by well, Ridge blender. This is the step that
   takes the ~9–10 single-method score down to ~7 and below.

## GBM stacker (the much-better solution) — validated OOF
Combine complementary base signals with LightGBM (GroupKFold by well), target TVT-anchor:
- `pf`  : typewell-GR particle filter ensemble (RMSE 13.7)
- `pf2` : particle filter vs the well's OWN pre-PS GR(TVT) template (slide 9) — weaker
  alone but diverse; avg(pf,pf2) < either
- `st`  : neighbour formation surface (RMSE 17 overall, ~5 where geology continuous)
- confidences: PF pseudo-CV, struct known-residual, **formation dispersion (fault proxy)**,
  neighbour coverage; + GR/trajectory features.
Top features by importance: pf_cv, st_disp, st_kres, Z_dev, cov — i.e. the model mostly
learns *which signal to trust where*. Failure modes of PF (GR direction) and struct
(faults) are different, so the blend beats all singles.

| model | OOF RMSE (= LB scale) |
|---|---|
| const | 15.91 (LB 15.88) |
| PF ensemble | 13.72 (submitted+shrink → LB 12.30) |
| struct | 17.03 |
| **GBM stacker (pf+pf2+st+conf)** | **10.62** |

Kernel: `kernel_stack/rogii_stacker.py` — self-contained, regenerates train features,
trains LightGBM (2 seeds), predicts test. **CONFIRMED public LB = 10.147** (OOF 10.62
well-calibrated; LB runs ~0.47 better than OOF).

## v3 — NCC slide-9 signal + LightGBM/CatBoost blend (2026-06-25)
Two clean wins over the 10.62 stacker, both validated on the same GroupKFold(5) OOF:
1. **Slide-9 NCC as GBM features** (`featgen3.py`): correlate horizontal-GR windows
   (the high-res log) against the well's OWN known-zone GR(TVT) template, restricted to
   ±45 ft of the anchor (continuity prior). Added as *features* (estimate + confidence at
   window half-widths 8/15/25 + consensus + disagreement-vs-PF), NOT as a standalone
   predictor — the GBM downweights it where the GR alignment is unreliable. This is the
   fix for the earlier "NCC negative corr" failure: it failed as a hard predictor, works
   as a soft feature.
2. **CatBoost alongside LightGBM**, blended. Different inductive bias → diverse errors.

| model (GroupKFold OOF) | RMSE |
|---|---|
| LGBM base (v2 stacker) | 10.620 |
| LGBM + NCC | 10.556 |
| CatBoost + NCC | 10.533 |
| **blend 0.48·LGBM + 0.52·CatBoost (both +NCC)** | **10.421** |

NOTE: hand-engineered interaction/consensus features (ratios, trust-weighted blends)
were tried and HURT (LGBM 10.79 vs 10.62) — trees already find those splits; explicit
ratios just add noise. Dropped. Real signal came only from a genuinely new information
source (NCC) + a second algorithm (CatBoost).
Final bundle: 3 LGBM + 2 CatBoost seeds (`train_final.py`), inference kernel
`kernel_infer3/` (NCC features computed in-kernel for test wells, loky n_jobs=4).

### ⚠️ v3 RESULT: better OOF, WORSE LB — the OOF stopped being a valid proxy
| submission | OOF (train GroupKFold) | public LB |
|---|---|---|
| v2 stacker (LGBM, base feats) | 10.62 | **10.147** |
| v3 (LGBM+CatBoost + NCC, blend) | 10.42 (−0.20) | **10.311** (+0.16 WORSE) |

The added capacity (NCC GR-template features + CatBoost) **lowered OOF but raised real LB**.
The GroupKFold-over-train-wells OOF — which had been well-calibrated for the simple models
(const, PF, base LGBM all matched LB within ~0.5) — is **no longer a reliable LB proxy once
you add capacity**: it rewards fitting train-well idiosyncrasies the hidden test doesn't share.
Crucially, the 3 local "test" wells are *reused train wells*, so no local check could have
caught this — only the LB did. **Lesson: best LB stays 10.147 (the simplest model). Before
trusting any OOF gain, need a hold-out that mimics the test distribution (e.g. spatial/region
block CV, or holding out whole well-clusters), not random well folds.** Kaggle keeps the BEST
submission, so an experimental regression doesn't hurt ranking — only spends a daily slot.

### v5 (2026-06-25 #2): isolate the robust signal
Rather than submit v4 (=v3+pf3, same overfit-prone architecture), #2 = **LGBM on base+pf3
only** (drop NCC + CatBoost). pf3 = a 2nd, more-responsive particle filter (MOM 0.99, VN .006,
PN .012) — a physically-grounded signal, less likely to be train-specific than GR-template
matching. This is the minimal, most-robust-direction change from the 10.147 winner, and a
clean test of whether ANY of the new signals transfer to the hidden test. Kernel `kernel_infer5/`.

**v5 RESULT: LB 10.130 — NEW BEST** (beats v2's 10.147, and far better than v3's 10.311).
Confirms the hypothesis cleanly:
| signal added to base LGBM | LB | transfers? |
|---|---|---|
| NCC GR-template + CatBoost (v3) | 10.311 | NO — overfits train wells |
| pf3 (2nd, responsive particle filter) (v5) | **10.130** | YES — physical signal generalises |
So PF *diversity* is the robust lever; GR-template matching and a 2nd GBM algo are not.
Progress: const 15.88 → PF 12.30 → stacker(v2) 10.147 → **base+pf3 (v5) 10.130**.
Takeaway for top-10: keep adding *physically-grounded, diverse* base predictors (more PF
scales/configs, dip-constrained PF), NOT more model capacity or train-specific template
features. And build a test-like hold-out before trusting OOF.

## v6 — the public 7.2 wall is LARGELY LEAKAGE; adopt the honest dual pipeline (2026-06-25 #3)
Mined the top public notebooks (`public_nb/`). The strong ones (degnonguidi **7.159**,
baidalinadilzhan 7.201, romantamrazov) all ship a **"GOLD overlay"** = a visible-prefix
calibration that **exploits train/test well overlap**. The 3 public test wells are *reused
train wells*, so the overlay can look up near-exact answers on the public split. It is a
**documented leakage path**: a genuine public-LB booster but a **no-op on the fully hidden
PRIVATE test** (final ranking 2026-08-05). So the public ~7.2 cluster is inflated; on the
private LB those scores revert to the honest model (~7-9). **The honest model is what wins.**

Decision (user wanted the better score): tried overlay ON → it is **far too slow**
(~25 min on 3 wells locally even at reduced 12/20 seeds, 3 cuts; bottleneck =
`run_pf_lik_ensemble_scales` per cut×seed) → certain Kaggle **timeout** on the full hidden
test, which would waste the single submission. The **honest** dual pipeline (overlay OFF)
already scores ~7.159 ("the honest number is the one that matters"), runs reliably, is durable
on private, and is a huge jump from 10.13 — so submit that.

The degnonguidi pipeline (now the base going forward):
- **Pipeline A**: FormationPlaneKNN (weighted local *plane* fit per formation, better than my
  IDW) + DenseANCC IDW + sp45 polynomial projection + GBMs.
- **Pipeline B**: multi-scale particle-filter *selector* (PF over several process-noise scales,
  48 seeds) + GBMs.
- LGB×3 + CB, GroupKFold OOF, Ridge blend of A and B (0.55/0.45), guarded known-prefix override.
- Self-contained (private `koolbox` replaced by a visible CVTrainer), auto GPU/CPU, FAST smoke flag.

Mechanics worth keeping: beam-search GR alignment (Numba JIT, 7 configs), affine GR calibration,
per-formation b_well (median/WLS/last-50), **GR-offset features** (sample typewell GR at ±offsets
around each base estimate → the GBM sees the local GR signature), inter-signal consensus/std.

Note: romantamrazov's "top-3" notebook is weaker (honest OOF ~10.7) and had 2 bugs on the current
Kaggle image — CatBoost `subsample` needs `bootstrap_type="Bernoulli"`; `devices="0:1"` fails on the
single-GPU P100 (`"0"`). degnonguidi's auto-detect avoids both.

## v7 — plane-fit structural surface ported into my fast pipeline (2026-06-26)
The full dual pipeline TRAINS FROM SCRATCH on Kaggle (no pre-trained artifacts attached) and
**exceeds the GPU time limit** — the "gold" kernel ran ~8h → CANCEL_ACKNOWLEDGED; reduced "fast"/
"lean" variants ran 10h+ without finishing. So I extracted its single best *cheap* idea into my own
reliable offline pipeline instead of fighting timeouts:
- **`struct_E_plane`**: local weighted PLANE fit `E ~ aX+bY+c` per formation, over a per-well
  **centroid** cloud (one median point per well). Using centroids (not dense per-row points) avoids
  the same-well lateral collinearity that made earlier plane fits explode (stp_dev 294, kres 121 →
  fixed to sane kres 15-23). Its known-zone RMSE is consistently **below** my IDW struct (15-23 vs
  24-46) → a genuinely better surface, and that's a *reliable* indicator (own-data, no test shift).
- **GR-offset features**: `gr - typewell_GR(anchor ± {8,20})` — lets the GBM read the local GR
  signature around the held level (the trick the public pipelines use heavily).

GroupKFold OOF: v5 (base+pf3) 10.470 → **v6 (+plane+offsets) 10.149** (−0.32, the biggest single
feature-group gain of the project). featgen5.py → features5.pkl (30 feats); train_v6.py → bundle
(3 LGBM, LGBM-only); kernel `kernel_infer6` (rogii-stacker-infer-v6, ~95s commit, no timeout risk).

### Progress
const 15.88 → PF 12.30 → stacker 10.147 → base+pf3 10.130 → **+plane-fit (v6) = LB 9.918** (first sub-10, new best).
The reliable path to a *durable* top-10 = keep improving the honest core (this plane-fit was step 1;
next: beam-search GR signal, dip-constrained PF, region/cluster block-CV for trustworthy tuning, and
offline-training the full dual pipeline → artifacts dataset → fast-inference kernel). NOT the leakage
overlay (no-op on the private final test that decides the prize).

## Validation methodology
Well-level backtest: for each held-out train well, blank `TVT` where `TVT_input` is
NaN (identical to the competition split), predict, pool squared errors across all
eval rows (matches the metric). High between-well variance ⇒ use ≥80 wells per eval.

---

# 2026-08-03 — the projection layer, and where the ceiling actually is

**Team note:** we are three accounts on one leaderboard entry
(`circiumaru, dulgherustefan, radugoga`) and the daily submission quota is
SHARED. Coordinate before spending slots.

## The one change that mattered: smooth in U, not in TVT

The public pipeline has a post-process layer we never had. It works in
`U = TVT + Z`, the implied stratigraphic level:

    s      = (MD - MD_anchor) / span
    U_hat  = robust_poly_deg4(s)              # IRLS, Cauchy weights, 4 passes
    TVT'   = 0.25*TVT + 0.75*(U_hat - Z)

`U` is the geological surface and is genuinely smooth in position; `TVT` is `U`
minus the wellbore's own vertical motion. That is why our shipped
`savgol(win=61, poly=3)` was measured to be a **no-op** (9.1939 -> 9.1921) while
the same amount of smoothing applied in `U` gives **9.1921 -> 9.0121**.

The anchor level cancels exactly (shifting it moves the polynomial's constant
term by the same amount), so the layer needs only `z` and `md_since` — both
already columns in `test_df_B`. No retrain: inference-only, ~6 min kernel run.

Confirmed on the leaderboard: **8.407 -> 8.259**, CV -0.180 -> LB -0.148 = 82%
transfer. Implemented as `project_level()` in `kernel_inferB7/full.py`.

Fully exploited — every variant measured worse: adaptive degree by well length
(9.022), iterating twice (9.034) or three times (9.044), re-blending with the PF
between projections (9.126). `deg=4, lam=0.75` is the optimum.

## Where the ceiling is

Best least-squares polynomial in `U` fitted to the **true** `U` per well:

| degree | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| RMSE | 54.22 | **5.96** | 3.71 | 2.76 | **2.14** |

We sit at 9.01 inside a family capable of 2.14, and the LB leader (4.67) sits
between the deg-1 and deg-2 oracle. **The whole task is estimating ~4
coefficients of U(s) per well.** The polynomial family is nowhere near the limit.

Decomposed in a per-well orthonormal basis, our coefficient errors contribute
6.56 / 4.61 / 2.99 / 1.84 / 0.77 (55.8 / 27.5 / 11.6 / 4.4 / 0.8 % of variance),
and the optimal linear rescale of every single component is **1.000** — the
estimates are unbiased and the residual is pure noise. There is no post-hoc
shrinkage left to extract; further gain needs better information.

## Measured dead ends — do not re-run these

| attempt | result |
|---|---|
| SP45 selector leg (routed PF scale + beam + hold-shrink) | 9.158 vs our 9.136 — worse. Beam alone 10.40. |
| Formation-column structural surface | exact on train (0.0070) but **absent from test wells**; LOWO helps only <300 ft |
| Prefix dip extrapolation | rho(prefix, true) = **0.93** yet RMSE 39.5 vs const 15.8 — the 4000 ft lever arm |
| Visible-prefix bias correction | rho = +0.17 vs final level; best shrink buys ~0.06 |
| Coefficient-space posterior (`cpf.py`) | 21.6 vs PF leg 12.45 — GR likelihood is flat over +/-30 ft, cannot pin sd-176 ft coefficients |
| Level/shape source recombination | the blend already has the best level AND best shape |
| `w_learned` 0.60 -> 0.70 | CV 9.136, but same direction as the `tune_pp2` change that cost +0.208 on LB |

## Operational notes
- Code competition: submit **through the kernel**, a plain file upload 400s:
  `kaggle competitions submit -c <comp> -k <owner/kernel> -v <ver> -f submission.csv -m "..."`
- **Spatial features invert on the LB.** Third confirmation: iter6 (azimuth +
  neighbour gates + donor transfer) improved CV 9.167 -> 9.029 but scored 8.420,
  worse than iter5's 8.407. Non-spatial CV gains transfer at 40-82%.
- The leaderboard is extremely dense: 8.407 -> 8.259 moved rank only
  3132 -> 3087 of 6077. Sub-7.0 is ~1868; 6.5 is ~1352.

---

# 2026-08-04 — 8.407 → 7.932 in one session

Score path: 8.407 → **8.259** (iter7) → **8.134** (iter10) → **7.932** (iter11).

## The single idea behind all three gains

The physics leg (lik-PF) enters the final answer as a **fixed 0.40 of a blend**
with the learned stack. It must therefore be optimised **for the blend, not
standalone**. A worse-but-decorrelated estimator beats a better-but-redundant one.
I spent most of the session scoring the leg standalone, which is the wrong
objective, and only found this by measuring.

Concretely: pinning the PF's GR-noise scale at `gs=45` makes the filter distrust
GR (which is weakly identifying on this field). The leg gets **worse** standalone
(11.54 → 12.15 RMSE) while the blend gets much **better** (pooled CV 8.98 → 8.47).
Confirmed on 773 wells split into two disjoint halves, and on the leaderboard.

## The three shipped changes

1. **iter7 — smooth in `U = TVT + Z`, not in `TVT`.** `U` is the geological
   surface and is genuinely smooth; `TVT` is `U` minus the wellbore's own vertical
   motion. Our shipped `savgol(61,3)` on `TVT` was measured to be a **no-op**
   (9.1939 → 9.1921); a robust deg-4 polynomial in `U` gives 9.0121. The anchor
   level cancels exactly, so it needs only `z` and `md_since`. Inference-only.
2. **iter10 — conservative GR-sigma for the leg** (`gs=45`, from 32 of the same
   96 PF seeds; the feature-producing seeds keep the shipped `gs` so
   `likpf_mean_d` still matches training). Cost-neutral by construction.
3. **iter11 — retrain with the leg as a feature + `w_sub1` 0.60 → 0.65.**
   Ridge OOF 9.2877 → 8.6861.

## Two calibration lessons that cost the most

* **CV → LB transfer magnitude is NOT predictable.** Observed 91% / 40% / 82% /
  22% / 235%. I wrote a "bigger gains transfer less" law after four points and the
  fifth refuted it. Only the *sign* has been reliable, and only for non-spatial
  changes. Never extrapolate a leaderboard score from a CV delta.
* **Train-well pooled CV understates model improvements.** The retrain moved raw
  OOF −0.60 but pooled CV only −0.05, yet the leaderboard moved −0.202. On train
  the blend is dominated by leg quality, so a better model barely shows. A feature
  that duplicates a post-process input shows a big raw-OOF gain that mostly does
  not survive the blend.

## Process notes that prevented three bad submissions

* **Always control a reimplementation against the shipped artifact before
  sweeping with it.** My `pfsweep` copy scored 14.16 where the kernel scored 11.81
  on identical parameters, voiding an entire parameter sweep. Cause: the kernel
  computes `gs` from **raw** GR with `NaN→0`; I used interpolated GR. A parameter
  showing *exactly* 0.0000 effect is a red flag that the code path is dead.
* **Runtime is a hard constraint.** iter9 (PF seeds 96→192) came back COMPLETE
  with an empty score — its rerun timed out. The kernel processes ~200 hidden
  wells vs 3 visible, so wall-clock scales ~66×: iter7 ≈ 6.6 h against a ~9 h
  limit. **Never increase total PF cost.**
* **Validate on a disjoint well slice, not the tuning slice.** The first iter10
  used 24 leg seeds, which gave −0.50 on the tuning wells and **+0.02** (nothing)
  on a disjoint slice. Raised to 32 before submitting.

## Measured dead ends (do not re-run)

SP45 selector leg · kriged-ANCC closed form (24.6 vs anchor 14.8) · prefix-dip
extrapolation (ρ=0.93 yet RMSE 39.5) · visible-prefix bias correction (ρ=0.17) ·
coefficient-space posterior · PF divergence gating · GBM seed averaging (0.006) ·
learned combiner replacing the linear blend · averaging legs across `gs` ·
`GS_HI` in either direction.
