# Review of external implementation plan

User selects the released original NPP, not a residual/identity-initialized variant. Keep all code and outputs in D:/STUDY/LAB/SAM. This review implements the selected feasible subset, not an unverified promise of a publishable gain.

Accepted now:
- Separate rounded/relaxed positive-power content-rate calibration, with finite/positive/bounds checks and a checkpoint-persistent enable buffer.
- SAM-only instance boundaries computed before union, a correctly shaped two-channel hashed cache, and shared offline masks for the two SAM arms.
- Three arms with all other coefficients fixed: control -> SAM interior -> SAM interior+boundary. GT masks remain native supervised task targets, not edge guidance.
- Real-codec Gaussian sigma0.5/0.8 baselines on the same64VAL sources and four QPs.
- Keep original NPP, exact shared warm-up/proxy/RNG, rate/fidelity gates, actual-stream byte accounting and whole-cohort COCO AP.

Corrections:
1. Original ReLU outputs do not prove a representational bottleneck or establish that all bit savings come from blur. On nonnegative RGB, two1x1 identity convolutions and a zero UNet output can represent identity without an input skip. A CPU capacity diagnostic checks this; no worker is initialized that way.
2. Zero-initializing the last convolution makes its gradients potentially nonzero, but earlier-layer gradients may be zero initially. Do not assert every parameter has a nonzero gradient at step0.
3. Cached mode-power error reduction is a development result. Real GPU screen, task convergence, VAL curves and independent repeats are still needed. It does not guarantee no H265 collapse or BD-rate<=-15%.
4. External SAM pseudocode mixes5D foreground with3D edge tensors. Both maps must be N,T,1,H,W, concatenated on channel2; masks/teacher/crop identities must be bound to cached data.
5. FFmpeg libx264/libx265 ROI paths explicitly skip ROI when adaptive quantization is disabled. Native ROI requires encoder-option/version verification and a matched CRF/AQ baseline, distinct from the current fixed-QP protocol. References:
   https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/libx264.c
   https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/libx265.c
6. COCO AP is not an average per-image AP. Paired bootstrap must preserve image multiplicity, clone IDs for repeated GT/predictions, recompute corpus AP and paid aggregate rate, and report invalid BD-rate resamples rather than quietly dropping them.

Deferred under current user scope:
- Residual NPP: user explicitly retains original architecture.
- SAM feature distillation/task-agnostic model: separate study and compute budget.
- Spatial gating at inference, SAM-guided VAL filters and native ROI: require accounting teacher/mask cost and protocol verification.
- Three seeds,500/5000VAL, paired bootstrap CI: after a successful64-image pilot, not replaced by approximate per-image AP.
