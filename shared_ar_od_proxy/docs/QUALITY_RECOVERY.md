# Quality recovery experiment

This is a follow-up to the failed three-arm DEV screen, not a claim of a 15% saving.
The regional screen saved bits at the cost of AR accuracy and frequently violated its
soft TRAIN loss allowance. The new experiment tests whether actual-codec guards and
on-policy proxy fitting improve this tradeoff.

## Fixed protocol

- One causal RGB NPP and one standard H.264/H.265 bitstream shared by frozen AR/OD heads.
- Exactly QP 30, 35, 40, 45; medium preset; 16-frame clips; 256-pixel processing.
- 64 TRAIN sources per family and 128 DEV sources per family, deterministically selected
  from the existing source-ID split. Calibration TRAIN remains in TRAIN; calibration
  DEV cannot enter TRAIN. TEST is untouched. The cohorts still have separate video AR
  and image OD ground truth; video OD uses teacher labels.
- Two arms: `quality_fixed` and `quality_refresh`. Both use the same architecture,
  saliency masks, identity initialization, source/QP schedule, and 4,096 **proposals**.
  Different acceptance counts are reported, not described as equal successful updates.
- Both retain the calibrated NPP trunk, reset the two RGB output layers, and initialize
  the regional contribution to zero. Thus both begin at exact identity, rather than
  at the potentially degraded previous NPP output. The calibrated proxy is preserved.

## Task protection

The task objective contains a normalized linear loss term even inside the allowance,
plus a quadratic excess penalty. AR/OD weights are 4/2, preservation weight is 50,
temporal weight is 1, and rate weight warms from .05 to 1 over 512 proposals.

Each proposed Adam displacement is evaluated through the real codec. It must satisfy
the identity task-loss allowance of .02 (using denominator `max(identity_loss,.1)`),
retain any identity-correct AR prediction for the current video, change the current
output by at most 1/255 RMS, and keep its total deviation from source below 6/255 RMS.
Failed proposals are tried at half/quarter displacement. Complete rejection or an
exception restores both model and Adam state. Smaller accepted displacements retain
the Adam moments calculated from the proposal gradient.

Every 256 proposals, a fixed TRAIN anchor panel (two sources per family, all eight
codec/QP combinations) checks identity-relative task losses, AR correctness and total
RMS. Failure rolls back the entire block, including Adam. Final checkpoints contain
only states that passed this panel. The panel is small; these checks provide no hard
bound on other TRAIN examples or DEV accuracy. A 2% normalized loss allowance is not
a 2% accuracy guarantee.

The regional coefficient is bounded by `.2*tanh(raw)` instead of `.9*tanh(raw)`. Its
learning rate is 2e-4 instead of 2e-3. The checkpoint records the filter limit; older
regional checkpoints continue to load with the historical .9 limit. Training masks
combine OD boxes, source motion and the frozen AR teacher's input sensitivity. They
do not require a teacher, boxes, or labels at inference.

## Proxy refresh arm

Only `quality_refresh` fits the proxy after proposals 1,024, 2,048 and 3,072. Each
refresh uses current-NPP outputs from TRAIN, with symmetric +/-2/255 RMS probes in
the cumulative NPP residual direction or a smooth random direction. The fit pool and
audit pool have globally disjoint source IDs at every refresh, though both originate
in TRAIN and may have been used for NPP optimization. Neither DEV nor TEST is used.

Rate fitting includes absolute log-rate and paired log-rate differences. Reconstruction
fitting includes absolute decoder errors and an input Jacobian-vector product
(`autograd.functional.jvp`, `create_graph=True`) against real-codec finite differences.
This directly trains the surrogate Jacobian used in backpropagation. It remains an
approximation at finite amplitude, not the derivative of the discrete codec.

A candidate proxy is committed only if held-out TRAIN audit metrics improve without
materially worsening absolute rate, rate-direction agreement, reconstruction L1 or
reconstruction JVP error. Per operating point rate/direction regressions are also
checked. Rejected refreshes restore the old proxy; every decision and input source
list is saved. Task network state must remain frozen throughout.

Separate rate-only and task-only descent probes are measured at .5/255 and 2/255 RMS
every 127 proposals. Diagnostic probes do not update models. Replay rate features,
targets and encoded stream hashes are retained for audit.

## Evaluation and deployment

All arms finish their declared proposal budget before DEV evaluation; no intermediate
DEV selection. Both are evaluated on identical source IDs and all four QPs using
actual bytes, decoded frames, AR Top-1/Top-5, and official COCO AP. The existing
conservative PCHIP BD-rate code retains all points and leaves plateau/nonmonotone
curves unavailable. Final outputs include full training/guard/refresh counts,
checkpoints, paired rate targets, direction diagnostics and bitstreams.

Kaggle action `quality` selects `ar_od_proxy.quality` and the packaged
`quality_recovery.json`. The notebook is private, requires a GPU and verifies the
exact checkpoint SHA-256 and embedded-source hashes. The launcher must additionally
verify those source bytes against the published immutable Git commit before submission.
The maximum requested session duration is six hours; this is a cap, not an ETA.

```powershell
python -m ar_od_proxy.kaggle prepare --experiment quality --account ACCOUNT `
  --slug ar-od-proxy-quality-20261022 --directory PAYLOAD --commit FULL_SHA `
  --baseline-sha256 CALIBRATED_SHA --checkpoint-dataset ACCOUNT/PRIVATE_DATASET
```

Follow with the existing `submit` command using external pool credentials. The input
checkpoint and all result files stay outside the public source allowlist.
