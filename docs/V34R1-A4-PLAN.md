# V34r1 A4: align development selection with the rate/quality objective

The user authorized continued improvement and Kaggle execution. This bounded iteration proceeds under that authorization. A3 results, protocols and failures remain immutable. Nothing in this plan changes the strict scientific BD-rate evaluator or its four raw QP30/35/40/45 points.

## OT diagnosis and bounded experiment

A3's same-QP mean-HOTA-drop <=.002 / maximum-drop <=.01 gate rejected all twelve FIT candidates. Some candidates reduce paid bytes while improving average HOTA. This gate is an additional design restriction, not the user's BD-rate target.

For A4 use observed_trailing32 only: it avoids full-prefix dilution demonstrated by the synthetic long-prefix test. Retain the six A3 thresholds and every actual codec/candidate byte. On the existing FIT sequences, set one scale per codec to log(rate_QP30/rate_QP45)/(HOTA_QP30-HOTA_QP45), requiring positive finite endpoint spans. Minimize mean per-window log paid-rate ratio plus the equally weighted scaled eight-condition HOTA drop. Identity has objective zero and wins unless a candidate strictly improves it; threshold ties choose the smaller value. This is a coarse Lagrangian diagnostic, **not BD-rate**, and does not repair interior nonmonotonicity.

Freeze the resulting single policy and parent-report hash before any new replay on the previous validation sequences. Those sequences are now explicitly reused development calibration; they are not a fresh independent validation claim. Recompute both FIT and calibration with the pinned official tracker on Kaggle CPU from frozen decoded predictions. Check exact anchor and selected-FIT parity against A3 before accepting calibration results. No codec, detector, new GPU inference, or TEST access is needed. If either calibration BD-rate is undefined or fails strict -15%, do not evaluate TEST. A passing calibration is motivation for a separately frozen fresh-data study, not proof of generalization.

## OD teacher correction

Per-image AP ignores the global confidence ordering used by full-cohort COCO AP. Add a training-only teacher that substitutes one image's official match tables into the full FIT cohort and reaccumulates official AP. Synthetic tests must demonstrate global-score-order sensitivity, multi-category/crowd parity, defined negative-image cost, and no mutation of cached prediction dictionaries. This teacher never enters the deployed sender. Actual labels/training are computed on Kaggle only, with separate FIT and calibration cohorts. Fresh64 DEV IDs and reserved512 TEST IDs remain untouched until a model and selection rule are frozen.

## Implementation checklist

- [x] Reproduce the conservative OT gate's rejected tradeoffs using already measured FIT numbers; no local task scoring.
- [x] RED/GREEN synthetic contracts for FIT tradeoff selection and the exact marginal-AP teacher.
- [x] Freeze the OT policy; implement replay with cloud-only guard and exact A3 parity checks.
- [x] Independent code review; run synthetic contracts and provenance checks before push.
- [ ] Run the frozen OT replay on Kaggle CPU; audit and publish the measured result without changing the gate after calibration.
- [ ] Build the OD training-label job from immutable A3 predictions; preserve separate cohorts and no TEST access.

Ruling: keep the original FIT/validation names in the parent evidence for byte fidelity, but call the reused validation set calibration in every A4 interpretation. Cost if wrong: development reuse could be mistaken for independent evidence; the fresh cohorts remain reserved to prevent that.

Review fixes: replace whole-suite codec integration with an explicit synthetic unittest allowlist; preserve setup failures in immutable status/archive with existing logs only; compare every parent/replay tracker output byte before asserting tracker parity. Three new failure/parity contracts were observed RED then GREEN. All21 relevant synthetic contracts pass locally; no real local task scoring. No deferred minor findings.
