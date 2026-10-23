# Paper-based continuous training pilot

User authorization: analyze causes and implement improvements while retaining both preprocessing papers. Proceed within existing authorization; no additional design approval. Existing workspace is a source snapshot, not a Git checkout; preserve baseline source bundle and publish with an expected-head lease on the dedicated branch.

## Evidence and scope

Capacity audit: matched 9,895,548 and large 12,039,798 parameters selected exact identity. Matched committed only 40/512 updates; 275 accepted proposals were later rolled back. Failures concentrate on two TRAIN video anchors. This supports investigating training constraints, not a proof that capacity or proxy gradients are sufficient.

Paper bases: https://github.com/XingtongGe/PreprocessingICM and https://arxiv.org/abs/2512.15331. Preserve preprocessing before a standard codec, frozen task networks, differentiable proxy, rate/task/distortion objective and QP conditioning. Image paper motivates real codec forward/proxy backward and fixed-QP before adaptive training. Video paper motivates temporal preprocessing; its second task is tracking, not detection. Joint AR+OD, our proxy and four QPs are adaptations, not an exact reproduction.

## Experiment specification

Two equal-budget arms, common 9.90M identity initialization, fixed calibrated proxy and teachers, FP32, 512 Adam steps, batch 2 video + 2 image. Same 32 TRAIN sources/task as capacity; add next 8 TRAIN sources/task for checkpoint selection, exclude calibration TRAIN from selection. DEV 128/task only after training, TEST untouched.

`continuous`: all QPs throughout. `curriculum`: QP35 for steps 1–128, then 30/35/40/45. Both share LR decay at step385, rate warmup, relative task loss plus soft barrier, semantic preservation and temporal regularization. No quality-triggered step/block rollback or hard-case replay. Retain finite-value checks and gradient clipping. Existing guarded capacity run is a historical reference, not a pure causal control (replay and selection also differ). The two new arms isolate QP scheduling only.

Every128 steps, measure all eight codec/QP points on the disjoint selection set. Select minimum mean rate ratio only when each codec/QP/task-family mean relative task loss <=2%, AR correct count does not fall, and all sample output RMS <=6/255. Do not roll back the training trajectory when selection fails. Identity is fallback; record both last and selected. These selection criteria are surrogate constraints, not a guarantee of DEV accuracy or BD-rate.

Audit identity and final unselected output on one selection source/task at every codec/QP: reconstruction L1/PSNR, proxy rate relative error, real task/rate changes for proxy-gradient descent at RMS0.5/255 and2/255. Diagnostic audit never selects checkpoints or changes parameters.

## Implementation sequence and interfaces

1. RED tests for `paper_training.split_train`, `phase_qps`, `selection_report`, numerical-only optimizer step, and pinned Kaggle action/arm validation.
2. Implement continuous trainer, selection and directional audit; adapt capacity runner setup into a separate entry point without changing existing experiments/evaluator.
3. Add packaged config and `paper-training` Kaggle action with explicit `paper_arm`.
4. GREEN full suite; real pretrained CPU smoke at16x64 for new gradient path and both codecs; validate actual9.90M construction and existing CUDA geometry on workers.
5. One final independent whole-change review (executing-plans skill), resolve findings, publish allowlisted source with SHA verification, verify private inputs, submit two pinned GPU jobs, inspect startup.

Review focus: TRAIN selection disjoint from optimizer/proxy fitting, correct codec gradients/RNG, no accidental rollback, identity fallback, equal budgets, accurate paper attribution, immutable private payloads. Do not claim a measured improvement before DEV results exist.

## Progress

- Baseline publication bundle preserved in output/ar_od_proxy_paper_training_20261023/baseline_source_bundle.json.
- RED: five new tests failed as expected before implementation (missing module/action).
- Implemented trainer, held-out selection, directional diagnostics, packaged configuration and pinned Kaggle action. Detailed Vietnamese evidence/limitations: docs/PAPER_TRAINING_VI.md.
- Full suite:90 passed. After review fix:14 focused paper/deployment tests passed. Final paper tests:8 passed, including added numerical directional-audit regression (91 distinct tests covered overall). Its first fixture incorrectly omitted the video OD term; corrected fixture to match the AR+OD interface.
- Real pretrained CPU smoke: exact9,895,548 parameters, two successful finite continuous updates through H264/H265, fixed proxy and task weights unchanged,16x64 geometry.
- Independent reviewer found source-order confounding in the first QP schedule; fixed by deriving both arms from the same full-grid schedule and replacing only QP. Reviewer verified identical source/codec order across512 steps and identical full cases after step128; no unresolved findings.
- Source publication/deployment receipts are recorded outside the immutable source bundle under output/ar_od_proxy_paper_training_20261023.
