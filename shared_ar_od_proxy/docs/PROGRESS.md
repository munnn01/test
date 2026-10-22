# Execution ledger: docs/PLAN.md

- Scope and implementation authorization: latest user message.
- Isolation ruling: standalone folder plus new GitHub branch, because the active workspace is not itself a repository.
- Environment: D:/STUDY/AI/envs/ten_env/python.exe; torch 2.5.1, torchvision 0.20.1, CPU. FFmpeg is in the environment's Library/bin.
- Kaggle credentials: external LAB pool, qktttttttttt account; authenticated list succeeds with network escalation. No credentials copied into project.
- Baseline: new folder, no pre-existing implementation or tests.
- Core, data/task adapters, training, evaluation and notebook launcher implemented. Exact four-QP protocol: 30/35/40/45 for H.264/H.265, 256 NPP steps cover 32 mixed TRAIN sources at all eight operating points.
- Fresh suite after review fixes: 24 CPU tests passed, including actual x264/x265 round trips, exact proxy Jacobian/value replacement, causal NPP, frozen detector input gradient/BN state, official COCO AP, dense detection agreement and notebook tamper rejection.
- A software-only fixture exercised proxy warmup, all codec/QP groups, NPP updates, proxy refresh and checkpoint reload. It is not scientific task/bitrate evidence. Actual data training is delegated to the authorized Kaggle pilot.
- Package wheel built; packaged four-QP config verified. Public-source allowlist checked against external credential values; zero matches.
- Independent review identified zero-exit Kaggle push errors and asymmetric source-box truncation. Both fixed with observed RED-to-GREEN regressions. Submission requires a returned version and matching URL; running/completion still require external status verification.
- Publication uses a new branch ar-od-proxy-20261022 in munnn01/test. Notebook sources will be pinned to the resulting immutable commit and byte-hash verified before submission. Runtime/submission receipts live outside this published source folder.

- v1 completed on Kaggle. Actual held-out rates exposed an unrepresentable logistic
  rate floor; scalar mismatch was reported, without claiming gradient fidelity.
- Current correction follows docs/PROXY_CALIBRATION.md. Five regressions first
  failed for missing rate representability, header units, sensitivity and explicit
  migration; three more failed for missing train-only fitting/reporting/launcher.
  Fresh full suite: 32 passed. A real x264/x265 CPU fixture exercised calibration,
  all eight points, reconstruction Jacobians, paired probes and descent steps;
  32 diagnostic rows completed. This fixture is software verification only.

- Calibration v1 completed on Tesla T4 (869.8 s cloud): 768 TRAIN rate targets,
  256 DEV before/after rows. Rate MAPE fell to 16.52% on videos and 3.03% on
  images; reconstruction improvement was small. Raw negative-gradient steps at
  RMS 1/255 reduced actual rate on 121/128 cases, while 4/255 often increased
  video rate at low QP. These are finite-step pilot diagnostics, not task gains.
- Deployment now continues the NPP from calibrated.pt, freezes the entire proxy,
  skips proxy fitting/refresh, uses LR 2e-5 and a current-case RMS 1/255 update
  guard. Source-ID manifest and checkpoint provenance are verified; starting and
  final NPPs receive the same actual-codec DEV evaluation. Five new tests first
  failed for missing deployment behavior; full suite now 37 passed. An actual
  calibrated-checkpoint CPU fixture completed bounded NPP steps through x264 and
  x265 and verified frozen proxy weights and checkpoint reload (software only).
- Independent deployment review found an omitted-RMS config could bypass the
  guard. Six observed RED regressions cover missing/zero/negative/nonfinite/too
  large bounds; the entry point now rejects them. Fresh full suite: 43 passed.
