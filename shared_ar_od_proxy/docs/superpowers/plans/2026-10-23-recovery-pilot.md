# E0–E2 shared AR/OD recovery pilot

Authority: user approved implementation of the research proposal on 2026-10-23.
Spec: `output/ar_od_proxy_research_20261022/HUONG_CAI_THIEN_CHI_TIET_VI.md` in the workspace.

Constraints: one NPP, one output/bitstream, frozen calibrated proxy and pretrained tasks;
H264/H265 medium, GOP16, QP30/35/40/45, causal inference. TRAIN only for fitting,
oracle selection, warm-start and guards. DEV for final paired evaluation; TEST untouched.
No claims of joint video GT or achieved 15% gain. Never cancel unrelated cloud jobs.

1. Candidate bank: seven bounded, causal TRAIN-only filter candidates. Source-paired
   seeds, real bytes/losses, identity fallback, selected feasible labels and audit records.
   Temporal candidate uses past-frame motion compensation and occlusion confidence.
2. Recovery optimizer: deterministic balanced four-source batches, hard-case replay,
   real-codec line search, full TRAIN panel every16 proposals, transactional Adam rollback,
   adaptive blocks and early stall detection. Preserve2% task allowance/correctness/RMS.
3. Warm-start paired arm: fit same NPP from E0 labels, validate/backtrack against TRAIN
   panel, then run same recovery budget/schedule. Keep original identity checkpoint as control.
4. Runner/payload/docs: all8operating points, immutable sources and calibrated checkpoint,
   per-arm final DEV curves and explicit N/A BD if invalid. Private Kaggle notebook/data.
5. Regression/integration tests, pretrained smoke, independent whole-change review;
   publish verified allowlist to existing dedicated GitHub branch and launch one paired job.

Ruling: E3–E6 are follow-up experiments gated on E0–E2 evidence, rather than simultaneously
changing proxy, architecture and optimizer. This implements the recommended immediate plan.
Ruling: existing project is not a Git checkout; isolate via dedicated existing remote branch
and retain local source snapshot/manifest before edits. Do not initialize or rewrite parent repo.
