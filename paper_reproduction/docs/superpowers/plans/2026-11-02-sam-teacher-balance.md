# SAM teacher warm-up and QP balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax.

**Goal:** Run a controlled improved-NPP trial for IS+OD without changing deployed pipeline structure.
**Architecture:** Reuse original NPP, NALCodec, ContentProxy, SAMTask and cached structured maps. Add a training recipe/driver and immutable cloud binder. Two arms differ only in frozen TRAIN-calibrated task gains.
**Tech Stack:** PyTorch, Torchvision, CompressAI, FFmpeg, COCOeval, pytest, Kaggle CLI/SDK.
**Spec:** docs/superpowers/specs/2026-11-02-sam-teacher-balance-design.md

## Global Constraints
Original42core files unchanged; no identity/residual/runtime Gaussian. QPs30/35/40/45. IS/OD only, same-owner private artifacts, public proxy unchanged. Full geometry, disjoint32TRAIN ranking and256newVAL, task1536updates, max10800s/worker, source/config/proxy/optimizer/RNG checks.

## Review Focus
- Parent/config/cache lineage must reject wrong owner/source/content, not merely accept matching shapes.
- Task RNG/noise must be paired in finite probes; calibration must not alter frozen state.
- Ranking must not use final VAL or method-dependent AP intervals; selected snapshot must have task updates.
- Budget/gate exits and partial curves must never mark scientific/experiment completion.
- Extra notebook cells, execution metadata and undeclared/private-cross-owner inputs must be rejected.

### Task1: Recipe and numerical contracts
Files: create reproduce/sam_teacher_recipe.py; tests/test_sam_teacher_recipe.py.
- [ ] Write tests for original topology/config, gains, target, cohort exclusions and common-domain selection.
- [ ] Observe RED; implement pure helpers and objective preserving gradients; observe GREEN.

### Task2: Driver and training provenance
Files: create reproduce/sam_teacher_study.py; tests/test_sam_teacher_study.py.
- [ ] Test parent validation, snapshot/source/config mismatch, deterministic finite probes, completion and source-selection contracts; observe RED.
- [ ] Implement shared full-geometry teacher warm-up, mixture decoder refit, frozen gains/proxy, matched training/ranking/evaluation with existing reusable functions.
- [ ] Probe actual CPU codec and processed_video; verify fail-fast gates and checkpoint metadata.

### Task3: Cloud binding and operations
Files: create reproduce/sam_teacher_cloud.py; tests/test_sam_teacher_cloud.py; local outputs/run_sam_teacher_jobs.py.
- [ ] Test exact one-cell source/config/owner/data/private-artifact binding and mutation rejection; observe RED.
- [ ] Implement canonical immutable notebooks and account-specific hash-bound parent inputs.
- [ ] Prepare local payloads, run full pytest, final independent review, fix Important/Critical in one TDD pass and verify green suite.
- [ ] Verify original42core hashes and publish exact source to existing authorized GitHub branch.
- [ ] Read fresh quota; submit verified private workers and verify remote source/attachments/startup. Record provisional status, not improvement.

## Execution rulings
- Prior user delegation of parameter choices and request to continue improvement covers these bounded training changes; no new architecture or data visibility approval is inferred.
- Keep original released NPP conditioning lambda/8 even though experimental task loss gains differ. This avoids changing the NPP interface; four operating-point gains are explicit training coefficients.
- Task-direction sign is diagnostic because real codecs are discontinuous; full-objective sign has a bounded gate and no guarantee beyond the finite probe.
