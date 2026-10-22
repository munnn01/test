# Controlled BD-rate Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Ship and run the user-approved three-arm comparison on Kaggle.
**Architecture:** Add an isolated experiment runner around the existing codec, task
heads and evaluation. Preserve existing checkpoint loading and add an explicit
regional NPP model kind. Keep all old protocols reproducible.
**Tech Stack:** PyTorch, TorchVision, FFmpeg, NumPy, SciPy, Kaggle CLI.
**Spec:** docs/BDRATE_SCREEN.md

## Global Constraints

- Exact QPs 30/35/40/45 and both codecs; one decoded stream shared by task heads.
- Freeze calibrated proxy, identical starts and budgets, no TEST-based selection.
- Private checkpoints, public allowlisted source only; immutable source publication.

## Review Focus

- Plateaus and reversed quality curves must yield unavailable BD-rate.
- Regional checkpoints must restore their own architecture and exact outputs.
- Zero initialization must retain the old NPP output while allowing filter learning.
- Identity baselines and dual weights must be isolated by TRAIN case and arm.
- Submission acknowledgment must not be reported as a running GPU job.

### Task 1: Objective, regional model and metric contracts

Files: models.py, engine.py, screen.py, bdrate.py, tests/test_screen.py.
Interfaces: RegionalVideoNPP.from_base(base); constrained_objective(...);
bd_rate(reference_rates, reference_quality, candidate_rates, candidate_quality).

- [x] Write failing tests for zero-contribution initialization, checkpoint restoration,
  rate normalization, constraint gradients and invalid BD curves.
- [x] Run tests and confirm failures, then implement contracts and rerun.

### Task 2: Controlled experiment and cloud integration

Files: screen.py, configs/bdrate_screen.json, kaggle.py, pyproject.toml,
tests/test_screen.py, docs/BDRATE_SCREEN.md.

- [x] Test cloud action, owned private input, equal schedules and separate dual state.
- [x] Implement 3072-step arms, periodic gradient/real-step diagnostics, 32-per-family
  DEV evaluation, summary and auditable receipts. Run the full suite and codec smoke.
- [ ] Independent review and fixes, then publish exact allowlisted source to GitHub.
- [ ] Submit private T4 notebook, verify RUNNING, retrieve and inspect results when ready.

Execution ledger: 51 tests passed on 2026-10-22; independent reviewer also ran
51 tests and found no remaining Critical/Important blockers. Fixed per-case RNG
pairing after observing its regression test fail. Ruling: compare the regional
architecture plus an explicit scalar LR of 2e-3, with all convolutional LRs 2e-5;
otherwise eight new scalar strengths cannot explore useful filtering in 3072
updates. This limits attribution to the combined architecture and recipe. No
confidence intervals or TEST claims are part of this initial screening.
