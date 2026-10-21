# Shared AR + OD Proxy Implementation Plan

> For agentic workers: execute directly with superpowers:executing-plans; perform a fresh review before publication.

**Goal:** Ship a standalone, tested real-codec/proxy training pipeline and start its private Kaggle GPU pilot.

**Architecture:** One NPP and two frozen task networks, exact real-codec forward values and a conditional video proxy backward path. Mixed partially labeled data retain task masks and honest evaluation scopes.

**Tech stack:** Python, PyTorch, TorchVision, OpenCV, FFmpeg x264/x265, pycocotools, pytest, Kaggle API.

**Spec:** docs/DESIGN.md. Execution and external actions are authorized by the user's latest request.

## Global constraints

- Source geometry and all video frames are preserved by NPP; AR samples only after decoding.
- Frozen parameters retain input derivatives; real codec runs outside autograd.
- A proxy checkpoint must exist before NPP training; fail on mismatched configuration.
- Never fabricate missing annotations, pretrained weights, GPU or experiment results.
- Publish only this folder's allowlisted sources; no credentials, generated caches or LAB documents.

## Review focus

1. Value replacement must preserve real values and the proxy Jacobian, including rate gradients.
2. Recurrence must not depend on future frames or leak between clips.
3. Both task paths must receive the same decoded video; source-derived targets must not be named GT.
4. Codec errors and missing source files must fail explicitly; all paid stream bytes count.
5. Kaggle source must be pinned and independently hashed; failure must preserve logs/checkpoints.

## Tasks

- [x] Core models, bridge and actual-codec adapter: write failing tests for exact forward/proxy gradient, causal shape behavior, frozen weights and H.264/H.265 round trips; implement and run the tests.
- [x] Data/task adapters: test source-group splits, COCO target geometry, differentiable frozen detector loss and unavailable labels; implement real Kinetics/COCO loading and pretrained TorchVision heads.
- [x] Research runner: test a real-codec update, checkpoint compatibility and evaluation provenance; implement proxy fitting, NPP fitting, shared-stream evaluation and inference CLI.
- [x] Packaging and Kaggle: test generated notebook compilation and source hash checks; implement immutable-source notebook builder and redacted local launcher; document commands and limitations.
- [ ] Validate the complete new-folder suite and a local real-codec smoke; review the final code, fix material issues, publish one folder to a new branch in munnn01/test, submit private T4 Kaggle pilot, verify its returned status and record links.
