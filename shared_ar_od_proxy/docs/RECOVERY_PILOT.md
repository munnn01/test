# E0–E2 shared AR/OD recovery pilot

One shared causal RegionalVideoNPP, one decoded tensor for both frozen task heads,
H264/H265 medium/GOP16/YUV420, exactly QP30/35/40/45. The calibrated proxy remains frozen.
This pilot addresses the previous zero committed updates; it does not promise15% BD-rate gain.

## Protocol

TRAIN32video+32image, DEV128video+128image; fixed source hashes retain calibration TRAIN.
No DEV pixels/metrics are read until both arms finish fitting. TEST is untouched.

E0 measures seven bounded candidates for each TRAIN source and operating point:
identity; three background low-pass strengths; dual-region low-pass; past-frame
motion-compensated filtering with photometric confidence; protected-region detail.
Teacher protection maps come from source AR saliency/OD boxes/motion, offline only.
These privileged maps may use the whole clip or middle-frame boxes; the TRAIN oracle
itself is not a deployed causal algorithm. Filter operators are past-only conditional
on fixed maps. The actual distilled NPP remains causal and receives no teacher maps.
Identity wins byte-rate ties. Labels must pass the same2% task-loss/correctness/RMS guard.
The oracle is a TRAIN candidate-bank result, not a deployed model or a DEV compression gain.
All3584measurement rows include actual stream bytes/SHA256, source IDs and paired seeds.

E1 (`replay_identity`) starts from exact identity. E2 (`replay_warmstart`) starts from
the same identity weights, distills E0 TRAIN labels for256balanced batches, then tests
weight-interpolation scales1/.5/.25/.125/0 on the TRAIN panel. A nonzero warm-start
requires real panel bitrate savings as well as quality/RMS feasibility; otherwise identity.
All cases, including identity labels, contribute to distillation. Teacher-only information
does not enter the NPP inference interface (RGB, codec, QP and causal history).

Both arms optimize up to512proposals with the same balanced source/point schedule:
two distinct videos+two distinct images, effective batch4 via gradient accumulation,
plus one hard replay case if available. Each epoch visits every source/point once.
Each proposed Adam displacement is checked through the real codec at1/.5/.25/.125.
Failures enter a bounded replay queue. Full fixed TRAIN panel(2sources/family×8points)
is checked every16proposals, shortened to8/4after failures. Rejected blocks restore
both weights and Adam. Final partial blocks are always checked.

No useful panel rate improvement for128proposals stops an arm early. Useful progress
requires changed RGB8 and at least0.0001absolute accumulated improvement in mean relative
panel rate since the last useful-progress event. Checkpoint selection keeps every strictly
lower feasible rate, even if its improvement is smaller than the stall threshold.
Quality-feasible blocks can commit without being useful; the selected checkpoint is the
feasible committed checkpoint with the lowest TRAIN-panel mean relative rate, including
the starting checkpoint. `last.pt`/`optimizer.pt` describe the last transaction; `selected.pt`
can precede it and is the only checkpoint sent to final DEV evaluation. These checkpoints
are not a resume protocol. Accepted/committed/rolled-back counts are explicitly distinct.

The panel guarantees neither the whole TRAIN set nor DEV quality. COCO AP has image GT;
Kinetics has AR GT and video OD source agreement, not simultaneous AR+OD video GT.
Final report uses actual stream bytes and four-QP PCHIP BD-rate; invalid curves remain N/A.
Two arms share seed/cohort and are a controlled ablation, not independent replicates.

## Run

```powershell
python -m ar_od_proxy.recovery --config ar_od_proxy/configs/recovery_pilot.json --ar-root PATH --coco-root PATH --annotations PATH --checkpoint calibrated.pt --out NEW_DIRECTORY
```

Kaggle packaging accepts `--experiment recovery` and an owned private calibrated input dataset.
The notebook pins the full source manifest and input checkpoint hash. A6hour session is a
maximum allocation, not a runtime estimate. Do not cancel unrelated GPU jobs for this pilot.

## Primary artifacts

- `oracle_train/candidates.jsonl`, `labels.jsonl`, `identity_targets.json`, `summary.json`.
- `warmstart/distillation.jsonl`, `summary.json`, verified `start.pt`.
- Per-arm `training.jsonl`, `guards.jsonl`, `training_summary.json`, checkpoints and optimizer.
- Per-arm `evaluation/` including real streams, predictions and official COCO AP.
- `recovery_summary.json` and `run.json`; a scientific result exists only after completion/audit.
