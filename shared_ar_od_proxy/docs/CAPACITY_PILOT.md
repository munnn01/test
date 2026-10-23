# Paper-sized NPP capacity comparison

User request: test whether an NPP at or above the paper's reported9.42M parameters
improves the current shared AR/OD pipeline. This is an exploratory single-seed pilot.
No gain is inferred from the parameter count itself.

Three fresh arms share the same calibrated proxy(width12), pretrained task heads,
source cohorts, small parent weights, exact identity output, optimizer/guard/loss settings,
QP30/35/40/45 and H264/H265 medium/GOP16/YUV420:

- `small`: original RegionalVideoNPP, approximately36.5k parameters.
- `matched`: shared parent plus a base29U-Net branch, approximately9.9M total parameters.
- `large`: same U-Net architecture family with base32, approximately12.0M total parameters.

The new branch has four spatial reductions, skip connections, per-level QP/codec
conditioning, and a bottleneck ConvGRU at1/16resolution. The RGB head starts at zero.
Every added parameter belongs to the forward path; there is no unused padding of the count.
The original pixel/regional branch remains active. Only present/past frames enter inference;
hidden states reset at each clip. No teacher maps, future frames, postprocessor or side data.

Non-reentrant activation checkpointing recomputes each frame during backward while
preserving the temporal gradient. FP32 is used for all arms; TF32 is disabled and cuDNN
deterministic selection is enabled. Full deterministic algorithms are not enforced because
of CUDA operation support; repeated runs/independent seeds are still needed.

TRAIN32videos+32images; DEV128videos+128images. Common source-ID selection retains
calibration TRAIN and excludes calibration DEV from fitting. Each notebook runs one arm.
All arms receive512proposals, effective batch2videos+2images plus hard-case replay;
the stall window is512so a larger model is not cut off after128proposals. Existing2%loss,
AR correctness, RMS, line-search and16→8→4transaction guards remain. Final model selection
uses the fixed TRAIN panel; DEV is evaluated only after fitting. TEST is untouched.
No E0 candidate labels/warm-start are used, to isolate the architecture/capacity comparison.

Small versus U-Net changes architecture as well as count; matched versus large changes
width within one architecture family. The old recovery result is not reused as the new
small control. A negative512proposal result on64TRAIN sources would not prove that
large NPPs cannot help after longer training or improved proxies.

The runner records parameter counts, a real16×256×256forward/backward smoke without an
optimizer step, cold-profile time including gradient validation, full training peak GPU
memory, initial parent/proxy hashes, checkpoint selection, actual DEV bitstreams and
separate AR/OD BD-rates. Profile memory fields are process-wide CUDA allocator peaks
with the base NPP, common parent and proxy still resident, not isolated NPP memory.
Task models enter only the separate full-training memory measurement.
COCO supplies image OD GT; video OD is source agreement, not simultaneous AR+OD GT.

```shell
python -m ar_od_proxy.capacity --arm matched --config ar_od_proxy/configs/capacity_pilot.json --checkpoint calibrated.pt --ar-root PATH --coco-root PATH --annotations PATH --out NEW_DIRECTORY
```

Kaggle: `--experiment capacity --capacity-arm small|matched|large`, owned private checkpoint
dataset and exact checkpoint SHA256. Each notebook is private and pins the same source
commit. A6hour allocation is a maximum, not a completion estimate. Do not cancel unrelated jobs.
