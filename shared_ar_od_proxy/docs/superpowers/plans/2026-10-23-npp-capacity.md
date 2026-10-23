# NPP capacity comparison

User request: design NPPs at least as large as the reference paper(~9.42M), and test
whether they improve over the current36.5k model. Implementation/deployment are authorized.

Design: retain the exact small regional NPP as a shared starting branch. Add a pixel-aligned
four-level U-Net, QP/codec modulation and causal bottleneck ConvGRU. Zero-initialize only
the new RGB output head. Matched/base29 targets9.42–10.5M total; large/base32 targets>11.5M.
Activation checkpoint each frame during training; no batch normalization or future frames.
This is a larger video adaptation, not a claim to reproduce the image paper exactly.

Controls: fixed calibrated proxy(width12), pretrained tasks, common parent NPP weights,
exact identity output at start, same TRAIN32/task, DEV128/task, seed20261023, loss/guards/
sampling/budget from the completed recovery pilot. No E0 labels or warm-start intervention.
All arms run fresh from a common seed, not from the previous selectedDEVcheckpoint.
QP30/35/40/45, H264/H265 medium/GOP16/YUV420. TEST untouched. Single seed exploratory.

1. Write RED tests for size, identity, geometry, causality, gradient/checkpoint parity and
   format2 checkpoint loading with proxy width independent of NPP capacity.
2. Implement LargeRegionalVideoNPP and backward-compatible loading/config persistence.
3. Write RED tests for one-arm capacity runner/protocol/private payload. Implement three
   arm definitions with identical fitting/evaluation paths, TRAIN-only checkpoint selection,
   parameter/latency/peak-memory receipts and immutable cloud action.
4. Test full suite and pretrained-task CPU smoke; independent review. Publish to existing
   dedicated GitHub branch with expected-head check; launch private small/matched/large
   jobs using authorized pool accounts, without canceling unrelated GPU jobs.
5. Confirm worker commit/GPU/parameter counts/initial identity before reporting deployment.
   Compare completed DEV BD-rate later; do not claim a gain from parameter count alone.

Ruling: the user delegated architecture/size selection; proceed with these concrete engineering
choices rather than re-asking for design authorization. Existing non-Git local workspace uses
baseline source snapshot and expected remote-head publication lease, as in preceding work.
