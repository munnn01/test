# Deploy the calibrated proxy on Kaggle

Continue the existing NPP from the calibrated format-2 checkpoint. Preserve the
complete calibrated proxy, freeze its parameters, skip proxy warmup and refresh,
and train only NPP for one complete 32-source × 8-point grid (256 updates).

Use the same source IDs, RGB 256, 16 video frames / one COCO image, closed GOP,
H.264/H.265, QPs 30/35/40/45 and joint task weights. Set Adam learning rate 2e-5.
Bound each update's NPP output change on the current TRAIN case to RMS 1/255 by
halving the proposed parameter displacement when needed. This bound is local to
that case/update, not a bound on cumulative drift or a guarantee of rate descent.

Before training, verify the input checkpoint SHA256, format and calibration
stage, compatible data/codec/model config, and exact source-ID manifest. Record
the input source hashes and all intentional training-config changes. Save the
starting checkpoint and evaluate it and the final NPP with actual codecs on the
same fixed DEV set. No DEV-driven step selection or tuning.

The private T4 notebook mounts the completed calibration kernel output. Publish
the source to the existing GitHub branch, pin its immutable commit, and verify
every source file hash. Run tests and review the warm-start/freeze/update-bound
paths before submission. Check the external running/completion state, download
actual results and verify frozen proxy weights and NPP changes before reporting.
