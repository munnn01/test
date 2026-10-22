# Controlled BD-rate screening

User-approved experiment: compare the existing NPP objective, normalized codec-relative
rate with task constraints, and that objective with a learned spatial filtering branch.
This screening does not promise a 15% saving and does not constitute held-out TEST evidence.

All arms start from the same calibrated checkpoint, use a frozen proxy and frozen
task heads, 32 TRAIN sources (16 per dataset), 3072 updates and exactly the same
source/codec/QP order. H.264/H.265, QP 30/35/40/45, medium preset and 16-frame GOP
stay fixed. The current-case output update guard remains 1/255 RMS.

The normalized objective uses real identity-codec bpp as denominator. AR and OD
loss constraints use the corresponding identity-codec decoded sample as reference,
with a 2% relative loss allowance, a 0.1 denominator floor and bounded online dual
weights per operating point and task. Source references remain fixed. Foreground
boxes and motion weight the input preservation penalty only; inference needs RGB,
codec and QP only. The filter has zero initial contribution and learned signed
strength (can blur or sharpen). Adam uses 2e-5 for convolutional parameters and
2e-3 for the eight new image-wide filter-strength scalars. The output update guard
applies to both parameter groups. This is an architecture-plus-recipe comparison;
the extra scalars have an explicit learning-rate scale. All extra parameters and
optimizer state are saved. Detector proposal sampling uses a fixed seed per TRAIN
source/codec/QP case for both its identity baseline and every candidate pass.

Only TRAIN cases feed objective baselines, masks and dual weights. DEV expands to
32 sources per dataset, including the previous DEV pilot for continuity. It is used
for final comparisons, not for checkpoint selection. TEST remains untouched.
Evaluate all final arms with real bitstreams and save their bytes and predictions.
Compute AR Top-1 BD-rate and COCO AP BD-rate separately, with PCHIP log-rate
integration over common quality support. Reject plateaus, nonmonotone curves,
duplicate rates, missing QPs and empty overlap; never drop a point to force a result.
No joint video AR+OD ground-truth BD-rate is claimed.

Log task/rate input-gradient norms periodically and real-codec rate changes along
actual optimizer updates. These are diagnostics of the frozen proxy, not claims
that its Jacobian equals a true codec derivative. Longer training, task-gradient
proxy refitting and a final TEST study require a subsequent experiment.
