# Proxy correction and validation protocol

The v1 logistic head has a provable bitrate floor from scale >= 0.05. Replace its
rate prediction with a positive content model, conditioned on codec and the fixed
QP grid 30/35/40/45. Separate learned stream overhead (bits per GOP) from payload
(bits per pixel); use spatial Y/chroma differences and temporal differences so
the input derivative has an interpretable content dependence. Preserve the v1
head only to reproduce baseline diagnostics and migrate old checkpoints explicitly.

Fit the rate head on TRAIN sources only, using real x264/x265 rates at all eight
operating points and controlled blur/sharpen perturbations. Fit reconstruction
against actual decoded TRAIN tensors. Freeze the existing NPP so before/after
comparisons share exactly the same inputs. Never select settings on DEV outcomes.

On disjoint DEV sources report, separately for video and image and codec/QP:
rate relative error and ratio; reconstruction L1, PSNR, and codec distortion;
central finite-difference signs for rate and decoded MSE using smooth blur/sharpen
and a fixed random direction; and actual rate changes for negative proxy-rate
gradient steps at RMS 1/255 and 4/255. Measure the proxy change at the exact same
clipped candidates. Count codec ties separately (one bit tolerance), report
agreement only for informative changes, and report loss-increasing steps.

The reconstruction probe is MSE(decoded(candidate), fixed unperturbed NPP output
y). Its reference remains fixed for both candidates; the proxy Jacobian is
weighted by the real decoded residual at y using the same value/gradient bridge
as NPP training. This measures one pixel-MSE sensitivity, not the AR/OD task-loss
Jacobians and not distortion against each perturbed candidate itself. Report
blur and fixed smooth random directions separately as well as combined.

Codec decisions are discrete: these are finite-step sensitivity diagnostics,
not a claim that an exact codec derivative exists. A low scalar rate error does
not establish useful input gradients. Do not re-train the NPP or claim improved
AR/OD accuracy from this calibration-only experiment.

Implementation: regression tests first; targeted real-codec CPU checks; full
suite; independent code review; immutable GitHub publication; private Kaggle GPU
run; download and audit the actual outputs before reporting conclusions.
