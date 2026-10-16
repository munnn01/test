# A4 OD encoder-risk critic

The frozen protocol is `configs/v34r1-a4-od-risk.json`. This is a new learned sender critic over the same21 measured filters, not training a new RGB preprocessor or changing the frozen detector. Real label creation, features, training, critic inference and COCO scoring run on Kaggle CPU. A later fresh-image codec/detector sweep runs on Kaggle GPUs only.

## Locked choices

- Two ExtraTrees regressors:128 trees, depth6 or10, minimum leaf64 rows, max_features.8, no bootstrap, seed303404, two CPU workers; scikit-learn1.7.2. Train only the64 FIT images. Each supplies the same168 profile/codec/QP rows; these correlated rows are not21,504 independent images.
- Target:64 times the change in **whole FIT-cohort** official COCO AP when one image candidate replaces decoded identity. Separate teachers for each codec/QP and for reused calibration. This is cohort-dependent and not an additive guarantee for jointly selected images.
- Features use source/identity/candidate detector statistics, weak-box/class-consistent IoU, detected class frequencies, paid rate, profile/codec/QP, current source cost, pooled-feature cost and source pseudo-AP. No GT, image IDs, membership, hashes or expensive pseudo-loss. No source image inference is repeated in the training job.
- Export bounded numeric tree JSON. On Kaggle require agreement with native sklearn predictions to1e-10 over all21,504 rows; retain the actual predicted-risk table for audit. An input is converted to float32 before tree decisions, matching sklearn. No executable model serialization.
- For each of seven fixed limits0,.002,.005,.01,.02,.04,.08, choose the cheapest actual paid candidate under predicted risk and paid<=same-QP identity. Identity remains available and wins equal-byte ties. Jointly assemble the selected predictions and score full64-image calibration COCO AP. Choose the best negative endpoint-scaled rate/quality objective across the fourteen prescribed options; model order then smaller limit breaks ties. This objective is not BD-rate. Report all attempts, including undefined raw BD curves.

Both FIT-trained model files are hashed before calibration error or metric computation. Then the selected model, limit, calibration scales, feature/training protocol and fresh64-ID protocol are locked in `frozen-policy.json` before fresh data inference. Calibration is reused development and carries no independent-validation claim.

## Fresh-data decision and final TEST

This A4 prospective gate replaces A3's gate only for the new A4 study; historical A3 evidence remains unchanged. The fresh64 DEV cohort is scored once after the complete policy lock. For each codec compute mean log paid ratio plus the **already frozen calibration scale** times mean four-QP AP drop. Both values must be strictly below log(.85), with complete provenance/block checks, before one reserved512-image TEST evaluation. This is a resource decision using an approximate objective, not a claimed15% BD saving. Fresh-DEV failure means no TEST; changes after viewing it consume it as development and need another untouched confirmation cohort.

The TEST conclusion still requires actual PCHIP BD-rate strictly below−15% for both codecs from exactly QP30/35/40/45, common quality support, no extrapolation or raw-point repair. Undefined or nonmonotone curves do not pass. Report point estimates and uncertainty separately. In the fresh study retain all21 under the old source guard as an ablation, so any improvement is not attributed solely to a larger search bank. All real candidate encodes/decodes and sender search time must be reported, alongside transmitted bytes including headers.

## Verification

Three handmade-data contracts were observed failing then passing: changing GT/IDs/provenance cannot change sender features; paid-cap/fallback/ties cannot depend on GT; portable trees respect float32 boundaries and reject cyclic/unsafe indices. The actual Kaggle training job also runs20 relevant synthetic contracts and the authenticated parent evidence checker. Label creation independently checks16 anchor cohorts and32 replacements against full official COCOeval before saving labels. These checks establish implementation consistency, not generalization or guaranteed target attainment.

The model family and its complexity controls follow the [official scikit-learn ExtraTrees documentation](https://sklearn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html). The teacher uses the [official COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). They do not establish a compression gain for this experiment.

Review fix: reject a changed fresh-ID reservation against the protocol's expected SHA before feature extraction/training; carry that validated hash into the final lock. The changed-reservation regression was observed RED then GREEN. The independent review found no other actionable issues or deferred minors.
