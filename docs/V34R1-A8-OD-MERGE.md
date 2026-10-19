# V34r1 A8 OD aggregation: prepared source and current evidence

The OD aggregation unit is prepared for **750 of the fixed 1,000 images**, using complete shards 0/1/2. It has not executed on Kaggle, and **no pooled AP or BD-rate is available**. The dataset upload was not executed because the automatic approval-review service returned `Selected model is at capacity`.

| Completed shard | Images | H.264 BD-rate | H.265 BD-rate |
|---|---:|---:|---:|
|0|250|−21.0651%|−17.5075%|
|1|250|−22.1981%|−18.4581%|
|2|250|−29.4932%|−21.1437%|

These are the measured **per-shard** results. AP and BD-rate must be computed from merged predictions; neither column can be averaged to produce the 750-image result. Detailed original curves and cloud receipts are in [per-shard-results.json](../results/v34r1_a8/od-merge750-prepared/per-shard-results.json).

## Frozen scoring protocol

The [CPU unit](../kaggle/v34r1-a8-od-merge750/) uses exactly QP 30/35/40/45, H.264/H.265, and the locked OD policy. It validates the three complete receipts, source partition, policy SHA, selected prediction seals, paid 40-byte headers, encoded packet SHA/length, identity anchors and adaptive paid caps before loading official COCO validation annotations.

It recomputes official COCO AP for each codec/QP/arm on three groups: all 750 available images, 384 images within the original reserved 512, and 366 within the additional 488. PCHIP BD-rate is computed from each pooled AP/rate curve. This entails 48 AP evaluations, zero new model inferences, zero new encodes/decodes and no policy retuning.

The full 1,000-image cohort is incomplete. Of the additions to that cohort, 475 images reopen the old conservative TRAIN hash partition; original dedicated training artifacts do not enumerate source IDs. Keep this provenance qualification, report the reserved subset separately, and do not call the full cohort proven independent TEST. No CI or statistical threshold confirmation is claimed.

## Run on Kaggle

1. Provide the exact private input dataset containing `od-inputs750.tgz`, `merge_od.py` and `test_a8_od_merge.py`. The immutable input/script hashes are in [preparation.json](../results/v34r1_a8/od-merge750-prepared/preparation.json). The archive is 53,957,389 bytes and contains 10,453 files for 12,000 selected packet references. It has been prepared locally; its Kaggle upload is pending.
2. Mount that dataset and `awsaf49/coco-2017-dataset`. The checked-in metadata already declares these sources and a private CPU run under the authorized account.
3. From this repository root, submit the prepared launcher:

```shell
kaggle kernels push -p kaggle/v34r1-a8-od-merge750
```

The launcher clones producer commit `db6d8b5b162d6e0303fc9a94422c32648d34c98d`, installs the pinned numerical/COCO dependencies, verifies every input/script anchor, runs three synthetic contracts and executes scoring. It intentionally uses the original producer revision; this preserves the input receipt contract even though the aggregation source is published in a later commit.

Expected outputs are `od-merged750/od-merged-report.json`, `od-merged750/od-merged-curves.csv`, `od-merged750/cloud-status.json`, `od-merged750.tgz` and `od-merge-tests.log`. Require successful tests, a complete cloud receipt and the exact input hash before reporting a pooled result.

## Verification before publication

Three synthetic contract tests passed after an observed failing stage. New metadata guards also passed on the existing 750-image/12,000-row caches, including the 384/366 split. Independent read-only review found no blockers. Actual cloud runtime remains unverified. All real model, codec, tracking and GT scoring work stays on Kaggle; local checks use saved metadata and synthetic fixtures only.
