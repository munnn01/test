# Frozen OD750 Kaggle aggregation unit

This unit aggregates the sealed predictions from completed A8 shards 0/1/2: 750 distinct COCO validation images. It recomputes COCO AP for both codecs, all four QPs 30/35/40/45 and both anchor/adaptive arms before calculating PCHIP BD-rate. The producer checkout stays pinned to `db6d8b5b162d6e0303fc9a94422c32648d34c98d`.

**Prepared and reviewed; not uploaded to Kaggle or executed. No pooled AP/BD-rate result exists yet.**

- `merge_od.py`: Kaggle-only CPU scoring; validates sealed identities, receipts, policy, packet lengths/hashes and paid rates. Reports 750 images plus 384 reserved and 366 additional images.
- `test_a8_od_merge.py`: three synthetic contract tests; no actual model/codec/GT computation.
- `notebook.py`: pinned environment/checkout, input/script hashes, tests and runner.
- `kernel-metadata.json`: private CPU job metadata for the authorized Kaggle owner.

From the repository root, run only the synthetic checks locally:

```shell
python -m unittest discover -s kaggle/v34r1-a8-od-merge750 -p test_a8_od_merge.py -v
```

The launcher requires the exact private `od-inputs750.tgz`, `merge_od.py` and test file in the declared Kaggle input dataset, plus original COCO annotations. Hashes and status are recorded in [preparation.json](../../results/v34r1_a8/od-merge750-prepared/preparation.json). Publishing this source folder does not create that dataset or launch the job.

See [the run instructions](../../docs/V34R1-A8-OD-MERGE.md) for prerequisites and measurement limitations.
