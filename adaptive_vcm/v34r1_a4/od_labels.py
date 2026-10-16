"""Kaggle-only exact cohort-dependent marginal-AP labels from authenticated DEV."""
import argparse
from collections import defaultdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time
from .marginal_ap import MarginalAP
from ..v34r1_a3.core import QPS,CODECS,audit_grid
from ..v34r1_a3.analyze import metric

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--annotation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only real GT label computation')
    if args.out.exists(): raise ValueError('immutable fresh label output required')
    raw=args.annotation.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    if sha!='e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f': raise ValueError('official COCO annotation drift')
    meta=json.loads(raw); gt=defaultdict(list)
    for annotation in meta['annotations']: gt[annotation['image_id']].append(annotation)
    manifest=json.loads((args.source/'source-manifest.json').read_text()); parent=json.loads((args.source/'dev-report.json').read_text())
    rows=[]
    for worker in (0,1): rows.extend(json.loads(gzip.decompress((args.source/f'worker{worker}/rows.json.gz').read_bytes())))
    audit=audit_grid(rows,[v['id'] for v in manifest['dev']]); groups=defaultdict(list)
    for row in rows: groups[(row['image_id'],row['codec'],row['qp'])].append(row)
    for group in groups.values(): group.sort(key=lambda r:r['profile'])
    ids={split:sorted(v['id'] for v in manifest['dev'] if v['split']==split) for split in ('fit','validation')}
    if len(ids['fit'])!=64 or len(ids['validation'])!=64 or set(ids['fit'])&set(ids['validation']): raise ValueError('separate complete cohorts required')
    args.out.mkdir(); started=time.perf_counter(); labels=[]; checks=[]; unique=0
    for split,image_ids in ids.items():
        for codec in CODECS:
            for qi,qp in enumerate(QPS):
                anchors={image:groups[(image,codec,qp)][0] for image in image_ids}
                anchor_predictions=[p for image in image_ids for p in anchors[image]['coco']]
                teacher=MarginalAP(meta,gt,image_ids,anchor_predictions)
                if abs(teacher.anchor_ap-parent['results'][split]['anchor']['curves'][codec]['quality'][qi])>1e-12:
                    raise ValueError('teacher anchor differs from official A3 full-cohort AP')
                cache={}; check_profiles={(image_ids[0],14),(image_ids[-1],20)}
                for image in image_ids:
                    for row in groups[(image,codec,qp)]:
                        prediction_sha=hashlib.sha256(json.dumps(row['coco'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                        key=(image,prediction_sha)
                        if row['profile']==0: ap=teacher.anchor_ap
                        elif key in cache: ap=cache[key]
                        else: ap=teacher.ap_with(image,row['coco']); cache[key]=ap; unique+=1
                        if (image,row['profile']) in check_profiles:
                            # Independent full default COCO evaluation, no stitched match tables.
                            joint=[dict(p) for other in image_ids for p in (row if other==image else anchors[other])['coco']]
                            full=metric(joint,image_ids,meta,gt)['AP']; error=abs(full-ap)
                            if error>1e-12: raise ValueError('marginal teacher differs from full official COCO AP')
                            checks.append(dict(split=split,codec=codec,qp=qp,image_id=image,profile=row['profile'],absolute_ap_error=error))
                        labels.append(dict(image_id=image,split=split,codec=codec,qp=qp,profile=row['profile'],
                            anchor_cohort_ap=teacher.anchor_ap,replacement_cohort_ap=ap,
                            scaled_ap_damage=len(image_ids)*(teacher.anchor_ap-ap)))
                print('A4_OD_LABEL_PROGRESS',split,codec,qp,len(labels),round(time.perf_counter()-started,2),flush=True)
                del teacher,cache; gc.collect()
    if len(labels)!=21504 or len(checks)!=32: raise ValueError('label/parity-check grid incomplete')
    payload=gzip.compress(json.dumps(labels,allow_nan=False,separators=(',',':')).encode(),mtime=0)
    (args.out/'marginal-ap-labels.json.gz').write_bytes(payload)
    report=dict(scope='cohort-dependent one-image replacement labels for training/development only; not additive joint AP damages or intrinsic image scores',
        grid=audit,annotation_sha256=sha,parent_publication_manifest_sha256=hashlib.sha256((args.source/'publication-manifest.json').read_bytes()).hexdigest(),
        label_sha256=hashlib.sha256(payload).hexdigest(),fit_images=len(ids['fit']),calibration_images=len(ids['validation']),
        separate_fit_calibration_teachers=True,full_official_parity_checks=checks,unique_nonidentity_reaccumulations=unique,
        new_encodes=0,new_model_inferences=0,test_scored=False,fresh_dev64_scored=False,wall_seconds=time.perf_counter()-started)
    (args.out/'label-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A4_OD_LABEL_COMPLETE',json.dumps(dict(rows=len(labels),checks=len(checks),seconds=report['wall_seconds'],test_scored=False)),flush=True)


if __name__=='__main__': main()
