"""Kaggle CPU diagnostic: GT-informed DEV bank headroom, never a deployable selector."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import spearmanr
from .core import QPS,CODECS,audit_grid,choose
from .analyze import metric
from ..v34r1.metrics import assess_curve

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only GT diagnostic; no local task scoring')
    if args.out.exists(): raise ValueError('immutable fresh diagnostic output required')
    args.out.mkdir(); started=time.perf_counter()
    status=json.loads((args.source/'cloud-status.json').read_text())
    if not status['complete'] or status['commit']!='33ef4181abb9afb6b829de6c6094a2e015b6346a': raise ValueError('frozen complete A3 parent required')
    sys.path.insert(0,str(ROOT/'scripts'))
    from prepare_v34r1_cloud_data import mounted
    annotation=mounted('annotations/instances_val2017.json')
    if annotation is None: raise FileNotFoundError('official frozen COCO annotations missing')
    raw=annotation.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f': raise ValueError('annotation hash drift')
    meta=json.loads(raw); gt=defaultdict(list)
    for a in meta['annotations']: gt[a['image_id']].append(a)
    source=json.loads((args.source/'source-manifest.json').read_text()); rows=[]
    for worker in (0,1): rows.extend(json.loads(gzip.decompress((args.source/f'worker{worker}/rows.json.gz').read_bytes())))
    audit=audit_grid(rows,[r['id'] for r in source['dev']]); groups=defaultdict(list); lookup={}
    for row in rows:
        groups[(row['image_id'],row['codec'],row['qp'])].append(row)
        lookup[(row['image_id'],row['codec'],row['qp'],row['profile'])]=row
    for group in groups.values(): group.sort(key=lambda r:r['profile'])
    ap_cache={}; labels=[]
    for index,row in enumerate(rows):
        image=row['image_id']; positive=any(not a.get('iscrowd',0) and a['area']>0 for a in gt[image])
        if not positive: ap=None
        else:
            prediction_hash=hashlib.sha256(json.dumps(row['coco'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            key=(image,prediction_hash)
            if key not in ap_cache:
                value=metric(row['coco'],[image],meta,gt)['AP']; ap_cache[key]=value if value>=0 else None
            ap=ap_cache[key]
        row['gt_negative_ap']=-ap if ap is not None else None
        labels.append(dict(image_id=image,codec=row['codec'],qp=row['qp'],profile=row['profile'],per_image_gt_ap=ap))
        if (index+1)%1024==0: print('CAPACITY_GT_ROWS',index+1,'/',len(rows),round(time.perf_counter()-started,1),flush=True)
    (args.out/'per-image-gt-ap.json.gz').write_bytes(gzip.compress(json.dumps(labels,allow_nan=False).encode(),mtime=0))
    split_ids={split:[r['id'] for r in source['dev'] if r['split']==split] for split in ('fit','validation')}
    split_ids['combined_exploratory']=[r['id'] for r in source['dev']]
    report=dict(scope='GT-informed exploratory DEV diagnostic only; not deployable; not a bound on optimal cohort AP; no reserved TEST scoring',
        qps=list(QPS),codecs=list(CODECS),grid=audit,slack_grid=[0,.005,.01,.02,.04,.08],results={},correlations={},test_scored=False)
    def evaluate(ids,slack):
        curves={}; choices=[]
        for codec in CODECS:
            curves[codec]=dict(rate=[],quality=[])
            for qp in QPS:
                selected=[]
                for image in ids:
                    group=groups[(image,codec,qp)]
                    action=0 if slack is None else choose(group,'gt_negative_ap',slack)
                    selected.append(group[action]); choices.append(dict(image_id=image,codec=codec,qp=qp,profile=action))
                curves[codec]['rate'].append(float(np.mean([r['bpp'] for r in selected])))
                curves[codec]['quality'].append(metric([p for r in selected for p in r['coco']],ids,meta,gt)['AP'])
        return dict(curves=curves,selected=choices)
    for split,ids in split_ids.items():
        anchor=evaluate(ids,None); result=dict(anchor=anchor,gt_diagnostics={})
        for slack in report['slack_grid']:
            diagnostic=evaluate(ids,slack)
            diagnostic['bd']={c:assess_curve(anchor['curves'][c],diagnostic['curves'][c],qps=QPS) for c in CODECS}
            result['gt_diagnostics'][str(slack)]=diagnostic
        report['results'][split]=result
    for proxy in ('current_cost','feature_cost','pseudo_ap','pseudo_loss'):
        x=[]; y=[]
        for row in rows:
            anchor=groups[(row['image_id'],row['codec'],row['qp'])][0]
            if row['profile']==0 or row['gt_negative_ap'] is None or anchor['gt_negative_ap'] is None or row[proxy] is None or anchor[proxy] is None: continue
            x.append(row[proxy]-anchor[proxy]); y.append(row['gt_negative_ap']-anchor['gt_negative_ap'])
        value=float(spearmanr(x,y).statistic)
        report['correlations'][proxy]=dict(spearman=value if math.isfinite(value) else None,rows=len(x),
            scope='descriptive correlated image/profile/QP rows; no independent-row significance claim; per-image AP is not cohort AP')
    report['wall_seconds']=time.perf_counter()-started
    (args.out/'capacity-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CAPACITY_COMPLETE',json.dumps(dict(rows=len(rows),wall_seconds=report['wall_seconds'],test_scored=False)),flush=True)


if __name__=='__main__': main()
