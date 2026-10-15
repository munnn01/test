"""Cloud-only FIT threshold selection, frozen validation, and full-cohort COCO AP."""
import argparse
from collections import defaultdict,Counter
import contextlib
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
from .core import QPS,CODECS,audit_grid,choose,fit_threshold
from ..v34r1.policy import select_measured
from ..v34r1.metrics import assess_curve

ROOT=Path(__file__).resolve().parents[2]


def metric(predictions,ids,meta,annotations):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not predictions: return dict(AP=0.,AP50=0.,AP75=0.,APsmall=0.)
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=dict(info=meta.get('info',{}),images=[dict(id=i) for i in ids],
            annotations=[a for i in ids for a in annotations[i]],categories=meta['categories']); gt.createIndex()
        dt=gt.loadRes(predictions); ev=COCOeval(gt,dt,'bbox'); ev.params.imgIds=ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(AP=float(ev.stats[0]),AP50=float(ev.stats[1]),AP75=float(ev.stats[2]),APsmall=float(ev.stats[3]))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('cloud-only analysis')
    protocol=json.loads((ROOT/'configs/v34r1-a3-od-probe.json').read_text())
    data=json.loads((args.root/'source-manifest.json').read_text())
    raw=(args.root/'instances_val2017.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=protocol['annotation_sha256']: raise ValueError('GT annotation hash drift')
    meta=json.loads(raw); annotations=defaultdict(list)
    for a in meta['annotations']: annotations[a['image_id']].append(a)
    rows=[]
    for worker in (0,1):
        folder=args.root/f'worker{worker}'
        if not (folder/'complete.json').is_file(): raise ValueError('incomplete DEV worker')
        rows.extend(json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes())))
    expected=[r['id'] for r in data['dev']]; grid=audit_grid(rows,expected)
    by_condition=defaultdict(list)
    for r in rows: by_condition[(r['image_id'],r['codec'],r['qp'])].append(r)
    for value in by_condition.values(): value.sort(key=lambda r:r['profile'])
    splits={split:[r['id'] for r in data['dev'] if r['split']==split] for split in ('fit','validation')}
    def select(group,policy):
        if policy['kind']=='fixed': return group[policy['profile']]
        if policy['kind']=='source_proxy':
            action=choose(group,policy['proxy'],policy['threshold']); return group[action]
        subset=[r for r in group if policy['kind']=='exhaust_guard' or r['profile'] in group[0]['a2_shortlist']]
        anchor=group[0]['bytes']
        chosen,_=select_measured([r['bytes']/anchor for r in subset],[[None,r['current_cost'],None] for r in subset],
            [False,True,False],[0,.05,0],budget=1.,quality_weight=.05)
        return subset[chosen]
    cache={}
    def evaluate(split,policy):
        key=(split,json.dumps(policy,sort_keys=True))
        if key in cache: return cache[key]
        ids=splits[split]; curves={}; extra={}; counts=Counter(); selected=[]; ratios=[]
        for codec in CODECS:
            curves[codec]=dict(rate=[],quality=[]); extra[codec]=[]
            for qp in QPS:
                chosen=[select(by_condition[(i,codec,qp)],policy) for i in ids]
                counts.update(r['profile'] for r in chosen)
                selected.extend(dict(image_id=r['image_id'],codec=codec,qp=qp,profile=r['profile']) for r in chosen)
                pred=[v for r in chosen for v in r['coco']]
                scored=metric(pred,ids,meta,annotations)
                curves[codec]['rate'].append(float(np.mean([r['bpp'] for r in chosen])))
                curves[codec]['quality'].append(scored['AP']); extra[codec].append(scored)
                ratios.extend(math.log(r['bytes']/by_condition[(r['image_id'],codec,qp)][0]['bytes']) for r in chosen)
        result=dict(policy=policy,curves=curves,metrics=extra,selection_counts=dict(counts),
                    log_rate_ratio=float(np.mean(ratios)),selected=selected)
        cache[key]=result
        return result
    anchor_policy=dict(kind='fixed',profile=0)
    fit_anchor=evaluate('fit',anchor_policy)
    anchor_quality=[v for c in CODECS for v in fit_anchor['curves'][c]['quality']]
    fits={}; frozen={}
    for proxy,limits in [('pseudo_ap',protocol['pseudo_ap_thresholds']),('pseudo_loss',protocol['pseudo_loss_thresholds'])]:
        candidates=[]
        for limit in limits:
            result=evaluate('fit',dict(kind='source_proxy',proxy=proxy,threshold=limit))
            candidates.append(dict(threshold=limit,quality=[v for c in CODECS for v in result['curves'][c]['quality']],
                                   log_rate_ratio=result['log_rate_ratio']))
        selected=fit_threshold(candidates,anchor_quality)
        frozen[proxy]=(dict(kind='source_proxy',proxy=proxy,threshold=selected['threshold']) if selected else anchor_policy)
        fits[proxy]=dict(candidates=candidates,selected=selected)
        print('A3 FIT_SELECTED',proxy,json.dumps(frozen[proxy]),flush=True)
    policy_bytes=(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),fit_only=True,
        source_id_protocol_sha256=data['source_id_protocol_sha256'],policies=frozen,fit=fits),indent=2)+'\n').encode()
    (args.root/'fit-frozen-policies.json').write_bytes(policy_bytes)
    # Validation is accessed only after the FIT policies have been written.
    policies=dict(anchor=anchor_policy,a2=dict(kind='a2'),exhaust_guard=dict(kind='exhaust_guard'),**frozen)
    report=dict(scope='exploratory fresh DEV only; reserved TEST512 never inferred',grid=grid,
                fit_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),fit=fits,results={})
    for split in ('fit','validation'):
        report['results'][split]={}
        anchor=evaluate(split,anchor_policy)
        for name,policy in policies.items():
            result=evaluate(split,policy)
            result['bd']={c:assess_curve(anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
            report['results'][split][name]=result
        # Fixed-profile capacity diagnostics cannot select/change frozen policies.
        fixed={}
        for i in range(21):
            result=evaluate(split,dict(kind='fixed',profile=i))
            result['bd']={c:assess_curve(anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
            fixed[str(i)]=result
        report['results'][split]['fixed_profile_diagnostics']=fixed
    report['validation_both_codecs_below_minus15']={name:all(v['point_passes'] for v in report['results']['validation'][name]['bd'].values()) for name in frozen}
    report['test_scored']=False
    (args.root/'dev-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A3_VALIDATION_GATE',json.dumps(report['validation_both_codecs_below_minus15']),flush=True)


if __name__=='__main__': main()
