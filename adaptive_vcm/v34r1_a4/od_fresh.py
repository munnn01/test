"""One frozen OD critic on fresh64 DEV; source-only decisions precede GT scoring."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .cloud_record import require_sha256
from .risk import features,predict_forest,choose_predicted
from .fresh_gate import fresh_rd_gate
from ..v34r1_a3.core import QPS,CODECS,audit_grid
from ..v34r1_a3.analyze import metric
from ..v34r1.policy import select_measured
from ..v34r1.metrics import assess_curve

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--models',type=Path,required=True)
    p.add_argument('--annotation',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only critic inference/real COCO scoring')
    started=time.perf_counter(); manifest_raw=(args.root/'source-manifest.json').read_bytes(); manifest=json.loads(manifest_raw)
    policy_raw=(args.models/'frozen-policy.json').read_bytes(); frozen=json.loads(policy_raw); policy=frozen['policy']
    if policy['kind']!='risk_forest': raise ValueError('identity has no frozen improvement to confirm')
    require_sha256(policy_raw,manifest['frozen_policy_sha256'])
    require_sha256((ROOT/'configs/v34r1-a4-od-next-dev.json').read_bytes(),frozen['fresh_dev_id_protocol_sha256'])
    require_sha256((ROOT/'configs/v34r1-a4-od-risk.json').read_bytes(),frozen['training_protocol_sha256'])
    if policy['model'] not in ('extra_trees_depth6','extra_trees_depth10'): raise ValueError('prespecified critic name required')
    model_raw=(args.models/(policy['model']+'.json')).read_bytes(); require_sha256(model_raw,policy['model_sha256']); model=json.loads(model_raw)
    expected={r['id'] for r in manifest['fresh_dev']}
    if len(expected)!=64 or manifest['role']!='fresh_dev64': raise ValueError('exactly64 frozen freshDEV images required')
    rows=[]; sources={}
    for w in (0,1):
        folder=args.root/f'worker{w}'; protocol=json.loads((folder/'protocol.json').read_text())
        require_sha256(manifest_raw,protocol['manifest_sha256'])
        complete=json.loads((folder/'complete.json').read_text())
        if complete['source_inferences']!=32 or complete['pseudo_loss_forwards']!=0 or not complete['reproducibility']['exact_packet_and_pixels']:
            raise ValueError('fresh worker source/reproducibility/cost drift')
        rows.extend(json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes())))
        for r in json.loads((folder/'sources.json').read_text()):
            if r['image_id'] in sources: raise ValueError('overlapping workers')
            sources[r['image_id']]=r
    if set(sources)!=expected: raise ValueError('source coverage mismatch')
    audit=audit_grid(rows,expected); groups=defaultdict(list)
    for i,row in enumerate(rows):
        if row['bytes']!=row['elementary_bytes']+row['metadata_bytes'] or min(row['bytes'],row['metadata_bytes'])<=0: raise ValueError('paid byte mismatch')
        groups[(row['image_id'],row['codec'],row['qp'])].append(i)
    for ix in groups.values(): ix.sort(key=lambda i:rows[i]['profile'])
    feature_started=time.perf_counter(); x=[]
    for row in rows:
        anchor=rows[groups[(row['image_id'],row['codec'],row['qp'])][0]]
        names,values=features(row,anchor,sources[row['image_id']])
        if names!=model['features']: raise ValueError('frozen feature schema drift')
        x.append(values)
    feature_seconds=time.perf_counter()-feature_started; critic_started=time.perf_counter()
    predicted=predict_forest(model,x); critic_seconds=time.perf_counter()-critic_started; selection_started=time.perf_counter()
    selected={name:[] for name in ('anchor','a2','all21_guard','risk_forest')}
    for key,indices in sorted(groups.items()):
        group=[rows[i] for i in indices]; selected['anchor'].append(indices[0])
        for name in ('a2','all21_guard'):
            allowed=[i for i in indices if name=='all21_guard' or rows[i]['profile'] in group[0]['a2_shortlist']]
            chosen,_=select_measured([rows[i]['bytes']/group[0]['bytes'] for i in allowed],
                [[None,rows[i]['current_cost'],None] for i in allowed],[False,True,False],[0,.05,0],budget=1.,quality_weight=.05)
            selected[name].append(allowed[chosen])
        choice=choose_predicted(group,[predicted[i] for i in indices],policy['threshold'])
        selected['risk_forest'].append(next(i for i in indices if rows[i]['profile']==choice))
    decisions=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),frozen_policy_sha256=hashlib.sha256(policy_raw).hexdigest(),
        scope='all sender decisions complete before real GT annotation staging/scoring',selected={name:[
            dict(image_id=rows[i]['image_id'],codec=rows[i]['codec'],qp=rows[i]['qp'],profile=rows[i]['profile'],bytes=rows[i]['bytes']) for i in indices] for name,indices in selected.items()})
    decision_raw=(json.dumps(decisions,indent=2,allow_nan=False)+'\n').encode(); (args.root/'source-only-selections.json').write_bytes(decision_raw)
    payload=[dict(image_id=r['image_id'],codec=r['codec'],qp=r['qp'],profile=r['profile'],predicted_damage=float(predicted[i])) for i,r in enumerate(rows)]
    (args.root/'risk-predictions.json.gz').write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    sender_seconds=dict(feature_extraction=feature_seconds,critic_inference=critic_seconds,selection_and_audit_write=time.perf_counter()-selection_started,
        total_pre_gt=time.perf_counter()-started,scope='CPU sender replay only; excludes actual codec/task-model work already measured by workers; total includes validation/loading')
    # Only now stage cohort GT for actual official COCO AP.
    raw=args.annotation.read_bytes(); protocol=json.loads((ROOT/'configs/v34r1-a4-od-risk.json').read_text())
    require_sha256(raw,protocol['annotation_sha256']); meta=json.loads(raw); gt=defaultdict(list)
    for a in meta['annotations']:
        if a['image_id'] in expected: gt[a['image_id']].append(a)
    results={}; ids=sorted(expected)
    for name,indices in selected.items():
        curves={c:dict(rate=[],quality=[]) for c in CODECS}; logratio={c:[] for c in CODECS}; counts=Counter()
        for c in CODECS:
            for q in QPS:
                subset=[rows[i] for i in indices if rows[i]['codec']==c and rows[i]['qp']==q]
                if len(subset)!=64: raise ValueError('incomplete selected condition')
                curves[c]['rate'].append(float(np.mean([r['bpp'] for r in subset])))
                curves[c]['quality'].append(metric([dict(p) for r in subset for p in r['coco']],ids,meta,gt)['AP'])
                for r in subset:
                    anchor=rows[groups[(r['image_id'],c,q)][0]]; logratio[c].append(math.log(r['bytes']/anchor['bytes'])); counts[r['profile']]+=1
        results[name]=dict(curves=curves,log_rate_ratio_by_codec={c:float(np.mean(v)) for c,v in logratio.items()},selection_counts=dict(counts))
        print('A4_FRESH_METRIC',name,flush=True)
    for name,result in results.items():
        result['bd']={c:assess_curve(results['anchor']['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
    gate=fresh_rd_gate(results['anchor']['curves'],results['risk_forest']['curves'],results['risk_forest']['log_rate_ratio_by_codec'],
        frozen['calibration_scales'],frozen['fresh_dev_gate']['per_codec_max_utility_exclusive'])
    report=dict(scope='one fresh64-image DEV confirmation after model/policy lock; TEST512 never inferred or GT-scored',grid=audit,
        frozen_policy_sha256=hashlib.sha256(policy_raw).hexdigest(),source_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        source_only_decisions_sha256=hashlib.sha256(decision_raw).hexdigest(),results=results,resource_gate=gate,
        no_retuning=True,test_scored=False,critic_predictions=len(rows),sender_seconds=sender_seconds,wall_seconds=time.perf_counter()-started)
    (args.root/'fresh-dev-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A4_FRESH_COMPLETE',json.dumps(dict(resource_gate=gate,test_scored=False,seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__': main()
