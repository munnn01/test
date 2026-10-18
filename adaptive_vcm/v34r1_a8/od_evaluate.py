"""Kaggle-only locked critic on streamed image caches; official AP after decisions."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np
from ..v34r1_a4.cloud_record import require_sha256
from ..v34r1_a4.risk import features,predict_forest,choose_predicted
from ..v34r1_a3.analyze import metric
from ..v34r1_a3.core import audit_grid
from ..v34r1.metrics import assess_curve
from .contracts import QPS,CODECS,audit_selected
from .evidence import packet_file

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--annotation',type=Path,required=True);a=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'):raise ValueError('Kaggle-only feature/critic inference and actual official AP')
    cfg=json.loads((ROOT/'configs/v34r1-a8-expansion.json').read_bytes());started=time.perf_counter();manifest_raw=(a.root/'source-manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    policy_raw=(ROOT/cfg['od']['policy_path']).read_bytes();require_sha256(policy_raw,cfg['od']['policy_sha256']);policy=json.loads(policy_raw)['policy']
    model_raw=(ROOT/'results/v34r1_a6/od-crossfit/training/models'/(policy['model']+'.json')).read_bytes();require_sha256(model_raw,policy['model_sha256']);model=json.loads(model_raw)
    expected={r['id'] for r in manifest['images']}
    if manifest['role']!='a8_shard250' or len(expected)!=250 or manifest['frozen_policy_sha256']!=cfg['od']['policy_sha256']:raise ValueError('complete locked250 cohort required')
    source_by_id={};owners={}
    for worker in (0,1):
        folder=a.root/f'worker{worker}';complete=json.loads((folder/'complete.json').read_bytes());wp=json.loads((folder/'protocol.json').read_bytes())
        require_sha256(manifest_raw,wp['manifest_sha256'])
        if complete['source_inferences']!=125 or complete['pseudo_loss_forwards']!=0 or not complete['reproducibility']['exact_packet_and_pixels']:raise ValueError('worker count/repeat failure')
        for row in json.loads((folder/'sources.json').read_bytes()):
            if row['image_id'] in source_by_id:raise ValueError('worker source overlap')
            source_by_id[row['image_id']]=row;owners[row['image_id']]=folder
    if set(source_by_id)!=expected:raise ValueError('incomplete sources')
    selected=[];risks=a.root/'risk-predictions';risks.mkdir();candidate_count=0
    for image_id in sorted(expected):
        raw=gzip.decompress((owners[image_id]/f'image-{image_id:012d}.json.gz').read_bytes());rows=json.loads(raw);audit_grid(rows,[image_id]);candidate_count+=len(rows)
        groups=defaultdict(list)
        for r in rows:
            if r['bytes']!=r['elementary_bytes']+40 or r['metadata_bytes']!=40:raise ValueError('paid candidate header drift')
            groups[(r['codec'],r['qp'])].append(r)
        saved=[];wire_path=owners[image_id]/f'image-{image_id:012d}-packets.zip';wire_archive=zipfile.ZipFile(wire_path)
        for (c,q),group in sorted(groups.items()):
            group.sort(key=lambda r:r['profile']);anchor=group[0];x=[]
            for row in group:
                names,values=features(row,anchor,source_by_id[image_id])
                if names!=model['features']:raise ValueError('locked302feature schema drift')
                x.append(values)
            predicted=predict_forest(model,x);action=choose_predicted(group,predicted,policy['threshold']);choice=next(r for r in group if r['profile']==action)
            for arm,row in [('anchor',anchor),('adaptive',choice)]:
                wire=wire_archive.read(row['wire_member']);path=packet_file(a.root,wire,row['packet_sha256'],row['bytes'])
                selected.append({k:row[k] for k in ('image_id','codec','qp','bytes','elementary_bytes','metadata_bytes','bpp','packet_sha256','decoded_sha256','coco')}|dict(id='od/coco2017/'+str(image_id),arm=arm,action=row['profile'],packet_path=path))
            saved.extend(dict(codec=c,qp=q,profile=row['profile'],bytes=row['bytes'],predicted_damage=float(value)) for row,value in zip(group,predicted))
        (risks/f'image-{image_id:012d}.json.gz').write_bytes(gzip.compress(json.dumps(saved,allow_nan=False).encode(),mtime=0))
        wire_archive.close();wire_path.unlink() # Selected wires are sealed; unselected temporary wires are no longer needed.
    audit_selected(selected,['od/coco2017/'+str(i) for i in sorted(expected)])
    # Seal all decisions and selected prediction payload before accessing scoring GT.
    decision=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),scope='locked250 source-only critic decisions before official GT',frozen_policy_sha256=cfg['od']['policy_sha256'],rows=selected)
    decision_raw=(json.dumps(decision,allow_nan=False)+'\n').encode();(a.root/'selected-streams.json.gz').write_bytes(gzip.compress(decision_raw,mtime=0))
    require_sha256(a.annotation.read_bytes(),cfg['od']['annotation_sha256']);meta=json.loads(a.annotation.read_bytes());gt=defaultdict(list)
    for annotation in meta['annotations']:
        if annotation['image_id'] in expected:gt[annotation['image_id']].append(annotation)
    curves={c:{arm:dict(rate=[],quality=[]) for arm in ('anchor','adaptive')} for c in CODECS}
    for c in CODECS:
        for q in QPS:
            for arm in ('anchor','adaptive'):
                subset=[r for r in selected if r['codec']==c and r['qp']==q and r['arm']==arm]
                if len(subset)!=250:raise ValueError('complete selected condition required')
                curves[c][arm]['rate'].append(float(np.mean([r['bpp'] for r in subset])))
                curves[c][arm]['quality'].append(metric([dict(p) for row in subset for p in row['coco']],sorted(expected),meta,gt)['AP'])
    report=dict(complete=True,scope='shard250 metrics only; full1000 requires merged official AP, not average shard AP',curves=curves,
        comparisons={c:assess_curve(curves[c]['anchor'],curves[c]['adaptive'],qps=QPS) for c in CODECS},candidate_rows=candidate_count,selected_rows=len(selected),
        policy=policy,source_only_decisions_sha256=hashlib.sha256(decision_raw).hexdigest(),frozen_policy_sha256=cfg['od']['policy_sha256'],
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),test_scored=True,retuned=False,wall_seconds=time.perf_counter()-started)
    (a.root/'evaluation-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('A8_OD_SHARD_COMPLETE',len(expected),flush=True)


if __name__=='__main__':main()
