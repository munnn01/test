"""Frozen A2 source-only AR decisions on exactly125 new videos per GPU."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import torch
from ..v34.checkpoint import load_checkpoint
from ..v34.preprocessing import region_maps
from ..v34.runner import input_pair
from ..v34r1.runner import records,evaluate_condition,load_config,summarize
from ..v34r1.tasks import Heads
from ..v34r1_a8.contracts import audit_selected
from .evidence import ForwardMeter,packet_file

ROOT=Path(__file__).resolve().parents[2]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing') or not torch.cuda.is_available():raise ValueError('Kaggle GPU only')
    if a.out.exists():raise ValueError('fresh immutable AR worker output required')
    a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(303401);torch.backends.cudnn.benchmark=False
    import cv2
    cv2.setNumThreads(1)
    cfg=load_config(ROOT/'configs/v34r1-kaggle.json');cfg['ar_bootstrap_draws']=0 # CI belongs to merged1000, not each125 shard.
    manifest=json.loads(a.manifest.read_bytes());declared=manifest['ar']
    if len(declared)!=125 or len({r['id'] for r in declared})!=125:raise ValueError('exact125 new sources required')
    expected=['ar/'+r['id'] for r in declared];labels={'ar/'+r['id']:r['label'] for r in declared}
    model=load_checkpoint(ROOT/'checkpoints/v34_r1/controller_303401.pth','cuda')
    if model.checkpoint_sha256!=cfg['controller_sha256_by_task']['ar']:raise ValueError('frozen A2 controller drift')
    heads=Heads('cuda','ar_ot',cfg);models=heads.hashes();prior=json.loads((ROOT/'results/v34r1/ar-test/protocol.json').read_bytes())
    if models!=prior['task_models']:raise ValueError('frozen A2 task-model hashes or transforms changed')
    meter=ForwardMeter()
    for name,module in [('ar',heads.action.model),('detector',heads.detector.model),('controller',model)]:meter.attach(name,module)
    sink=lambda packet:packet_file(a.out,packet.data,packet.packet_sha256,packet.coded_bytes)
    protocol=dict(task='ar',config=cfg,controller_sha256=model.checkpoint_sha256,task_models=models,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        expected_ids=expected,gpu=torch.cuda.get_device_name(0),scope='unchanged A2 choices; only defer receiver GT until all source-only decisions saved')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[];repeats=[];started=time.perf_counter()
    for index,record in enumerate(records(manifest,'ar','test',cfg)):
        if hashlib.sha256(record.rgb.tobytes()).hexdigest()!=record.item['canonical_sha256']:raise ValueError('global canonical AR source hash drift')
        meter.stage('source');begin=time.perf_counter();heads.source(record);maps=region_maps(record.rgb,record.source_predictions['od']);pair=input_pair(record.rgb,maps);preparation=time.perf_counter()-begin;filtered={}
        for c in cfg['codecs']:
            for q in cfg['qps']:
                values=evaluate_condition(record,model,heads,maps,pair,c,q,cfg,filtered_cache=filtered,defer_ar_labels=True,stage_callback=meter.stage,packet_sink=sink)
                if index==0 and q==30:
                    again=evaluate_condition(record,model,heads,maps,pair,c,q,cfg,filtered_cache={},defer_ar_labels=True,stage_callback=lambda name:meter.stage('repeat/'+name))
                    for old,new in zip(values,again):
                        for field in ('id','codec','qp','arm','action','bytes','elementary_bytes','metadata_bytes','packet_sha256','decoded_sha256','predicted_label'):
                            if old[field]!=new[field]:raise ValueError('independent AR condition repeat drift')
                    repeats.append(dict(codec=c,qp=q,packet_pixels_predictions_identical=True,extra_candidate_encodes=next(r['candidate_encode_count'] for r in again if r['arm']=='adaptive')))
                for row in values:
                    if row['arm']=='adaptive':row['source_preparation_seconds_shared_across_8_conditions']=preparation
                rows.extend(values)
        partial=(json.dumps(rows,allow_nan=False)+'\n').encode();(a.out/'source-rows.partial.json.gz').write_bytes(gzip.compress(partial,mtime=0))
        print('A8_AR_SOURCE',index+1,record.id,len(rows),round(time.perf_counter()-started,1),flush=True)
    audit_selected(rows,expected)
    decision=(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),scope='all AR sender decisions and receiver predictions before GT correctness',rows=rows),allow_nan=False)+'\n').encode()
    (a.out/'source-only-selections.json.gz').write_bytes(gzip.compress(decision,mtime=0))
    for row in rows:row['correct']=int(row['predicted_label']==labels[row['id']])
    if heads.hashes()!=models:raise ValueError('frozen task models mutated')
    summary=summarize(rows,'ar',manifest,cfg,a.out,{}, {},None,expected)
    if len(repeats)!=2:raise ValueError('both codec independent repeats required')
    if (meter.counts['source/ar_calls']!=125 or meter.counts['source/detector_units']!=2000
        or meter.counts['controller/controller_calls']!=1000 or meter.counts['repeat/controller/controller_calls']!=2
        or any(k.startswith('unclassified/') for k in meter.counts)):raise ValueError('actual source/controller/repeat inference accounting drift')
    meter.close()
    summary.update(source_only_decisions_sha256=hashlib.sha256(decision).hexdigest(),wall_seconds=time.perf_counter()-started,scope='AR125 shard; full1000 CI computed only after complete merge',retuned=False,actual_model_work=dict(meter.counts),
        independent_condition_repeats=repeats,actual_candidate_encodes=sum(r['candidate_encode_count'] for r in rows if r['arm']=='adaptive')+sum(r['extra_candidate_encodes'] for r in repeats))
    (a.out/'rows.json.gz').write_bytes(gzip.compress(json.dumps(rows,allow_nan=False).encode(),mtime=0));(a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('A8_AR_WORKER_COMPLETE',len(declared),flush=True)


if __name__=='__main__':main()
