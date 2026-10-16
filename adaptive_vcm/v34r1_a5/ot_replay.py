"""Kaggle-only FIT selection then one locked replay on reused OT calibration."""
import argparse
from collections import defaultdict,Counter
import configparser
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .ot_choice import choose_lagrangian
from ..v34r1_a4.cloud_record import assert_same_files,require_sha256
from ..v34r1_a4.rd_fit import select_rd_fit
from ..v34r1_a3.ot_analyze import arrays,official
from ..v34r1_a3.tracking import CausalProxyGuard
from ..v34r1_a3.core import QPS,CODECS
from ..v34r1.metrics import assess_curve
from ..v32.tracking import load_trackeval,stage_ground_truth,write_mot

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only actual tracking and official GT HOTA')
    if args.out.exists(): raise ValueError('fresh immutable output required')
    protocol_raw=(ROOT/'configs/v34r1-a5-ot-soft.json').read_bytes(); protocol=json.loads(protocol_raw)
    parent_raw=(args.cache/'dev-report.json').read_bytes(); require_sha256(parent_raw,protocol['parent_report_sha256']); parent=json.loads(parent_raw)
    status=json.loads((args.cache/'cloud-status.json').read_text())
    if not status['complete'] or status['test_scored'] or status['commit']!=protocol['parent_commit']: raise ValueError('complete authenticated parent required')
    source_raw=(args.cache/'source-manifest.json').read_bytes(); require_sha256(source_raw,parent['source_protocol']['source_manifest_sha256'])
    manifest=json.loads(source_raw); declared={v['name']:v for v in manifest['sequences'] if v['split']!='reserved_test'}
    rows=[]; sources={}; cached_cost=[]
    for w in (0,1):
        folder=args.cache/f'worker{w}'; completed=json.loads((folder/'complete.json').read_text()); cached_cost.append(completed)
        part=json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes()))
        if len(part)!=completed['rows']: raise ValueError('parent worker row count mismatch')
        rows.extend(part); found=json.loads(gzip.decompress((folder/'sources.json.gz').read_bytes()))
        if set(found)&set(sources): raise ValueError('overlapping worker source sequences')
        sources.update(found)
    if set(sources)!=set(declared): raise ValueError('DEV-only complete source set required')
    splits={split:sorted(n for n,s in sources.items() if s['split']==split) for split in ('fit','validation')}
    if any(len(names)!=2 for names in splits.values()): raise ValueError('two physical sequences per split required')
    groups=defaultdict(list); seen=set()
    for r in rows:
        key=(r['sequence'],r['start'],r['codec'],r['qp'],r['action'])
        if key in seen or r['split']!=sources[r['sequence']]['split']: raise ValueError('duplicate or mislabeled cached candidate')
        seen.add(key); groups[key[:4]].append(r)
    expected={(n,i,c,q) for n,s in sources.items() for i in range(1,s['frames']+1,16) for c in CODECS for q in QPS}
    if set(groups)!=expected or len(rows)!=772: raise ValueError('complete frozen four-QP bank required')
    for group in groups.values():
        order=group[0]['actions']
        if not order or order[0]!=0 or len(order)!=len(set(order)) or set(order)!={r['action'] for r in group}: raise ValueError('shortlist mismatch')
        if any(r['actions']!=order for r in group): raise ValueError('unaligned candidate banks')
        group.sort(key=lambda r:order.index(r['action']))
        for r in group:
            if r['bytes']!=r['elementary_bytes']+r['metadata_bytes'] or min(r['bytes'],r['metadata_bytes'])<=0: raise ValueError('unpaid bytes')
            r['detections']=arrays(r['detections'])
    plans_by_name={}
    for n,s in sources.items():
        d=declared[n]
        ini=configparser.ConfigParser()
        if not ini.read(args.sources/n/'seqinfo.ini'): raise ValueError('original native sequence metadata missing')
        seq=ini['Sequence']
        if (s['frames']!=d['frames'] or s['frames']!=64 or s['split']!=d['split'] or seq['name']!=n
            or any(int(seq[k])!=s[v] for k,v in [('frameRate','fps'),('imWidth','width'),('imHeight','height')])):
            raise ValueError('native source protocol drift')
        plans_by_name[n]=dict(name=n,path=args.sources/n,frames=s['frames'],full_frames=int(seq['seqLength']))
        s['source_predictions']=arrays(s['source_predictions'])
    package=load_trackeval(ROOT/'data/TrackEval'); args.out.mkdir(); started=time.perf_counter(); replay_cache={}

    def evaluate(split,weight):
        key=(split,weight)
        if key in replay_cache: return replay_cache[key]
        tag='anchor' if weight is None else 'lambda-'+str(weight)
        out=args.out/split/tag; out.mkdir(parents=True); selections=[]; rates=defaultdict(list); ratios=defaultdict(list); counts=Counter(); plans=[]
        replay_started=time.perf_counter(); proxy_trials=0
        for name in splits[split]:
            s=sources[name]; plans.append(plans_by_name[name])
            for c in CODECS:
                for q in QPS:
                    state=CausalProxyGuard(package.metrics.HOTA() if weight is not None else None,fps=s['fps'],lookback=32); frames=[]
                    for offset in range(0,s['frames'],16):
                        group=groups[(name,offset+1,c,q)]; state.begin(name,s['source_predictions'][offset:offset+16])
                        subset=group[:1] if weight is None else group; trials=[state.trial(r['detections']) for r in subset]
                        if weight is None: choice=0; audit=dict(anchor=True)
                        else:
                            proxy_trials+=len(trials)
                            costs=[t.proxy_costs[0] if t.proxy_costs is not None else None for t in trials]
                            choice,audit=choose_lagrangian([r['bytes']/group[0]['bytes'] for r in subset],costs,weight,[r['action'] for r in subset])
                        selected=subset[choice]; trial=trials[choice]; state.commit(trial); frames.extend(trial.frames)
                        rates[(c,q)].append(selected['bytes']*8/(16*960*960)); ratios[c].append(math.log(selected['bytes']/group[0]['bytes'])); counts[selected['action']]+=1
                        selections.append(dict(sequence=name,start=offset+1,codec=c,qp=q,action=selected['action'],bytes=selected['bytes'],
                            source_hota=trial.source_metrics,identity_source_hota=trials[0].source_metrics,candidate_actions=[r['action'] for r in subset],audit=audit))
                    write_mot(out/'trackers'/f'{c}_{q}'/'data'/f'{name}.txt',frames,s['geometry'],s['width'],s['height'])
        decision_raw=(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),scope='this replay decisions before its GT scoring; no future or GT policy inputs',
            split=split,weight=weight,selected=selections),indent=2,allow_nan=False)+'\n').encode()
        (out/'source-only-selections.json').write_bytes(decision_raw); replay_seconds=time.perf_counter()-replay_started
        if weight is None: assert_same_files(out/'trackers',args.cache/'tracking'/split/'anchor/trackers')
        # All decisions for this policy are saved before staging its GT evaluator.
        gt_started=time.perf_counter(); gt=stage_ground_truth(plans,out/'ground_truth')
        (out/'ground-truth-hashes.json').write_text(json.dumps(gt,indent=2)+'\n')
        metrics=official(package,plans,out,[f'{c}_{q}' for c in CODECS for q in QPS]); gt_seconds=time.perf_counter()-gt_started
        curves={c:dict(rate=[float(np.mean(rates[(c,q)])) for q in QPS],quality=[metrics[f'{c}_{q}']['combined']['HOTA'] for q in QPS]) for c in CODECS}
        if weight is None:
            old_anchor=parent['results'][split]['anchor']
            if curves!=old_anchor['curves'] or metrics!=old_anchor['metrics']:
                raise ValueError('exact A3 anchor curves and all combined/per-sequence metrics parity failed')
        result=dict(weight=weight,curves=curves,metrics=metrics,selected=selections,selection_counts=dict(counts),
            log_rate_ratio=float(np.mean([v for c in CODECS for v in ratios[c]])),log_rate_ratio_by_codec={c:float(np.mean(v)) for c,v in ratios.items()},
            source_only_decisions_sha256=hashlib.sha256(decision_raw).hexdigest(),sender_replay_seconds=replay_seconds,source_hota_trial_calls=proxy_trials,gt_seconds=gt_seconds)
        replay_cache[key]=result; print('A5_OT_REPLAY',split,tag,result['log_rate_ratio'],flush=True); return result

    fit_anchor=evaluate('fit',None); fits={}; candidates=[]
    for weight in protocol['lambdas']:
        result=evaluate('fit',weight); result['bd']={c:assess_curve(fit_anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}; fits[str(weight)]=result
        candidates.append(dict(threshold=weight,quality=[v for c in CODECS for v in result['curves'][c]['quality']],log_rate_ratio=result['log_rate_ratio']))
    selection=select_rd_fit(fit_anchor['curves'],candidates); chosen=selection['selected']
    weight=chosen['threshold'] if chosen is not None else None
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),fit_only=True,protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
        policy=dict(kind='source_hota_lagrangian',weight=weight) if weight is not None else dict(kind='anchor'),
        fit_selection=selection,calibration_reused=True,independent_validation=False,test_scored=False)
    frozen_raw=(json.dumps(frozen,indent=2,allow_nan=False)+'\n').encode(); (args.out/'fit-frozen-policy.json').write_bytes(frozen_raw)
    # Exactly one policy reaches reused calibration, after the above lock.
    cal_anchor=evaluate('validation',None); cal=evaluate('validation',weight)
    cal['bd']={c:assess_curve(cal_anchor['curves'][c],cal['curves'][c],qps=QPS) for c in CODECS}
    report=dict(scope=protocol['scope'],protocol=protocol,protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),fit_anchor=fit_anchor,fit_candidates=fits,
        frozen=frozen,frozen_policy_sha256=hashlib.sha256(frozen_raw).hexdigest(),calibration=dict(anchor=cal_anchor,selected=cal),
        calibration_both_codecs_below_minus15=all(v['point_passes'] for v in cal['bd'].values()),independent_validation=False,
        exact_anchor_tracker_bytes_and_metrics=True,new_model_inferences=0,new_encodes=0,parent_cached_cost=cached_cost,
        test_scored=False,test_authorized_by_this_run=False,wall_seconds=time.perf_counter()-started)
    (args.out/'replay-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A5_OT_COMPLETE',json.dumps(dict(policy=frozen['policy'],calibration_both_codecs_below_minus15=report['calibration_both_codecs_below_minus15'],test_scored=False,seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__': main()
