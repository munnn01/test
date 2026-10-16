"""Replay source-only causal policies on cached DEV detections, then official GT HOTA."""
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
from .core import QPS,CODECS,fit_threshold
from .tracking import CausalProxyGuard
from .ot_cache import mot17_plan
from ..v32.tracking import load_trackeval,stage_ground_truth,write_mot,json_safe
from ..v33ap.sources import serial
from ..v34r1.policy import select_measured
from ..v34r1.metrics import assess_curve

ROOT=Path(__file__).resolve().parents[2]


def arrays(predictions):
    return [{k:np.asarray(v,dtype=int if k=='labels' else float).reshape(-1,4) if k=='boxes'
             else np.asarray(v,dtype=int if k=='labels' else float) for k,v in p.items()} for p in predictions]


def official(package,plans,out,tracker_names):
    dataset=package.datasets.MotChallenge2DBox(dict(GT_FOLDER=str(out/'ground_truth'),TRACKERS_FOLDER=str(out/'trackers'),
        OUTPUT_FOLDER=str(out/'official_metrics'),TRACKERS_TO_EVAL=tracker_names,BENCHMARK='MOT17',SPLIT_TO_EVAL='train',
        SKIP_SPLIT_FOL=True,DO_PREPROC=True,PRINT_CONFIG=False,SEQ_INFO={r['name']:r['frames'] for r in plans}))
    evaluator=package.Evaluator(dict(PRINT_RESULTS=False,PRINT_CONFIG=False,TIME_PROGRESS=False,USE_PARALLEL=False,
        BREAK_ON_ERROR=True,LOG_ON_ERROR=None,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,PLOT_CURVES=False))
    cfg=dict(PRINT_CONFIG=False,THRESHOLD=.5)
    with contextlib.redirect_stdout(io.StringIO()):
        results,messages=evaluator.evaluate([dataset],[package.metrics.HOTA(),package.metrics.CLEAR(cfg),package.metrics.Identity(cfg)])
    raw=results[dataset.get_name()]
    (out/'trackeval-raw.json').write_text(json.dumps(json_safe(dict(results=raw,messages=messages)),allow_nan=False))
    def extract(value):
        person=value['pedestrian']; hota=person['HOTA']
        return dict(HOTA=float(np.mean(hota['HOTA'])),DetA=float(np.mean(hota['DetA'])),AssA=float(np.mean(hota['AssA'])),
                    IDF1=float(person['Identity']['IDF1']),MOTA=float(person['CLEAR']['MOTA']))
    return {n:dict(combined=extract(raw[n]['COMBINED_SEQ']),sequences={r['name']:extract(raw[n][r['name']]) for r in plans}) for n in tracker_names}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--sources',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only analysis')
    protocol=json.loads((ROOT/'configs/v34r1-a3-ot-probe.json').read_text())
    package=load_trackeval(ROOT/'data/TrackEval'); rows=[]; sources={}
    for worker in (0,1):
        folder=args.root/f'worker{worker}'
        if not (folder/'complete.json').is_file(): raise ValueError('incomplete candidate cache')
        rows.extend(json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes())))
        found=json.loads(gzip.decompress((folder/'sources.json.gz').read_bytes()))
        if set(sources)&set(found): raise ValueError('worker sequence overlap')
        sources.update(found)
    groups=defaultdict(list)
    for r in rows: groups[(r['sequence'],r['start'],r['codec'],r['qp'])].append(r)
    seen=set()
    for r in rows:
        key=(r['sequence'],r['start'],r['codec'],r['qp'],r['action'])
        if key in seen: raise ValueError('duplicate candidate row')
        seen.add(key)
    expected={(s,start,c,q) for s,v in sources.items() for start in range(1,v['frames']+1,16) for c in CODECS for q in QPS}
    if set(groups)!=expected: raise ValueError('incomplete OT source/window/four-QP grid')
    for group in groups.values():
        actions=group[0]['actions']; table={r['action']:r for r in group}
        if not actions or actions[0]!=0 or set(table)!=set(actions) or len(actions)!=len(set(actions)) or any(r['actions']!=actions for r in group):
            raise ValueError('candidate bank differs from frozen A2 shortlist')
        group.sort(key=lambda r:actions.index(r['action']))
        for r in group: r['detections']=arrays(r['detections'])
    for value in sources.values(): value['source_predictions']=arrays(value['source_predictions'])
    splits={name:sorted(s for s,v in sources.items() if v['split']==name) for name in ('fit','validation')}
    if len(splits['fit'])!=2 or len(splits['validation'])!=2: raise ValueError('two complete independent sequences per DEV split required')
    cache={}
    def evaluate(split,policy):
        key=(split,json.dumps(policy,sort_keys=True))
        if key in cache: return cache[key]
        tag=policy['kind']+(('-'+str(policy['threshold']).replace('.','p')) if 'threshold' in policy else '')
        out=args.root/'tracking'/split/tag; out.mkdir(parents=True)
        rates=defaultdict(list); ratios=[]; counts=Counter(); selections=[]
        plans=[mot17_plan(args.sources,name,sources[name]['frames']) for name in splits[split]]
        for name in splits[split]:
            source=sources[name]
            for codec in CODECS:
                for qp in QPS:
                    metric=package.metrics.HOTA() if policy['kind'] in ('observed_prefix','observed_trailing32') else None
                    state=CausalProxyGuard(metric,fps=source['fps'],lookback=32 if policy['kind']=='observed_trailing32' else None)
                    frames=[]
                    for offset in range(0,source['frames'],16):
                        group=groups[(name,offset+1,codec,qp)]
                        state.begin(name,source['source_predictions'][offset:offset+16])
                        trial_rows=group[:1] if policy['kind']=='anchor' else group
                        trials=[state.trial(r['detections']) for r in trial_rows]
                        rate=[r['bytes']/group[0]['bytes'] for r in trial_rows]
                        if policy['kind']=='anchor' or (metric is not None and trials[0].proxy_costs is None): choice=0
                        else:
                            if policy['kind']=='a2_native':
                                costs=[[None,None,t.cost] for t in trials]; mask=[False,False,True]; limits=[0,0,.05]
                            else:
                                costs=[t.proxy_costs for t in trials]; mask=[True]*3; limits=[policy['threshold']]*3
                            choice,_=select_measured(rate,costs,mask,limits,budget=1.,quality_weight=.05)
                        selected=trial_rows[choice]; trial=trials[choice]; state.commit(trial); frames.extend(trial.frames)
                        rates[(codec,qp)].append(selected['bytes']*8/(16*960*960)); ratios.append(math.log(selected['bytes']/group[0]['bytes']))
                        counts[selected['action']]+=1
                        selections.append(dict(sequence=name,start=offset+1,codec=codec,qp=qp,action=selected['action'],bytes=selected['bytes'],
                            legacy_source_cost=trial.cost,source_hota=trial.source_metrics))
                    write_mot(out/'trackers'/f'{codec}_{qp}'/'data'/f'{name}.txt',frames,source['geometry'],source['width'],source['height'])
        # Ground truth is staged/scored only after this complete source-only replay.
        gt_manifest=stage_ground_truth(plans,out/'ground_truth')
        (out/'ground-truth-hashes.json').write_text(json.dumps(gt_manifest,indent=2))
        metrics=official(package,plans,out,[f'{c}_{q}' for c in CODECS for q in QPS])
        curves={c:dict(rate=[float(np.mean(rates[(c,q)])) for q in QPS],quality=[metrics[f'{c}_{q}']['combined']['HOTA'] for q in QPS]) for c in CODECS}
        result=dict(policy=policy,curves=curves,metrics=metrics,selection_counts=dict(counts),selected=selections,log_rate_ratio=float(np.mean(ratios)))
        cache[key]=result
        print('OT_REPLAY_DONE',split,tag,result['log_rate_ratio'],flush=True)
        return result
    anchor_policy=dict(kind='anchor'); anchor=evaluate('fit',anchor_policy)
    anchor_quality=[v for c in CODECS for v in anchor['curves'][c]['quality']]; fits={}; frozen={}
    for mode in protocol['proxy_modes']:
        candidates=[]
        for threshold in protocol['thresholds']:
            result=evaluate('fit',dict(kind=mode,threshold=threshold))
            candidates.append(dict(threshold=threshold,quality=[v for c in CODECS for v in result['curves'][c]['quality']],log_rate_ratio=result['log_rate_ratio']))
        selected=fit_threshold(candidates,anchor_quality)
        frozen[mode]=dict(kind=mode,threshold=selected['threshold']) if selected else anchor_policy
        fits[mode]=dict(candidates=candidates,selected=selected)
    policy_bytes=(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),fit_only=True,policies=frozen,fit=fits),indent=2)+'\n').encode()
    (args.root/'fit-frozen-policies.json').write_bytes(policy_bytes)
    report=dict(scope='fresh MOT17 exploratory DEV; zero reserved TEST inference',source_protocol=protocol,
                grid=dict(sequences=4,windows=16,conditions=len(groups),candidate_rows=len(rows),qps=list(QPS),codecs=list(CODECS)),
                fit=fits,fit_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),results={},test_scored=False)
    for split in ('fit','validation'):
        report['results'][split]={}; anchor=evaluate(split,anchor_policy)
        for name,policy in dict(anchor=anchor_policy,a2_native=dict(kind='a2_native'),**frozen).items():
            result=evaluate(split,policy)
            result['bd']={c:assess_curve(anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
            report['results'][split][name]=result
    report['validation_both_codecs_below_minus15']={name:all(v['point_passes'] for v in report['results']['validation'][name]['bd'].values()) for name in frozen}
    (args.root/'dev-report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False)+'\n')
    print('OT_VALIDATION_GATE',json.dumps(report['validation_both_codecs_below_minus15']),flush=True)


if __name__=='__main__': main()
