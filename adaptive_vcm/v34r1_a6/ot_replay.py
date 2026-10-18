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
from ..v34r1_a5.ot_choice import choose_lagrangian
from .contracts import audit_ot21,scale4_gate
from .evidence import publication
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
    protocol_raw=(ROOT/'configs/v34r1-a6-ot21.json').read_bytes();protocol=json.loads(protocol_raw)
    old_root=ROOT/'results/v34r1_a3/ot-dev';a5_root=ROOT/'results/v34r1_a5/ot-soft'
    publication(old_root,protocol['parent_a3_publication_sha256']);publication(a5_root,protocol['parent_a5_publication_sha256'])
    parent=json.loads((old_root/'dev-report.json').read_text());a5=json.loads((a5_root/'replay-report.json').read_text())
    from ..v34r1_a4.cloud_record import unpack_archive
    a5_archive=ROOT/'data/v34r1_a6/a5-parity';unpack_archive(a5_root/'v34r1-a5-ot-soft.tgz',a5_archive);a5_trackers=a5_archive/'a5-ot-soft'
    manifest=json.loads((old_root/'source-manifest.json').read_text());declared={v['name']:v for v in manifest['sequences'] if v['split']!='reserved_test'}
    expected_splits={r['name']:r['split'] for r in protocol['pilot_sequences']}
    if set(declared)!=set(expected_splits):raise ValueError('exact DEV4 source names required')
    rows=[];sources={};cached_cost=[]
    bundle_raw=(args.cache/'bundle-manifest.json').read_bytes();bundle=json.loads(bundle_raw)
    if set(bundle['sequences'])!=set(expected_splits) or bundle['qps']!=list(QPS) or bundle['profiles']!=list(range(21)):raise ValueError('declared four-video cache bundle required')
    for name in sorted(expected_splits):
        part=args.cache/bundle['sequences'][name]['folder'];status=json.loads((part/'cloud-status.json').read_text())
        if not part.resolve().is_relative_to(args.cache.resolve()):raise ValueError('private cache folder escape')
        if (not status['complete'] or status['test_scored'] or status['gt_scored'] or status['sequence']!=name
            or status['commit']!='60b21868c74e81aea231f51489f0d91c3665304e'):raise ValueError('complete source-only pinned cloud cache required')
        for entry in bundle['sequences'][name]['files']:
            path=(part/entry['path']).resolve()
            if not path.is_relative_to(part.resolve()) or path.stat().st_size!=entry['bytes']:raise ValueError('private cache file/path drift')
            require_sha256(path.read_bytes(),entry['sha256'])
        for w,c in enumerate(CODECS):
            folder=part/f'worker{w}';complete=json.loads((folder/'complete.json').read_text());cached_cost.append(complete)
            repeat=complete['independent_anchor_repeat']
            if (complete['gt_extracted'] or complete['tracker_calls']!=0 or complete['controller_calls']!=0 or not complete['old_candidate_checks']
                or complete['model_sha256']!='c4616e10536fb767f901700c621bb851a10259b6ed90f3af0e4596d5a6886533'
                or len(repeat)!=1 or not repeat[0]['packet_identical'] or not repeat[0]['pixels_identical']
                or not 0<=repeat[0]['max_abs_error']<=1e-6 or not 0<=complete['source_prediction_max_abs_error']<=1e-6
                or any(not 0<=v['max_abs_error']<=1e-6 for v in complete['old_candidate_checks'])):
                raise ValueError('complete source/model/old-bank/independent-repeat proof required')
            found=json.loads(gzip.decompress((folder/'sources.json.gz').read_bytes()))
            if set(found)!={name} or complete['sequence']!=name or complete['codec']!=c:raise ValueError('one-video complete codec shard required')
            if name in sources and sources[name]!=found[name]:raise ValueError('duplicate source forwards disagree across codec workers')
            sources[name]=found[name];part_rows=json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes()))
            if len(part_rows)!=complete['rows']:raise ValueError('worker row count mismatch')
            audit_ot21(part_rows,{name:expected_splits[name]},codec=c);rows.extend(part_rows)
    audit_ot21(rows,expected_splits)
    splits={split:sorted(n for n,s in sources.items() if s['split']==split) for split in ('fit','validation')}
    if any(len(names)!=2 for names in splits.values()):raise ValueError('two physical videos per DEV split required')
    groups=defaultdict(list)
    for r in rows:groups[(r['sequence'],r['start'],r['codec'],r['qp'])].append(r)
    old_actions={}
    for w in (0,1):
        for r in json.loads(gzip.decompress((old_root/f'worker{w}/rows.json.gz').read_bytes())):
            k=(r['sequence'],r['start'],r['codec'],r['qp'])
            if k in old_actions and old_actions[k]!=r['actions']:raise ValueError('old shortlist drift')
            old_actions[k]=r['actions']
    for k,group in groups.items():
        group.sort(key=lambda r:r['action'])
        if any(r['old_actions']!=old_actions[k] for r in group):raise ValueError('authenticated old shortlist required')
        for r in group:r['detections']=arrays(r['detections'])
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

    def evaluate(split,weight,mode='full'):
        key=(split,weight,mode)
        if key in replay_cache: return replay_cache[key]
        tag='anchor' if weight is None else mode+'-lambda-'+str(weight)
        out=args.out/split/tag; out.mkdir(parents=True); selections=[]; rates=defaultdict(list); ratios=defaultdict(list); counts=Counter(); plans=[]
        replay_started=time.perf_counter(); proxy_trials=0
        for name in splits[split]:
            s=sources[name]; plans.append(plans_by_name[name])
            for c in CODECS:
                for q in QPS:
                    state=CausalProxyGuard(package.metrics.HOTA() if weight is not None else None,fps=s['fps'],lookback=32); frames=[]
                    for offset in range(0,s['frames'],16):
                        group=groups[(name,offset+1,c,q)]; state.begin(name,s['source_predictions'][offset:offset+16])
                        subset=group[:1] if weight is None else (group if mode=='full' else [next(r for r in group if r['action']==a) for a in old_actions[(name,offset+1,c,q)]])
                        trials=[state.trial(r['detections']) for r in subset]
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
            split=split,weight=weight,mode=mode,selected=selections),indent=2,allow_nan=False)+'\n').encode()
        (out/'source-only-selections.json').write_bytes(decision_raw); replay_seconds=time.perf_counter()-replay_started
        if weight is None: assert_same_files(out/'trackers',a5_trackers/split/'anchor/trackers')
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
        if mode=='shortlist' and weight==32:
            prior=a5['fit_candidates']['32'] if split=='fit' else a5['calibration']['selected']
            if result['curves']!=prior['curves'] or result['metrics']!=prior['metrics']:raise ValueError('exact A5 lambda32 old-shortlist metrics parity failed')
            assert_same_files(out/'trackers',a5_trackers/split/'lambda-32/trackers')
        replay_cache[key]=result; print('A6_OT_REPLAY',split,tag,result['log_rate_ratio'],flush=True); return result

    fit_anchor=evaluate('fit',None); fits={}; candidates=[]
    for weight in protocol['lambdas']:
        result=evaluate('fit',weight); result['bd']={c:assess_curve(fit_anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}; fits[str(weight)]=result
        candidates.append(dict(threshold=weight,quality=[v for c in CODECS for v in result['curves'][c]['quality']],log_rate_ratio=result['log_rate_ratio']))
    selection=select_rd_fit(fit_anchor['curves'],candidates); chosen=selection['selected']
    weight=chosen['threshold'] if chosen is not None else None
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),fit_only=True,bank_profiles=21,protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
        policy=dict(kind='source_hota_lagrangian',weight=weight) if weight is not None else dict(kind='anchor'),
        fit_selection=selection,calibration_reused=True,independent_validation=False,test_scored=False)
    frozen_raw=(json.dumps(frozen,indent=2,allow_nan=False)+'\n').encode(); (args.out/'fit-frozen-policy.json').write_bytes(frozen_raw)
    # Exactly one policy reaches reused calibration, after the above lock.
    cal_anchor=evaluate('validation',None); cal=evaluate('validation',weight)
    cal['bd']={c:assess_curve(cal_anchor['curves'][c],cal['curves'][c],qps=QPS) for c in CODECS}
    # Controls are prespecified and never select another CAL policy or codec-specific winner.
    controls={}
    for split in ('fit','validation'):
        control32=evaluate(split,32,'shortlist')
        same=evaluate(split,weight,'shortlist') if weight is not None else evaluate(split,None)
        anchor=fit_anchor if split=='fit' else cal_anchor
        for result in (control32,same):result['bd']={c:assess_curve(anchor['curves'][c],result['curves'][c],qps=QPS) for c in CODECS}
        controls[split]=dict(old_a5_lambda32=control32,same_lambda_shortlist=same)
    blocks=dict(grid=True,policy_lock=True,source_provenance=True,paid_header=True,independent_decode=True,
        source_only=True,no_substitution=True,measurement_reproducibility=True)
    gate=scale4_gate(cal['bd'],blocks)
    report=dict(scope='full21 FIT-only selection and reusedCAL; no reserved or external video inference',resource_gate=gate,blocks=blocks,protocol=protocol,protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),fit_anchor=fit_anchor,fit_candidates=fits,
        frozen=frozen,frozen_policy_sha256=hashlib.sha256(frozen_raw).hexdigest(),calibration=dict(anchor=cal_anchor,selected=cal),
        calibration_both_codecs_below_minus15=all(v['point_passes'] for v in cal['bd'].values()),independent_validation=False,
        exact_anchor_tracker_bytes_and_metrics=True,anchor_byte_reference='authenticated A5 archive; official metrics also match A3',
        new_model_inferences=0,new_encodes=0,parent_cached_cost=cached_cost,cache_bundle_sha256=hashlib.sha256(bundle_raw).hexdigest(),controls=controls,full21_rows=len(rows),
        test_scored=False,expanded_evaluation_not_executed=True,wall_seconds=time.perf_counter()-started)
    (args.out/'replay-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A6_OT_COMPLETE',json.dumps(dict(policy=frozen['policy'],calibration_both_codecs_below_minus15=report['calibration_both_codecs_below_minus15'],test_scored=False,seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__': main()
