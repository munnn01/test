"""Kaggle-only four-QP shared-profile probe; preserve original A6 controls."""
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
from ..v34r1_a6.contracts import audit_ot21,one_codec_gate
from .shared import choose_shared,select_valid_fit
from ..v34r1_a6.evidence import publication
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
    protocol_raw=(ROOT/'configs/v34r1-a7-ot-shared-qp.json').read_bytes();protocol=json.loads(protocol_raw)
    source_protocol=json.loads((ROOT/protocol['source_protocol']).read_text())
    if protocol['qps']!=list(QPS) or protocol['codecs']!=['h264'] or protocol['profiles']!=list(range(21)):
        raise ValueError('exact four-QP H264 shared21 protocol required')
    a6_root=ROOT/'results/v34r1_a6/ot-full21'
    publication(a6_root,protocol['parent_a6_publication_sha256'])
    a6=json.loads((a6_root/'replay-report.json').read_text())
    from ..v34r1_a4.cloud_record import unpack_archive
    a6_archive=ROOT/'data/v34r1_a7/a6-parity';unpack_archive(a6_root/'v34r1-a6-ot-full21.tgz',a6_archive)
    a6_trackers=a6_archive/'a6-ot-full21'
    old_root=ROOT/'results/v34r1_a3/ot-dev';a5_root=ROOT/'results/v34r1_a5/ot-soft'
    publication(old_root,source_protocol['parent_a3_publication_sha256']);publication(a5_root,source_protocol['parent_a5_publication_sha256'])
    parent=json.loads((old_root/'dev-report.json').read_text());a5=json.loads((a5_root/'replay-report.json').read_text())
    from ..v34r1_a4.cloud_record import unpack_archive
    a5_archive=ROOT/'data/v34r1_a6/a5-parity';unpack_archive(a5_root/'v34r1-a5-ot-soft.tgz',a5_archive);a5_trackers=a5_archive/'a5-ot-soft'
    manifest=json.loads((old_root/'source-manifest.json').read_text());declared={v['name']:v for v in manifest['sequences'] if v['split']!='reserved_test'}
    expected_splits={r['name']:r['split'] for r in source_protocol['pilot_sequences']}
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

    def evaluate(split,weight,mode='shared'):
        key=(split,weight,mode)
        if key in replay_cache:return replay_cache[key]
        tag='anchor' if weight is None else mode+'-lambda-'+str(weight)
        out=args.out/split/tag;out.mkdir(parents=True);selections=[];rates=defaultdict(list);ratios=[];counts=Counter();plans=[]
        replay_started=time.perf_counter();proxy_trials=0
        for name in splits[split]:
            s=sources[name];plans.append(plans_by_name[name]);frames={q:[] for q in QPS}
            states={q:CausalProxyGuard(package.metrics.HOTA() if weight is not None else None,fps=s['fps'],lookback=protocol['lookback']) for q in QPS}
            for offset in range(0,s['frames'],protocol['window_frames']):
                candidate_groups=[];all_trials=[]
                for q in QPS:
                    state=states[q];state.begin(name,s['source_predictions'][offset:offset+16])
                    group=groups[(name,offset+1,'h264',q)];subset=group[:1] if weight is None else group
                    trials=[state.trial(r['detections']) for r in subset];candidate_groups.append(subset);all_trials.append(trials)
                    if weight is not None:proxy_trials+=len(trials)
                if weight is None:chosen=[0]*4;audits=[dict(anchor=True)]*4
                elif mode=='shared':
                    action,audit=choose_shared([[r['bytes']/g[0]['bytes'] for r in g] for g in candidate_groups],
                        [[t.proxy_costs[0] if t.proxy_costs is not None else None for t in trials] for trials in all_trials],
                        weight,[r['action'] for r in candidate_groups[0]])
                    chosen=[next(i for i,r in enumerate(g) if r['action']==action) for g in candidate_groups];audits=[audit]*4
                else:
                    if mode!='independent' or weight!=32:raise ValueError('only authenticated A6 independent32 control allowed')
                    chosen=[];audits=[]
                    for g,trials in zip(candidate_groups,all_trials):
                        i,audit=choose_lagrangian([r['bytes']/g[0]['bytes'] for r in g],
                            [t.proxy_costs[0] if t.proxy_costs is not None else None for t in trials],weight,[r['action'] for r in g])
                        chosen.append(i);audits.append(audit)
                for q,g,trials,i,audit in zip(QPS,candidate_groups,all_trials,chosen,audits):
                    selected=g[i];trial=trials[i];states[q].commit(trial);frames[q].extend(trial.frames)
                    rates[q].append(selected['bytes']*8/(16*960*960));ratios.append(math.log(selected['bytes']/g[0]['bytes']));counts[selected['action']]+=1
                    selections.append(dict(sequence=name,start=offset+1,codec='h264',qp=q,action=selected['action'],bytes=selected['bytes'],
                        source_hota=trial.source_metrics,identity_source_hota=trials[0].source_metrics,candidate_actions=[r['action'] for r in g],audit=audit))
            for q in QPS:write_mot(out/'trackers'/f'h264_{q}'/'data'/f'{name}.txt',frames[q],s['geometry'],s['width'],s['height'])
        decision_raw=(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),scope='source-only joint four-QP decisions before GT; separate causal state per QP',
            split=split,weight=weight,mode=mode,selected=selections),indent=2,allow_nan=False)+'\n').encode()
        (out/'source-only-selections.json').write_bytes(decision_raw);replay_seconds=time.perf_counter()-replay_started
        prior=a6['fit_anchor'] if split=='fit' else a6['calibration']['anchor']
        old_tag='anchor'
        if mode=='independent':
            prior=a6['fit_candidates']['32'] if split=='fit' else a6['calibration']['selected'];old_tag='full-lambda-32'
            if prior['weight']!=32:raise ValueError('authenticated A6 lambda32 reference required')
        if weight is None or mode=='independent':
            for q in QPS:assert_same_files(out/'trackers'/f'h264_{q}',a6_trackers/split/old_tag/'trackers'/f'h264_{q}')
        gt_started=time.perf_counter();gt=stage_ground_truth(plans,out/'ground_truth')
        (out/'ground-truth-hashes.json').write_text(json.dumps(gt,indent=2)+'\n')
        metrics=official(package,plans,out,[f'h264_{q}' for q in QPS]);gt_seconds=time.perf_counter()-gt_started
        curve=dict(rate=[float(np.mean(rates[q])) for q in QPS],quality=[metrics[f'h264_{q}']['combined']['HOTA'] for q in QPS])
        if weight is None or mode=='independent':
            if curve!=prior['curves']['h264'] or metrics!={k:prior['metrics'][k] for k in metrics}:raise ValueError('exact A6 H264 curves and combined/per-sequence metrics control failed')
        result=dict(weight=weight,curve=curve,metrics=metrics,selected=selections,selection_counts=dict(counts),log_rate_ratio=float(np.mean(ratios)),
            source_only_decisions_sha256=hashlib.sha256(decision_raw).hexdigest(),sender_replay_seconds=replay_seconds,source_hota_trial_calls=proxy_trials,gt_seconds=gt_seconds)
        replay_cache[key]=result;print('A7_OT_REPLAY',split,tag,result['log_rate_ratio'],flush=True);return result

    fit_anchor=evaluate('fit',None);fits={};candidates=[]
    # These controls establish that the changed loop order preserves all old tracker bytes and metrics.
    fit_control=evaluate('fit',32,'independent')
    for weight in protocol['lambdas']:
        result=evaluate('fit',weight);result['bd']=assess_curve(fit_anchor['curve'],result['curve'],qps=QPS);fits[str(weight)]=result
        candidates.append(dict(weight=weight,quality=result['curve']['quality'],log_rate_ratio=result['log_rate_ratio'],bd_valid=result['bd']['valid']))
    selection=select_valid_fit(fit_anchor['curve'],candidates);chosen=selection['selected'];weight=chosen['weight'] if chosen is not None else None
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),fit_only=True,bank_profiles=21,protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
        policy=dict(kind='source_hota_shared_qp_lagrangian',weight=weight) if weight is not None else dict(kind='anchor'),
        h265_policy='unchanged authenticated A6 lambda32',fit_selection=selection,calibration_reused=True,independent_validation=False,test_scored=False)
    frozen_raw=(json.dumps(frozen,indent=2,allow_nan=False)+'\n').encode();(args.out/'fit-frozen-policy.json').write_bytes(frozen_raw)
    cal_anchor=evaluate('validation',None);cal=evaluate('validation',weight)
    cal['bd']=assess_curve(cal_anchor['curve'],cal['curve'],qps=QPS)
    cal_control=evaluate('validation',32,'independent')
    for control,anchor in ((fit_control,fit_anchor),(cal_control,cal_anchor)):
        control['bd']=assess_curve(anchor['curve'],control['curve'],qps=QPS)
    blocks=dict(grid=True,policy_lock=True,source_provenance=True,paid_header=True,independent_decode=True,source_only=True,no_substitution=True,measurement_reproducibility=True)
    codec_comparisons=dict(h264=cal['bd'],h265=a6['calibration']['selected']['bd']['h265'])
    report=dict(scope='H264 shared-QP FIT-only probe and reusedCAL; H265 delegated unchanged A6; no independent confirmation',protocol=protocol,
        protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),fit_anchor=fit_anchor,fit_candidates=fits,frozen=frozen,frozen_policy_sha256=hashlib.sha256(frozen_raw).hexdigest(),
        calibration=dict(anchor=cal_anchor,selected=cal),controls=dict(fit_independent32=fit_control,validation_independent32=cal_control),
        h265_reference=dict(source='authenticated A6',policy=a6['frozen']['policy'],anchor=a6['calibration']['anchor']['curves']['h265'],
            selected=a6['calibration']['selected']['curves']['h265'],bd=a6['calibration']['selected']['bd']['h265']),
        codec_comparisons=codec_comparisons,blocks=blocks,amended_resource_gate=one_codec_gate(codec_comparisons,blocks),
        exact_a6_anchor_and_independent32_tracker_bytes_and_metrics=True,independent_validation=False,test_scored=False,expanded_evaluation_not_executed=True,
        new_model_inferences=0,new_encodes=0,parent_cached_cost=cached_cost,cache_bundle_sha256=hashlib.sha256(bundle_raw).hexdigest(),full21_rows=len(rows),
        deployment_candidate_encodes_per_window_even_single_qp=84,wall_seconds=time.perf_counter()-started)
    (args.out/'replay-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A7_OT_COMPLETE',json.dumps(dict(policy=frozen['policy'],h264_bd=cal['bd'],amended_gate=report['amended_resource_gate'],test_scored=False,seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':main()
