"""Cloud-only causal replay of one prospectively frozen A4 policy on reused DEV."""
import argparse
from collections import defaultdict,Counter
import configparser
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from ..v34r1_a3.ot_analyze import arrays,official
from ..v34r1_a3.tracking import CausalProxyGuard
from ..v34r1_a3.core import QPS,CODECS
from ..v34r1.policy import select_measured
from ..v34r1.metrics import assess_curve
from ..v32.tracking import load_trackeval,stage_ground_truth,write_mot
from .rd_fit import select_rd_fit
from .cloud_record import assert_same_files

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if ROOT!=Path('/kaggle/temp/adaptive_preprocessing'): raise ValueError('Kaggle-only real tracking/GT evaluation')
    if args.out.exists(): raise ValueError('fresh immutable output required')
    protocol_raw=(ROOT/'configs/v34r1-a4-ot-rd-fit.json').read_bytes(); protocol=json.loads(protocol_raw)
    parent_raw=(args.cache/'dev-report.json').read_bytes(); parent=json.loads(parent_raw)
    if hashlib.sha256(parent_raw).hexdigest()!=protocol['parent_report_sha256']: raise ValueError('parent report drift')
    status=json.loads((args.cache/'cloud-status.json').read_text())
    if not status['complete'] or status['test_scored'] or status['commit']!=protocol['parent_commit']: raise ValueError('invalid parent completion')
    fit=select_rd_fit(parent['results']['fit']['anchor']['curves'],parent['fit']['observed_trailing32']['candidates'])
    if fit!=protocol['fit']: raise ValueError('prospectively frozen FIT objective drift')
    chosen=fit['selected']; expected_policy=dict(kind='observed_trailing32',threshold=chosen['threshold']) if chosen else dict(kind='anchor')
    if protocol['policy']!=expected_policy: raise ValueError('frozen policy does not match FIT choice')
    source_raw=(args.cache/'source-manifest.json').read_bytes()
    if hashlib.sha256(source_raw).hexdigest()!=parent['source_protocol']['source_manifest_sha256']: raise ValueError('source manifest drift')
    manifest=json.loads(source_raw); declared={v['name']:v for v in manifest['sequences'] if v['split']!='reserved_test'}
    rows=[]; sources={}
    for w in (0,1):
        folder=args.cache/f'worker{w}'
        if not (folder/'complete.json').is_file(): raise ValueError('incomplete worker')
        rows.extend(json.loads(gzip.decompress((folder/'rows.json.gz').read_bytes())))
        found=json.loads(gzip.decompress((folder/'sources.json.gz').read_bytes()))
        if set(found)&set(sources): raise ValueError('overlapping worker sequences')
        sources.update(found)
    if set(sources)!=set(declared): raise ValueError('DEV-only source set required')
    groups=defaultdict(list); seen=set()
    for r in rows:
        key=(r['sequence'],r['start'],r['codec'],r['qp'],r['action'])
        if key in seen: raise ValueError('duplicate cached candidate')
        seen.add(key); groups[key[:4]].append(r)
    expected={(n,i,c,q) for n,s in sources.items() for i in range(1,s['frames']+1,16) for c in CODECS for q in QPS}
    if set(groups)!=expected or len(rows)!=772: raise ValueError('incomplete frozen four-QP cache')
    for group in groups.values():
        order=group[0]['actions']
        if order[0]!=0 or len(set(order))!=len(order) or set(order)!={r['action'] for r in group}: raise ValueError('shortlist drift')
        group.sort(key=lambda r:order.index(r['action']))
        for r in group:
            if r['bytes']!=r['elementary_bytes']+r['metadata_bytes'] or min(r['bytes'],r['metadata_bytes'])<=0: raise ValueError('unpaid transport bytes')
            r['detections']=arrays(r['detections'])
    for s in sources.values(): s['source_predictions']=arrays(s['source_predictions'])
    package=load_trackeval(ROOT/'data/TrackEval'); args.out.mkdir(); started=time.perf_counter()

    def evaluate(split,policy):
        names=sorted(n for n,v in sources.items() if v['split']==split)
        tag='anchor' if policy['kind']=='anchor' else 'frozen_a4'; out=args.out/split/tag; out.mkdir(parents=True)
        plans=[]; selections=[]; rates=defaultdict(list); ratios=[]; counts=Counter()
        for name in names:
            s=sources[name]; path=args.sources/name; ini=configparser.ConfigParser(); ini.read(path/'seqinfo.ini'); seq=ini['Sequence']
            if seq['name']!=name or int(seq['frameRate'])!=s['fps'] or int(seq['imWidth'])!=s['width'] or int(seq['imHeight'])!=s['height']:
                raise ValueError('native sequence geometry/cadence drift')
            plans.append(dict(name=name,path=path,frames=s['frames'],full_frames=int(seq['seqLength'])))
            for c in CODECS:
                for q in QPS:
                    guard=CausalProxyGuard(package.metrics.HOTA() if tag!='anchor' else None,fps=s['fps'],lookback=32)
                    frames=[]
                    for offset in range(0,s['frames'],16):
                        group=groups[(name,offset+1,c,q)]; guard.begin(name,s['source_predictions'][offset:offset+16])
                        subset=group[:1] if tag=='anchor' else group; trials=[guard.trial(r['detections']) for r in subset]
                        if tag=='anchor' or trials[0].proxy_costs is None: choice=0
                        else:
                            choice,_=select_measured([r['bytes']/group[0]['bytes'] for r in subset],[t.proxy_costs for t in trials],
                                [True]*3,[policy['threshold']]*3,budget=1.,quality_weight=.05)
                        selected=subset[choice]; trial=trials[choice]; guard.commit(trial); frames.extend(trial.frames)
                        rates[(c,q)].append(selected['bytes']*8/(16*960*960)); ratios.append(math.log(selected['bytes']/group[0]['bytes'])); counts[selected['action']]+=1
                        selections.append(dict(sequence=name,start=offset+1,codec=c,qp=q,action=selected['action'],bytes=selected['bytes'],source_hota=trial.source_metrics))
                    write_mot(out/'trackers'/f'{c}_{q}'/'data'/f'{name}.txt',frames,s['geometry'],s['width'],s['height'])
        # Every sender decision above is complete before the official GT stage.
        gt=stage_ground_truth(plans,out/'ground_truth'); (out/'ground-truth-hashes.json').write_text(json.dumps(gt,indent=2))
        metrics=official(package,plans,out,[f'{c}_{q}' for c in CODECS for q in QPS])
        curves={c:dict(rate=[float(np.mean(rates[(c,q)])) for q in QPS],quality=[metrics[f'{c}_{q}']['combined']['HOTA'] for q in QPS]) for c in CODECS}
        print('A4_OT_REPLAY',split,tag,flush=True)
        return dict(policy=policy,curves=curves,metrics=metrics,selected=selections,selection_counts=dict(counts),log_rate_ratio=float(np.mean(ratios)))

    report=dict(scope=protocol['scope'],protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),protocol=protocol,results={},test_scored=False,
                new_model_inferences=0,new_encodes=0,independent_validation=False)
    for split in ('fit','validation'):
        anchor=evaluate(split,dict(kind='anchor')); selected=evaluate(split,protocol['policy']) if protocol['policy']['kind']!='anchor' else anchor
        if anchor['curves']!=parent['results'][split]['anchor']['curves']: raise ValueError('exact A3 anchor parity failed')
        assert_same_files(args.out/split/'anchor/trackers',args.cache/'tracking'/split/'anchor/trackers')
        if split=='fit' and chosen:
            tag='observed_trailing32-'+str(protocol['policy']['threshold']).replace('.','p')
            assert_same_files(args.out/split/'frozen_a4/trackers',args.cache/'tracking'/split/tag/'trackers')
            quality=[v for c in CODECS for v in selected['curves'][c]['quality']]
            if any(abs(x-y)>1e-12 for x,y in zip(quality,chosen['quality'])) or abs(selected['log_rate_ratio']-chosen['log_rate_ratio'])>1e-12:
                raise ValueError('selected FIT replay differs from frozen A3 candidate')
        selected['bd']={c:assess_curve(anchor['curves'][c],selected['curves'][c],qps=QPS) for c in CODECS}
        report['results'][split]=dict(anchor=anchor,frozen_a4=selected)
    report['calibration_both_codecs_below_minus15']=all(v['point_passes'] for v in report['results']['validation']['frozen_a4']['bd'].values())
    report['exact_parent_tracker_output_and_aggregate_fit_parity']=True; report['wall_seconds']=time.perf_counter()-started
    (args.out/'calibration-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('A4_OT_COMPLETE',json.dumps(dict(gate=report['calibration_both_codecs_below_minus15'],seconds=report['wall_seconds'],test_scored=False)),flush=True)


if __name__=='__main__': main()
